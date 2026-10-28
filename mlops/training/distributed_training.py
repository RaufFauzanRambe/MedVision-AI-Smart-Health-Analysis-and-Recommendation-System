"""
distributed_training.py
========================
Distributed training helpers for the MedVision-AI training module.

Supports three strategies out of the box:

- **single** — single-GPU / CPU; no process group.
- **ddp** — ``torch.nn.parallel.DistributedDataParallel`` over a
  single-node multi-GPU topology, using a ``torch.distributed``
  process group (``nccl`` for GPU, ``gloo`` for CPU).
- **fsdp** — ``torch.distributed.fsdp.FullyShardedDataParallel``,
  used when the model is too large for a single GPU.

The :class:`DistributedContext` is the single entry point: callers
construct it from a :class:`DistributedSpec`, and the context exposes
``setup()``, ``cleanup()``, ``wrap_model(model)``, and ``device``.
This keeps the rest of the training module strategy-agnostic.

For strategies that require spawning one process per rank (``ddp``
single-node, multi-GPU), use :func:`spawn_workers` which wraps
``torch.multiprocessing.spawn``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .training_config import DistributedSpec

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Distributed context
# --------------------------------------------------------------------


@dataclass
class DistributedContext:
    """Manage the lifetime of a torch.distributed process group.

    The class is intentionally minimal: it owns the process group
    lifecycle (``setup`` / ``cleanup``) and exposes a single
    ``wrap_model`` method that picks the right DDP/FSDP wrapper for
    the configured strategy.
    """

    spec: DistributedSpec
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.spec.strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError(
                f"Unknown distributed strategy {self.spec.strategy!r}; "
                "expected one of: single, ddp, fsdp"
            )
        if self.spec.strategy != "single":
            self.rank = int(os.environ.get("RANK", self.spec.local_rank))
            self.world_size = int(os.environ.get("WORLD_SIZE", self.spec.world_size))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize the distributed process group (no-op for ``single``)."""
        if self.spec.strategy == "single":
            return
        try:
            import torch.distributed as dist
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("distributed setup requires torch") from exc
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this build")
        if dist.is_initialized():
            logger.warning("Distributed process group already initialized; skipping setup()")
            return
        dist.init_process_group(
            backend=self.spec.backend,
            world_size=self.world_size,
            rank=self.rank,
        )
        logger.info(
            "Initialized process group: strategy=%s backend=%s rank=%d world_size=%d",
            self.spec.strategy, self.spec.backend, self.rank, self.world_size,
        )

    def cleanup(self) -> None:
        """Tear down the process group. Safe to call multiple times."""
        if self.spec.strategy == "single":
            return
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except ImportError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model: Any) -> Any:
        """Wrap ``model`` with the appropriate distributed wrapper."""
        if self.spec.strategy == "single":
            return model
        try:
            import torch
            import torch.distributed as dist
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("wrap_model requires torch") from exc

        if self.spec.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.spec.strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            device_id = self._local_device_id()
            return DDP(
                model,
                device_ids=[device_id] if device_id is not None else None,
                find_unused_parameters=self.spec.find_unused_params,
            )

        if self.spec.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            return FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self._local_device_id(),
            )

        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        if self.spec.strategy == "single":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                return "cpu"
        return f"cuda:{self._local_device_id()}"

    def _local_device_id(self) -> Optional[int]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            local_rank = int(os.environ.get("LOCAL_RANK", self.spec.local_rank))
            return local_rank
        except ImportError:  # pragma: no cover
            return None


# --------------------------------------------------------------------
# Worker spawning
# --------------------------------------------------------------------


def spawn_workers(
    fn: Callable[[int, "DistributedContext"], None],
    world_size: int,
    spec: DistributedSpec,
    backend: str = "nccl",
) -> None:
    """Spawn one process per rank and call ``fn(rank, ctx)`` in each.

    Used for single-node, multi-GPU training. For multi-node
    deployments, launch via ``torchrun`` and have each rank construct
    its own :class:`DistributedContext` from the env.
    """
    if spec.strategy == "single":
        fn(0, DistributedContext(spec=DistributedSpec(strategy="single")))
        return
    try:
        import torch.multiprocessing as mp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("spawn_workers requires torch") from exc

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    mp.spawn(
        _spawn_worker_entrypoint,
        args=(fn, world_size, backend, spec),
        nprocs=world_size,
        join=True,
    )


def _spawn_worker_entrypoint(
    rank: int,
    fn: Callable[[int, "DistributedContext"], None],
    world_size: int,
    backend: str,
    base_spec: DistributedSpec,
) -> None:
    """Entry-point for ``mp.spawn``: configure env, build context, call ``fn``."""
    import os as _os

    _os.environ["RANK"] = str(rank)
    _os.environ["WORLD_SIZE"] = str(world_size)
    spec = DistributedSpec(
        strategy=base_spec.strategy,
        world_size=world_size,
        backend=backend,
        local_rank=rank,
        sync_bn=base_spec.sync_bn,
        find_unused_params=base_spec.find_unused_params,
    )
    ctx = DistributedContext(spec=spec, rank=rank, world_size=world_size)
    ctx.setup()
    try:
        fn(rank, ctx)
    finally:
        ctx.cleanup()


@contextmanager
def distributed(spec: DistributedSpec):
    """Context manager wrapping :class:`DistributedContext` setup/cleanup."""
    ctx = DistributedContext(spec=spec)
    ctx.setup()
    try:
        yield ctx
    finally:
        ctx.cleanup()


__all__ = [
    "DistributedContext",
    "spawn_workers",
    "distributed",
]
