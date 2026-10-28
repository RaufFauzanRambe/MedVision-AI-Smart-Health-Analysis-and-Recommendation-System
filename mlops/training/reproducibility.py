"""
reproducibility.py
===================
Reproducibility helpers for the MedVision-AI training module.

The module is responsible for:

  * Seeding Python's ``random``, NumPy, and (when available) PyTorch
    RNGs from a single source so that any given seed reproduces
    identical runs.
  * Switching cuDNN to deterministic / non-benchmark mode to trade a
    little speed for bit-stable convolutions.
  * Setting ``PYTHONHASHSEED`` so that dictionary iteration order is
    stable across processes (important for distributed pipelines that
    shard work by hash).
  * Returning a structured :class:`ReproducibilityState` snapshot so
    that the experiment logger can stamp the run with the actual
    settings that were applied (not just the intended ones).

Use it early in any training script:

    from mlops.training.reproducibility import seed_everything

    seed_everything(seed=42, deterministic=True, cudnn_benchmark=False)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReproducibilityState:
    """Snapshot of the reproducibility settings actually applied."""

    seed: int
    deterministic: bool
    cudnn_benchmark: bool
    python_hash_seed: str
    torch_deterministic: Optional[bool] = None

    def to_dict(self):
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "python_hash_seed": self.python_hash_seed,
            "torch_deterministic": self.torch_deterministic,
        }


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    python_hash_seed: Optional[str] = None,
) -> ReproducibilityState:
    """Seed every RNG used by the training stack.

    Parameters
    ----------
    seed:
        Master seed. Must be a non-negative integer.
    deterministic:
        When True and torch is available, sets
        ``torch.use_deterministic_algorithms(True)`` and
        ``torch.backends.cudnn.deterministic = True``.
    cudnn_benchmark:
        When False and torch is available, sets
        ``torch.backends.cudnn.benchmark = False`` to disable
        cuDNN's autotuner (which is non-deterministic across
        input shapes).
    python_hash_seed:
        When provided, sets ``PYTHONHASHSEED`` in the current
        process. The default behavior is to set it to the string
        form of ``seed``. Use the string ``"random"`` to opt out
        of deterministic hashing (not recommended for training
        reproducibility).

    Returns
    -------
    ReproducibilityState
        Snapshot of the values actually applied.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Python
    random.seed(seed)
    if python_hash_seed is None:
        python_hash_seed = str(seed)
    os.environ["PYTHONHASHSEED"] = python_hash_seed

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    # PyTorch (optional)
    torch_deterministic: Optional[bool] = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                # Older torch versions
                pass
            torch_deterministic = True
        else:
            torch_deterministic = False
    except ImportError:
        # torch is optional at config-validation time
        pass

    return ReproducibilityState(
        seed=seed,
        deterministic=deterministic,
        cudnn_benchmark=cudnn_benchmark,
        python_hash_seed=python_hash_seed,
        torch_deterministic=torch_deterministic,
    )


def seed_worker(worker_id: int, base_seed: int = 42) -> None:
    """Seed a torch.utils.data.DataLoader worker.

    PyTorch spawns DataLoader workers with a different seed each; for
    reproducibility, ``worker_init_fn`` should call this with the
    base seed used by the training process.
    """
    try:
        import numpy as np
        import torch
        import random as _random

        # torch's seeding for worker processes
        seed = base_seed + worker_id
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


__all__ = [
    "ReproducibilityState",
    "seed_everything",
    "seed_worker",
]
