"""
checkpoint_manager.py
======================
Checkpoint save / load / resume manager for the MedVision-AI training module.

A checkpoint contains everything needed to resume training from exactly
where it left off: model weights, optimizer & scheduler state, current
epoch & global step, the :class:`TrainingConfig` snapshot, and any
user-provided extra metadata (e.g. MLflow run ID).

The manager enforces a *best-K* retention policy: only the top-K models
(by a configurable monitored metric, with min/max mode) are kept on
disk; the rest are pruned. Periodic checkpoints (every N epochs) are
also kept but rotate independently.

Example
-------
    from mlops.training.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager(
        checkpoint_dir="checkpoints",
        monitor_metric="val_loss",
        monitor_mode="min",
        save_top_k=3,
    )
    path = mgr.save(
        epoch=10,
        global_step=1000,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"val_loss": 0.42, "val_accuracy": 0.91},
        config=cfg.to_dict(),
    )
    state = CheckpointManager.load(path, model=model, optimizer=optimizer)
    # state.epoch == 10, state.global_step == 1000
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# --------------------------------------------------------------------
# State container
# --------------------------------------------------------------------


@dataclass
class CheckpointState:
    """In-memory representation of a loaded checkpoint."""

    epoch: int
    global_step: int
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": dict(self.metrics),
            "config": self.config,
            "extras": dict(self.extras),
            "path": self.path,
        }


# --------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------


@dataclass
class CheckpointManager:
    """Save, prune, and load training checkpoints.

    Attributes
    ----------
    checkpoint_dir:
        Directory where checkpoints are written.
    monitor_metric:
        Metric used to rank checkpoints for top-K retention.
    monitor_mode:
        ``"min"`` or ``"max"`` — defines whether lower or higher is
        better for ``monitor_metric``.
    save_top_k:
        Maximum number of best checkpoints to keep on disk. Older
        non-best checkpoints are pruned after each save.
    save_every_n_epochs:
        Also retain a rotating set of periodic checkpoints (every
        N epochs), independently of the top-K best.
    keep_last_n_periodic:
        How many periodic checkpoints to keep on disk.
    """

    checkpoint_dir: str = "checkpoints"
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    save_top_k: int = 3
    save_every_n_epochs: int = 1
    keep_last_n_periodic: int = 1

    def __post_init__(self) -> None:
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")
        if self.save_top_k < 1:
            raise ValueError(f"save_top_k must be >= 1, got {self.save_top_k}")
        self._dir = Path(self.checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        epoch: int,
        global_step: int,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a checkpoint and prune old ones.

        Returns the path to the freshly written checkpoint.
        """
        metrics = metrics or {}
        extras = extras or {}

        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": config,
            "extras": extras,
        }
        # Serialize model & optimizer state via torch when available
        try:
            import torch

            if hasattr(model, "state_dict"):
                payload["model_state_dict"] = model.state_dict()
            if optimizer is not None and hasattr(optimizer, "state_dict"):
                payload["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                payload["scheduler_state_dict"] = scheduler.state_dict()
        except ImportError:  # pragma: no cover
            pass

        score = metrics.get(self.monitor_metric)
        is_best = False
        if score is not None and self._is_best_so_far(score):
            is_best = True
            self._best_score = score

        fname = self._filename(epoch=epoch, global_step=global_step, is_best=is_best)
        path = self._dir / fname
        self._save_payload(payload, path)
        self._save_metadata(path, payload, is_best=is_best)

        # Prune top-K best
        self._prune_best()
        # Prune periodic checkpoints
        if epoch % self.save_every_n_epochs == 0:
            self._rotate_periodic(path)
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        path: Union[str, Path],
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        map_location: Any = None,
    ) -> CheckpointState:
        """Load a checkpoint from disk and optionally restore model/optim state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = CheckpointManager._load_payload(path, map_location=map_location)
        if model is not None and "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return CheckpointState(
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", 0)),
            metrics=payload.get("metrics", {}),
            config=payload.get("config"),
            extras=payload.get("extras", {}),
            path=str(path),
        )

    def latest(self) -> Optional[Path]:
        """Return the path of the most recent checkpoint, or ``None``."""
        candidates = sorted(self._dir.glob("ckpt_epoch_*.pt*"))
        if not candidates:
            return None
        return candidates[-1]

    def best(self) -> Optional[Path]:
        """Return the path of the best-so-far checkpoint."""
        best_files = list(self._dir.glob("ckpt_epoch_*_best.*"))
        if not best_files:
            return None
        return best_files[-1]

    def list_all(self) -> List[Path]:
        """Return all checkpoint paths sorted by epoch."""
        return sorted(self._dir.glob("ckpt_epoch_*.pt*"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    _best_score: Optional[float] = None

    def _is_best_so_far(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.monitor_mode == "min":
            return score < self._best_score
        return score > self._best_score

    def _filename(self, epoch: int, global_step: int, is_best: bool) -> str:
        suffix = "best" if is_best else f"step{global_step}"
        return f"ckpt_epoch_{epoch:04d}_{suffix}.pt"

    def _save_payload(self, payload: Dict[str, Any], path: Path) -> None:
        try:
            import torch

            torch.save(payload, path)
            return
        except ImportError:  # pragma: no cover
            pass
        # Fallback: write a JSON-only checkpoint (useful for tests
        # that don't have torch installed).
        serializable = {k: v for k, v in payload.items() if "_state_dict" not in k}
        with path.with_suffix(".json").open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, default=str)

    @staticmethod
    def _load_payload(path: Path, map_location: Any = None) -> Dict[str, Any]:
        try:
            import torch

            return torch.load(path, map_location=map_location, weights_only=False)
        except ImportError:
            with path.with_suffix(".json").open("r", encoding="utf-8") as fh:
                return json.load(fh)

    def _save_metadata(self, path: Path, payload: Dict[str, Any], is_best: bool) -> None:
        meta_path = path.with_suffix(".meta.json")
        meta = {
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "metrics": payload["metrics"],
            "is_best": is_best,
            "created_at_epoch": payload["epoch"],
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def _prune_best(self) -> None:
        """Keep only the top-K best checkpoints (by monitor_metric)."""
        metas = self._load_all_metas()
        if len(metas) <= self.save_top_k:
            return
        metas.sort(
            key=lambda m: m["metrics"].get(self.monitor_metric, float("inf")),
            reverse=(self.monitor_mode == "max"),
        )
        # Best come first; prune the tail
        for m in metas[self.save_top_k:]:
            ckpt_path = self._dir / m["path"]
            ckpt_path.unlink(missing_ok=True)
            (ckpt_path.with_suffix(".meta.json")).unlink(missing_ok=True)

    def _rotate_periodic(self, current: Path) -> None:
        """Keep only ``keep_last_n_periodic`` non-best periodic checkpoints."""
        periodic = [p for p in self._dir.glob("ckpt_epoch_*_step*.pt")]
        periodic.sort(key=lambda p: p.name)
        while len(periodic) > self.keep_last_n_periodic:
            victim = periodic.pop(0)
            victim.unlink(missing_ok=True)
            (victim.with_suffix(".meta.json")).unlink(missing_ok=True)

    def _load_all_metas(self) -> List[Dict[str, Any]]:
        metas: List[Dict[str, Any]] = []
        for meta_path in self._dir.glob("ckpt_epoch_*.meta.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                data["path"] = meta_path.stem.replace(".meta", "") + ".pt"
                # Coerce to actual file stem
                data["path"] = meta_path.name.replace(".meta.json", ".pt")
                metas.append(data)
            except (OSError, json.JSONDecodeError):
                continue
        return metas


__all__ = ["CheckpointState", "CheckpointManager"]
