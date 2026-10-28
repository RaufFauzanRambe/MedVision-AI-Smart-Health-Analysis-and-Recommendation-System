"""
training_config.py
==================
Configuration management for the MedVision-AI training module.

A :class:`TrainingConfig` is the single source of truth for any training
run — it carries the model definition, dataset paths, optimizer settings,
distributed topology, reproducibility seeds, and checkpoint policy.
All other modules in ``mlops.training`` consume a ``TrainingConfig``
instance rather than reading raw kwargs, so that the configuration
contract is enforced in exactly one place.

The class supports:

- Loading from YAML (the typical case for shipped configs under
  ``configs/``).
- Programmatic construction with defaults (useful in notebooks and tests).
- Composition by *base + override*, so that ``baseline.yaml`` can be
  extended by ``distributed.yaml`` with only the changed keys.

Example
-------
    from mlops.training.training_config import TrainingConfig

    cfg = TrainingConfig.from_yaml("configs/baseline.yaml")
    cfg.merge_yaml("configs/distributed.yaml")  # apply overrides
    print(cfg.to_dict())
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml


# --------------------------------------------------------------------
# Sub-config dataclasses
# --------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Describes which model architecture to instantiate."""

    name: str = "resnet50"
    params: Dict[str, Any] = field(default_factory=dict)
    pretrained: bool = True
    num_classes: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "pretrained": self.pretrained,
            "num_classes": self.num_classes,
        }


@dataclass
class DatasetSpec:
    """Describes the training/validation/test data sources."""

    train_path: str = "data/processed/train"
    val_path: str = "data/processed/val"
    test_path: str = "data/processed/test"
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = True
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    channels_first: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "augment": self.augment,
            "image_size": list(self.image_size),
            "channels_first": self.channels_first,
        }


@dataclass
class OptimizerSpec:
    """Optimizer + LR scheduler configuration."""

    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # used only for SGD-family optimizers
    scheduler: str = "cosine"  # none|cosine|step|exponential
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_epochs: int = 0
    grad_clip: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "scheduler": self.scheduler,
            "scheduler_params": dict(self.scheduler_params),
            "warmup_epochs": self.warmup_epochs,
            "grad_clip": self.grad_clip,
        }


@dataclass
class DistributedSpec:
    """Distributed training topology."""

    strategy: str = "single"  # single|ddp|fsdp|horovod
    world_size: int = 1
    backend: str = "nccl"  # nccl|gloo|mpi
    local_rank: int = 0
    sync_bn: bool = False
    find_unused_params: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "world_size": self.world_size,
            "backend": self.backend,
            "local_rank": self.local_rank,
            "sync_bn": self.sync_bn,
            "find_unused_params": self.find_unused_params,
        }


@dataclass
class CheckpointSpec:
    """Checkpointing policy for a training run."""

    dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # min|max
    resume_from: Optional[str] = None
    save_optimizer_state: bool = True
    save_metadata: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dir": self.dir,
            "save_every_n_epochs": self.save_every_n_epochs,
            "save_top_k": self.save_top_k,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "resume_from": self.resume_from,
            "save_optimizer_state": self.save_optimizer_state,
            "save_metadata": self.save_metadata,
        }


@dataclass
class ReproducibilitySpec:
    """Seeds and deterministic-mode toggles."""

    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    python_hash_seed: str = "0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "python_hash_seed": self.python_hash_seed,
        }


# --------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Top-level training configuration.

    Construct programmatically or from YAML. Use :meth:`merge_yaml` to
    layer overrides on top of the current values.
    """

    experiment_name: str = "medvision_default"
    run_name: Optional[str] = None
    epochs: int = 30
    max_steps: Optional[int] = None
    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    early_stopping_patience: int = 0  # 0 = disabled
    mixed_precision: str = "fp16"  # none|fp16|bf16
    gradient_accumulation_steps: int = 1

    model: ModelSpec = field(default_factory=ModelSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    distributed: DistributedSpec = field(default_factory=DistributedSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)
    reproducibility: ReproducibilitySpec = field(default_factory=ReproducibilitySpec)

    extras: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Build a config from a plain dictionary, ignoring unknown keys."""
        known_top = {k for k in cls.__dataclass_fields__ if k != "extras"}
        cfg_kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            if k in known_top:
                cfg_kwargs[k] = v
            else:
                cfg_kwargs.setdefault("extras", {})[k] = v
        # Sub-configs
        if "model" in cfg_kwargs and isinstance(cfg_kwargs["model"], Mapping):
            cfg_kwargs["model"] = ModelSpec(**cfg_kwargs["model"])
        if "dataset" in cfg_kwargs and isinstance(cfg_kwargs["dataset"], Mapping):
            cfg_kwargs["dataset"] = DatasetSpec(**cfg_kwargs["dataset"])
        if "optimizer" in cfg_kwargs and isinstance(cfg_kwargs["optimizer"], Mapping):
            cfg_kwargs["optimizer"] = OptimizerSpec(**cfg_kwargs["optimizer"])
        if "distributed" in cfg_kwargs and isinstance(cfg_kwargs["distributed"], Mapping):
            cfg_kwargs["distributed"] = DistributedSpec(**cfg_kwargs["distributed"])
        if "checkpoint" in cfg_kwargs and isinstance(cfg_kwargs["checkpoint"], Mapping):
            cfg_kwargs["checkpoint"] = CheckpointSpec(**cfg_kwargs["checkpoint"])
        if "reproducibility" in cfg_kwargs and isinstance(
            cfg_kwargs["reproducibility"], Mapping
        ):
            cfg_kwargs["reproducibility"] = ReproducibilitySpec(
                **cfg_kwargs["reproducibility"]
            )
        return cls(**cfg_kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Training config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Merge / override
    # ------------------------------------------------------------------

    def merge_yaml(self, path: Union[str, Path]) -> "TrainingConfig":
        """Apply a YAML override file on top of the current config (in place).

        Only keys explicitly present in the override file are modified.
        Sub-configs are merged recursively (override wins on conflicts).
        Returns ``self`` for chaining.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}

        merged = self.to_dict()
        merged = _deep_merge(merged, override)
        new_cfg = TrainingConfig.from_dict(merged)
        # Preserve the original `extras` if not overwritten
        new_cfg.extras = {**self.extras, **(override.get("extras", {}))}
        # Replace fields in place
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(new_cfg, f))
        return self

    def merge_dict(self, override: Mapping[str, Any]) -> "TrainingConfig":
        """Like :meth:`merge_yaml` but accepts an in-memory dict."""
        merged = _deep_merge(self.to_dict(), override)
        new_cfg = TrainingConfig.from_dict(merged)
        new_cfg.extras = {**self.extras, **(override.get("extras", {}))}
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(new_cfg, f))
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "eval_every_n_epochs": self.eval_every_n_epochs,
            "log_every_n_steps": self.log_every_n_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "distributed": self.distributed.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "extras": copy.deepcopy(self.extras),
        }

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write this config to a YAML file (for snapshotting at run time)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def resolve_paths(self, base: Union[str, Path]) -> "TrainingConfig":
        """Resolve relative paths (dataset/checkpoint dirs) against ``base``."""
        base = Path(base)
        self.dataset.train_path = str((base / self.dataset.train_path).resolve())
        self.dataset.val_path = str((base / self.dataset.val_path).resolve())
        self.dataset.test_path = str((base / self.dataset.test_path).resolve())
        self.checkpoint.dir = str((base / self.checkpoint.dir).resolve())
        return self

    def env_overrides(self) -> Dict[str, str]:
        """Return a dict of environment-variable overrides for this config."""
        env = {
            "MEDVISION_EXPERIMENT_NAME": self.experiment_name,
            "MEDVISION_SEED": str(self.reproducibility.seed),
            "MEDVISION_WORLD_SIZE": str(self.distributed.world_size),
            "MEDVISION_LOCAL_RANK": str(self.distributed.local_rank),
            "MEDVISION_BACKEND": self.distributed.backend,
        }
        if self.run_name:
            env["MEDVISION_RUN_NAME"] = self.run_name
        return env


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


__all__ = [
    "TrainingConfig",
    "ModelSpec",
    "DatasetSpec",
    "OptimizerSpec",
    "DistributedSpec",
    "CheckpointSpec",
    "ReproducibilitySpec",
]
