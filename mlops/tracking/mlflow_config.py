"""
mlflow_config.py
================
Centralized MLflow configuration for the MedVision-AI project.

This module is responsible for:
  * Loading tracking configuration from YAML files.
  * Initializing the MLflow tracking URI, experiment name, and registry URI.
  * Providing a singleton-style accessor so that all modules share the
    same MLflow client and experiment context.

Typical usage:

    from mlops.tracking.mlflow_config import get_mlflow_config

    cfg = get_mlflow_config()
    cfg.setup()  # configures mlflow.set_tracking_uri, set_experiment, etc.

The configuration file location can be overridden via the
``MEDVISION_TRACKING_CONFIG`` environment variable. By default the module
looks for ``configs/tracking.yaml`` relative to this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    import mlflow
except ImportError:  # pragma: no cover - mlflow is optional at import time
    mlflow = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "tracking.yaml"
ENV_CONFIG_PATH = os.environ.get("MEDVISION_TRACKING_CONFIG")


@dataclass
class MLflowConfig:
    """Runtime representation of the tracking configuration.

    Attributes
    ----------
    tracking_uri:
        Where MLflow should store runs. May be a local path
        (``file:///...``), an HTTP/HTTPS tracking server URL, or a
        Databricks profile.
    registry_uri:
        Optional model registry URI. Defaults to ``tracking_uri`` when
        not provided.
    experiment_name:
        Default experiment name used when callers do not specify one.
    run_name_prefix:
        Optional prefix applied to every run created through
        ``experiment_logger``.
    tags:
        Default tags applied to every run. Useful for stamping
        environment info (e.g. ``git_branch``, ``host``).
    autolog:
        Whether to enable ``mlflow.<framework>.autolog()`` automatically.
        Currently only a boolean toggle; framework-specific autolog is
        invoked inside ``experiment_logger``.
    artifacts_dir:
        Local directory used as a staging area before artifacts are
        logged to MLflow.
    """

    tracking_uri: str = "file:///tmp/mlruns"
    registry_uri: Optional[str] = None
    experiment_name: str = "medvision_default"
    run_name_prefix: str = "medvision"
    tags: Dict[str, str] = field(default_factory=dict)
    autolog: bool = False
    artifacts_dir: str = "./mlartifacts"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLflowConfig":
        """Build a :class:`MLflowConfig` from a plain dictionary."""
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLflowConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tracking config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config back to a plain dict (for logging)."""
        return {
            "tracking_uri": self.tracking_uri,
            "registry_uri": self.registry_uri,
            "experiment_name": self.experiment_name,
            "run_name_prefix": self.run_name_prefix,
            "tags": dict(self.tags),
            "autolog": self.autolog,
            "artifacts_dir": self.artifacts_dir,
        }

    def setup(self) -> None:
        """Apply this configuration to the global MLflow runtime.

        Raises
        ------
        RuntimeError
            If ``mlflow`` is not installed.
        """
        if mlflow is None:  # pragma: no cover
            raise RuntimeError(
                "mlflow is not installed. Install with `pip install mlflow`."
            )
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_experiment(self.experiment_name)
        os.makedirs(self.artifacts_dir, exist_ok=True)


_CONFIG_SINGLETON: Optional[MLflowConfig] = None


def get_mlflow_config(config_path: Optional[str | Path] = None) -> MLflowConfig:
    """Return the shared :class:`MLflowConfig` instance.

    The first call loads the configuration from disk (using either the
    provided path, the ``MEDVISION_TRACKING_CONFIG`` environment
    variable, or the default ``configs/tracking.yaml``). Subsequent
    calls return the cached singleton so that all modules see the same
    MLflow settings.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML configuration file. When
        provided, the singleton is refreshed using this file.
    """
    global _CONFIG_SINGLETON
    if config_path is not None or _CONFIG_SINGLETON is None:
        path = Path(config_path or ENV_CONFIG_PATH or DEFAULT_CONFIG_PATH)
        _CONFIG_SINGLETON = MLflowConfig.from_yaml(path)
    return _CONFIG_SINGLETON


def reset_mlflow_config() -> None:
    """Drop the cached singleton. Primarily useful for unit tests."""
    global _CONFIG_SINGLETON
    _CONFIG_SINGLETON = None


__all__ = [
    "MLflowConfig",
    "get_mlflow_config",
    "reset_mlflow_config",
    "DEFAULT_CONFIG_PATH",
    "ENV_CONFIG_PATH",
]
