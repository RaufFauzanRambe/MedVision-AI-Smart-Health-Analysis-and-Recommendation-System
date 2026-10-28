"""
experiment_logger.py
====================
Context-managed MLflow experiment tracking for MedVision-AI.

The :class:`ExperimentLogger` is the primary entry point for tracking
model training and evaluation runs. It wraps the bookkeeping that
every run needs — loading config, configuring MLflow, applying default
tags, starting and ending the run cleanly — so that downstream
training scripts can focus on model code.

Example
-------
    from mlops.tracking.experiment_logger import ExperimentLogger

    with ExperimentLogger(
        experiment="medvision-classification",
        run_name="baseline_v1",
        tags={"model": "resnet50", "dataset": "chest_xray"},
    ) as run:
        run.log_params({"lr": 1e-4, "epochs": 30})
        run.log_metric("accuracy", 0.91)
        run.log_artifact("model.pt", artifact_path="models")

The class can also be used without a context manager:

    run = ExperimentLogger(...).start()
    try:
        ...
    finally:
        run.end()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .mlflow_config import MLflowConfig, get_mlflow_config
from .tracking_utils import collect_environment_tags, flatten_tags, generate_run_name

logger = logging.getLogger(__name__)


@dataclass
class ExperimentLogger:
    """Context-managed MLflow run orchestrator.

    Attributes
    ----------
    experiment:
        MLflow experiment name. Defaults to the value in
        ``tracking.yaml``.
    run_name:
        Optional run name. If omitted, a timestamped name is
        generated using ``run_name_prefix``.
    tags:
        User-supplied tags. Merged with environment tags before the
        run starts.
    autolog:
        If True, attempt ``mlflow.<framework>.autolog()`` for known
        frameworks. Currently autologs for ``sklearn`` and
        ``pytorch`` when those packages are importable.
    config:
        Optional override for the MLflow configuration. When not
        provided, the singleton from :func:`get_mlflow_config` is
        used.
    nested:
        If True, run is created as nested inside the currently active
        run (useful for hyperparameter sweeps).
    """

    experiment: Optional[str] = None
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    autolog: bool = False
    config: Optional[MLflowConfig] = None
    nested: bool = False

    _active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._config = self.config or get_mlflow_config()

    def start(self) -> "ExperimentLogger":
        """Begin the MLflow run. Idempotent per instance."""
        if self._active:
            logger.warning("ExperimentLogger.start() called while already active; ignoring.")
            return self
        import mlflow

        self._config.setup()
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        run_name = self.run_name or generate_run_name(self._config.run_name_prefix)
        merged_tags = {**collect_environment_tags(), **flatten_tags(self.tags)}
        if self.autolog:
            self._maybe_autolog()
        mlflow.start_run(
            run_name=run_name,
            tags=merged_tags,
            nested=self.nested,
        )
        self._active = True
        logger.info("Started MLflow run: %s (experiment=%s)", run_name, self.experiment or self._config.experiment_name)
        return self

    def end(self, status: str = "FINISHED") -> None:
        """End the active MLflow run. No-op if not active."""
        if not self._active:
            return
        import mlflow

        try:
            mlflow.end_run(status=status)
        finally:
            self._active = False
        logger.info("Ended MLflow run with status %s", status)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of parameters."""
        import mlflow

        flat = flatten_tags(params)
        for k, v in flat.items():
            mlflow.log_param(k, v)

    def log_tags(self, tags: Dict[str, Any]) -> None:
        """Add tags to the active run (merged with environment tags)."""
        import mlflow

        mlflow.set_tags(flatten_tags(tags))

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric (delegated to :mod:`metrics_logger`)."""
        from .metrics_logger import log_metric

        log_metric(name, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a batch of metrics at the same step."""
        from .metrics_logger import log_metrics

        log_metrics(metrics, step=step)

    def log_artifact(self, source, artifact_path: Optional[str] = None, name: Optional[str] = None) -> None:
        """Log a single artifact (delegated to :mod:`artifact_logger`)."""
        from .artifact_logger import log_artifact

        log_artifact(source=source, artifact_path=artifact_path, name=name)

    def log_artifacts(self, items) -> None:
        """Log multiple artifacts at once (see :func:`artifact_logger.log_artifacts_batch`)."""
        from .artifact_logger import log_artifacts_batch

        log_artifacts_batch(items)

    def _maybe_autolog(self) -> None:
        import importlib

        try:
            sklearn = importlib.import_module("sklearn")
            import mlflow.sklearn

            mlflow.sklearn.autolog()
        except ImportError:
            pass
        try:
            importlib.import_module("torch")
            import mlflow.pytorch

            mlflow.pytorch.autolog()
        except ImportError:
            pass

    # Context manager protocol ----------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "FINISHED" if exc_type is None else "FAILED"
        self.end(status=status)


@contextmanager
def log_run(
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    autolog: bool = False,
) -> Iterator[ExperimentLogger]:
    """Functional shortcut for the :class:`ExperimentLogger` context manager."""
    with ExperimentLogger(
        experiment=experiment,
        run_name=run_name,
        tags=tags or {},
        autolog=autolog,
    ) as run:
        yield run


__all__ = ["ExperimentLogger", "log_run"]
