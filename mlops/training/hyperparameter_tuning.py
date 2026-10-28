"""
hyperparameter_tuning.py
=========================
Hyperparameter search orchestration for MedVision-AI.

The :class:`HyperparameterTuner` wraps Optuna (when available) and
provides:

1. A small set of **suggestion templates** for common hyperparameters
   (learning rate, batch size, optimizer, dropout, model depth, ...)
   so users do not have to handcraft a search space from scratch.
2. A single :meth:`run` entrypoint that takes a
   ``train_fn(trial, params) -> metrics`` callable and returns a
   structured :class:`TuningReport`.
3. Integration with the MLOps tracking stack: each trial is logged as
   a nested MLflow run when ``log_to_mlflow=True``.

When optuna is not installed, the tuner falls back to a random search
over the configured space — useful for smoke tests on CI where adding
optuna is undesirable.
"""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .training_config import TrainingConfig

logger = logging.getLogger(__name__)


TrainFn = Callable[[Any, Mapping[str, Any]], Dict[str, float]]


# --------------------------------------------------------------------
# Report
# --------------------------------------------------------------------


@dataclass
class TrialResult:
    """One trial of a hyperparameter search."""

    number: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    state: str = "COMPLETE"  # COMPLETE|PRUNED|FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "state": self.state,
        }


@dataclass
class TuningReport:
    """Aggregated result of a hyperparameter search."""

    trials: List[TrialResult] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_trial_number: int = -1
    n_complete: int = 0
    n_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trials": [t.to_dict() for t in self.trials],
            "best_params": dict(self.best_params),
            "best_metrics": dict(self.best_metrics),
            "best_trial_number": self.best_trial_number,
            "n_complete": self.n_complete,
            "n_failed": self.n_failed,
        }


# --------------------------------------------------------------------
# Suggestion templates
# --------------------------------------------------------------------


def suggest_learning_rate(trial: Any, low: float = 1e-5, high: float = 1e-2) -> float:
    """Log-uniform learning-rate suggestion."""
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float("lr", low, high, log=True)
    return 10 ** random.uniform(_log10(low), _log10(high))


def suggest_batch_size(trial: Any, options: Optional[List[int]] = None) -> int:
    """Categorical batch-size suggestion (powers of 2)."""
    options = options or [16, 32, 64, 128]
    return _suggest_categorical(trial, "batch_size", options)


def suggest_optimizer(trial: Any, options: Optional[List[str]] = None) -> str:
    options = options or ["adam", "sgd", "adamw"]
    return _suggest_categorical(trial, "optimizer", options)


def suggest_dropout(trial: Any, low: float = 0.0, high: float = 0.5) -> float:
    """Uniform dropout suggestion."""
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float("dropout", low, high)
    return random.uniform(low, high)


def suggest_weight_decay(trial: Any, low: float = 1e-6, high: float = 1e-2) -> float:
    """Log-uniform weight-decay suggestion."""
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float("weight_decay", low, high, log=True)
    return 10 ** random.uniform(_log10(low), _log10(high))


# --------------------------------------------------------------------
# Tuner
# --------------------------------------------------------------------


@dataclass
class HyperparameterTuner:
    """Orchestrate a hyperparameter search.

    Attributes
    ----------
    n_trials:
        Total number of trials to run.
    base_config:
        :class:`TrainingConfig` to clone for each trial. Trial-specific
        hyperparameters are merged on top via :meth:`merge_dict`.
    direction:
        ``"maximize"`` or ``"minimize"``.
    monitor_metric:
        Metric key in the trial result dict to optimize.
    sampler:
        Search algorithm. ``"tpe"`` (Tree-structured Parzen Estimator,
        Optuna default) or ``"random"``.
    log_to_mlflow:
        If True, each trial is logged as a nested MLflow run via
        :class:`mlops.tracking.experiment_logger.ExperimentLogger`.
    pruner:
        Optuna pruner name; ``"none"`` (default) disables pruning.
    """

    n_trials: int = 20
    base_config: Optional[TrainingConfig] = None
    direction: str = "maximize"
    monitor_metric: str = "val_accuracy"
    sampler: str = "tpe"
    log_to_mlflow: bool = False
    pruner: str = "none"

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {self.direction!r}"
            )
        if self.sampler not in ("tpe", "random"):
            raise ValueError(
                f"sampler must be 'tpe' or 'random', got {self.sampler!r}"
            )

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------

    def run(self, search_fn: Callable[[Any], Dict[str, Any]], train_fn: TrainFn) -> TuningReport:
        """Run ``n_trials`` trials of ``train_fn`` with params from ``search_fn``.

        Parameters
        ----------
        search_fn:
            A callable that takes an Optuna trial (or our fallback stub)
            and returns a params dict.
        train_fn:
            A callable ``train_fn(trial, params) -> metrics`` that runs
            one training trial and returns the evaluation metrics.

        Returns
        -------
        TuningReport
        """
        if self._optuna_available() and self.sampler == "tpe":
            return self._run_optuna(search_fn, train_fn)
        # Random-search fallback
        return self._run_random(search_fn, train_fn)

    # ------------------------------------------------------------------
    # Optuna path
    # ------------------------------------------------------------------

    def _run_optuna(self, search_fn: Callable, train_fn: TrainFn) -> TuningReport:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler()
        pruner = None
        if self.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        report = TuningReport()

        def _optuna_objective(trial: "optuna.trial.Trial") -> float:
            params = search_fn(trial)
            try:
                with self._maybe_mlflow_run(params) as ctx:
                    metrics = train_fn(trial, params)
                    score = metrics.get(self.monitor_metric)
                    if score is None:
                        raise ValueError(
                            f"trial result missing {self.monitor_metric!r}: {metrics}"
                        )
                    state = "COMPLETE"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Trial failed: %s", exc)
                metrics = {}
                score = float("nan")
                state = "FAILED"
            result = TrialResult(
                number=trial.number,
                params=dict(params),
                metrics=dict(metrics),
                state=state,
            )
            report.trials.append(result)
            if state == "COMPLETE":
                report.n_complete += 1
            else:
                report.n_failed += 1
            return score if score == score else float("-inf" if self.direction == "maximize" else "inf")

        study.optimize(_optuna_objective, n_trials=self.n_trials)
        best = study.best_trial
        report.best_params = dict(best.params)
        report.best_trial_number = best.number
        report.best_metrics = {self.monitor_metric: float(best.value)}
        return report

    # ------------------------------------------------------------------
    # Random-search fallback
    # ------------------------------------------------------------------

    def _run_random(self, search_fn: Callable, train_fn: TrainFn) -> TuningReport:
        report = TuningReport()
        # ``trial`` here is a tiny shim that proxies suggest_float /
        # suggest_categorical to random.uniform / random.choice.
        for i in range(self.n_trials):
            trial = _RandomTrialStub(i)
            params = search_fn(trial)
            try:
                with self._maybe_mlflow_run(params):
                    metrics = train_fn(trial, params)
                    score = metrics.get(self.monitor_metric)
                    if score is None:
                        raise ValueError(
                            f"trial result missing {self.monitor_metric!r}: {metrics}"
                        )
                    state = "COMPLETE"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Trial %d failed: %s", i, exc)
                metrics = {}
                score = float("nan")
                state = "FAILED"
            report.trials.append(
                TrialResult(number=i, params=dict(params), metrics=dict(metrics), state=state)
            )
            if state == "COMPLETE":
                report.n_complete += 1
            else:
                report.n_failed += 1

        # Determine best
        completed = [t for t in report.trials if t.state == "COMPLETE"]
        if completed:
            reverse = self.direction == "maximize"
            best = max(
                completed,
                key=lambda t: t.metrics.get(self.monitor_metric, float("-inf" if reverse else "inf")),
            )
            report.best_params = dict(best.params)
            report.best_metrics = dict(best.metrics)
            report.best_trial_number = best.number
        return report

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def _optuna_available(self) -> bool:
        try:
            import optuna  # noqa: F401

            return True
        except ImportError:
            return False

    @contextmanager
    def _maybe_mlflow_run(self, params: Mapping[str, Any]):
        """If log_to_mlflow is True, run train_fn inside a nested MLflow run."""
        if not self.log_to_mlflow:
            yield None
            return
        from mlops.tracking.experiment_logger import ExperimentLogger

        with ExperimentLogger(
            experiment="medvision_hpo",
            run_name=f"trial_{params!r}"[:80],
            tags={"sampler": self.sampler, "monitor": self.monitor_metric},
            nested=True,
        ) as run:
            run.log_params({k: str(v) for k, v in params.items()})
            yield run


# --------------------------------------------------------------------
# Random-search trial stub
# --------------------------------------------------------------------


class _RandomTrialStub:
    """Minimal Optuna-trial-like object backed by ``random``.

    Only implements the methods used by the suggestion templates above.
    """

    def __init__(self, number: int) -> None:
        self.number = number

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if log:
            return 10 ** random.uniform(_log10(low), _log10(high))
        return random.uniform(low, high)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return random.randint(low, high)

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        return random.choice(choices)

    def report(self, *args, **kwargs) -> None:  # pragma: no cover
        pass

    def should_prune(self) -> bool:  # pragma: no cover
        return False


def _suggest_categorical(trial: Any, name: str, options: List[Any]) -> Any:
    if hasattr(trial, "suggest_categorical"):
        return trial.suggest_categorical(name, options)
    return random.choice(options)


def _log10(x: float) -> float:
    import math

    return math.log10(x)


__all__ = [
    "HyperparameterTuner",
    "TrialResult",
    "TuningReport",
    "TrainFn",
    "suggest_learning_rate",
    "suggest_batch_size",
    "suggest_optimizer",
    "suggest_dropout",
    "suggest_weight_decay",
]
