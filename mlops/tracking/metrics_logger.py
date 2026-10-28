"""
metrics_logger.py
=================
Typed metric-logging helpers for the MedVision-AI MLOps stack.

The module wraps ``mlflow.log_metric`` with:

- type validation (so that ``"accuracy"`` is never accidentally logged
  as a string),
- batch logging (``log_metrics`` accepts a dictionary),
- step-tagged time-series logging (``log_timeseries``),
- safety against ``inf``/``NaN`` values, which would otherwise
  silently corrupt MLflow dashboards.

The :func:`summarize_metrics` helper reduces a list of per-fold
metrics (common in cross-validation) into mean ± std, ready for
tabular reporting.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class MetricValidationError(ValueError):
    """Raised when a metric value or name fails validation."""


_VALID_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. /")


def validate_metric_name(name: str) -> str:
    """Return ``name`` if it is a legal MLflow metric name.

    MLflow metric names must be non-empty strings composed of
    alphanumeric characters, underscores, hyphens, periods, spaces, or
    slashes. We additionally forbid leading/trailing whitespace.
    """
    if not isinstance(name, str) or not name.strip():
        raise MetricValidationError(f"Metric name must be a non-empty string, got: {name!r}")
    if name != name.strip():
        raise MetricValidationError(f"Metric name must not have leading/trailing whitespace: {name!r}")
    for ch in name:
        if ch not in _VALID_NAME_CHARS:
            raise MetricValidationError(
                f"Metric name {name!r} contains illegal character {ch!r}"
            )
    return name


def validate_metric_value(value: Any) -> float:
    """Coerce ``value`` to a finite float.

    Raises :class:`MetricValidationError` for ``NaN``/``inf`` or
    non-numeric inputs. MLflow accepts ``inf`` technically, but in
    practice infinite metrics break the UI charts and downstream
    parity checks.
    """
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricValidationError(
            f"Metric value must be numeric, got {type(value).__name__}: {value!r}"
        ) from exc
    if math.isnan(v):
        raise MetricValidationError("Metric value is NaN; refusing to log.")
    if math.isinf(v):
        raise MetricValidationError(f"Metric value is infinite: {value!r}")
    return v


def log_metric(name: str, value: Any, step: Optional[int] = None) -> Tuple[str, float, Optional[int]]:
    """Validate and forward a single metric to MLflow.

    Returns the validated ``(name, value, step)`` triple so callers
    can keep an audit trail without re-validating.
    """
    import mlflow

    name = validate_metric_name(name)
    v = validate_metric_value(value)
    if step is not None:
        if not isinstance(step, int) or step < 0:
            raise MetricValidationError(f"step must be a non-negative int, got {step!r}")
    mlflow.log_metric(name, v, step=step)
    return name, v, step


def log_metrics(metrics: Mapping[str, Any], step: Optional[int] = None) -> Dict[str, float]:
    """Log a batch of metrics at the same step.

    Returns a dictionary of the validated values keyed by metric name.
    If any single metric fails validation, the entire batch is aborted
    (no partial state is logged) so callers can fix the offending
    entry and retry cleanly.
    """
    validated: Dict[str, float] = {}
    for name, value in metrics.items():
        n = validate_metric_name(name)
        v = validate_metric_value(value)
        validated[n] = v
    # All validated — now actually log
    import mlflow

    mlflow.log_metrics(validated, step=step)
    return validated


def log_timeseries(
    name: str,
    values: Sequence[float],
    start_step: int = 0,
) -> None:
    """Log a list of values as a step-indexed time series.

    ``values[i]`` is logged at ``step = start_step + i``. Useful for
    per-epoch loss tracking during training.
    """
    name = validate_metric_name(name)
    for i, raw in enumerate(values):
        v = validate_metric_value(raw)
        import mlflow

        mlflow.log_metric(name, v, step=start_step + i)


def summarize_metrics(
    per_fold: Iterable[Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Reduce per-fold metrics into ``{metric: {"mean": ..., "std": ...}}``.

    The function is robust to folds that report different metric sets:
    only metrics present in *every* fold are summarized. If only one
    fold is provided, ``std`` is set to ``0.0``.
    """
    folds = list(per_fold)
    if not folds:
        return {}
    common_keys = set(folds[0].keys())
    for fold in folds[1:]:
        common_keys &= set(fold.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(common_keys):
        values = [float(fold[key]) for fold in folds]
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std, "n_folds": float(len(values))}
    return summary


__all__ = [
    "MetricValidationError",
    "validate_metric_name",
    "validate_metric_value",
    "log_metric",
    "log_metrics",
    "log_timeseries",
    "summarize_metrics",
]
