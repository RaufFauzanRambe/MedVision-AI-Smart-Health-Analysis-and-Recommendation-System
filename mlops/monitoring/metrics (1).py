"""
MedVision-AI — Monitoring metrics.

Two layers:

1. *Operational metrics* — about the service itself (request count,
   inference latency, error rate, in-flight requests). These are
   exposed in Prometheus exposition format via :func:`render_prometheus`.

2. *Model-evaluation metrics* — computed OFFLINE against a labeled
   reference set (precision, recall, F1, ROC-AUC, calibration error).
   These are NOT computed from live request traffic, because live
   traffic rarely has reliable reference labels. See :func:`evaluate_model`.

The operational metrics are intentionally implementation-light so they
work with or without the `prometheus_client` library: if it is
installed, metrics are also registered into the default registry and
can be scraped at ``/metrics``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Optional Prometheus integration
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter as _PromCounter,
        Gauge as _PromGauge,
        Histogram as _PromHistogram,
    )
    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover — optional dependency
    _HAS_PROMETHEUS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NAMESPACE = "medvision"

# Tracks whether the Prometheus bridge has already been wired for this
# process. The prometheus_client default registry forbids duplicate
# timeseries, so we register exactly once.
_PROM_BRIDGE_INITIALIZED = False

DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

DEFAULT_CONFIDENCE_BUCKETS: tuple[float, ...] = (
    0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0,
)


# ---------------------------------------------------------------------------
# In-memory metric primitives (always available)
# ---------------------------------------------------------------------------

class _Counter:
    """A simple monotonically-increasing counter keyed by label dict."""
    __slots__ = ("name", "help", "_values", "_lock")

    def __init__(self, name: str, help: str) -> None:
        self.name = name
        self.help = help
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += amount

    def collect(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


class _Histogram:
    """A simple bucketed histogram keyed by label dict."""
    __slots__ = ("name", "help", "buckets", "_state", "_lock")

    def __init__(self, name: str, help: str,
                 buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS) -> None:
        self.name = name
        self.help = help
        self.buckets = tuple(sorted(set(buckets) | {float("inf")}))
        self._state: dict[tuple[tuple[str, str], ...],
                          dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sum": 0.0,
                     "buckets": [0] * len(self.buckets)})
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            st = self._state[key]
            st["count"] += 1
            st["sum"] += float(value)
            for i, b in enumerate(self.buckets):
                if value <= b:
                    st["buckets"][i] += 1

    def collect(self) -> list[tuple[dict[str, str], dict[str, Any]]]:
        with self._lock:
            return [(dict(k), dict(v)) for k, v in self._state.items()]


class _Gauge:
    """A simple gauge keyed by label dict."""
    __slots__ = ("name", "help", "_values", "_lock")

    def __init__(self, name: str, help: str) -> None:
        self.name = name
        self.help = help
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

@dataclass
class MetricRegistry:
    """Holds all operational metrics for the MedVision service."""
    requests_total: _Counter = field(init=False)
    request_duration_seconds: _Histogram = field(init=False)
    request_errors_total: _Counter = field(init=False)
    requests_in_flight: _Gauge = field(init=False)
    model_inference_duration_seconds: _Histogram = field(init=False)
    model_predictions_total: _Counter = field(init=False)
    model_confidence: _Histogram = field(init=False)
    data_quality_completeness_pct: _Gauge = field(init=False)
    data_quality_missing_values_total: _Counter = field(init=False)

    def __post_init__(self) -> None:
        self.requests_total = _Counter(
            "requests_total", "Total HTTP requests handled by the service.")
        self.request_duration_seconds = _Histogram(
            "request_duration_seconds",
            "End-to-end HTTP request latency in seconds.",
            buckets=DEFAULT_LATENCY_BUCKETS)
        self.request_errors_total = _Counter(
            "request_errors_total", "Total HTTP requests that ended in an error.")
        self.requests_in_flight = _Gauge(
            "requests_in_flight", "Number of requests currently being served.")
        self.model_inference_duration_seconds = _Histogram(
            "model_inference_duration_seconds",
            "Time spent inside the model inference call, in seconds.",
            buckets=DEFAULT_LATENCY_BUCKETS)
        self.model_predictions_total = _Counter(
            "model_predictions_total", "Total model predictions produced.")
        self.model_confidence = _Histogram(
            "model_confidence",
            "Reported model confidence per prediction (0..1).",
            buckets=DEFAULT_CONFIDENCE_BUCKETS)
        self.data_quality_completeness_pct = _Gauge(
            "data_quality_completeness_pct",
            "Percentage of required fields present in incoming requests.")
        self.data_quality_missing_values_total = _Counter(
            "data_quality_missing_values_total",
            "Total count of missing required fields observed in requests.")

        # Mirror into Prometheus registry if available, so /metrics works.
        # The bridge is created at most once per process — the prometheus
        # default registry does not allow duplicate timeseries, and our
        # MetricRegistry may be instantiated multiple times (tests, hot
        # reloads). After the first wiring, subsequent MetricRegistry
        # instances stay self-contained (in-memory only) but still expose
        # `render_prometheus()`.
        if _HAS_PROMETHEUS and not _PROM_BRIDGE_INITIALIZED:
            self._mirror_prometheus()

    # ---- Prometheus bridge -------------------------------------------------
    def _mirror_prometheus(self) -> None:  # pragma: no cover — I/O bridge
        global _PROM_BRIDGE_INITIALIZED
        if _PROM_BRIDGE_INITIALIZED:
            return
        _PROM_BRIDGE_INITIALIZED = True
        self._prom_counter = {
            "requests": _PromCounter(
                f"{NAMESPACE}_requests_total",
                "Total HTTP requests handled by the service.",
                ["endpoint", "method", "status"]),
            "errors": _PromCounter(
                f"{NAMESPACE}_request_errors_total",
                "Total HTTP requests that ended in an error.",
                ["endpoint", "method", "status", "error_type"]),
            "predictions": _PromCounter(
                f"{NAMESPACE}_model_predictions_total",
                "Total model predictions produced.",
                ["model_name", "model_version", "endpoint"]),
            "missing_values": _PromCounter(
                f"{NAMESPACE}_data_quality_missing_values_total",
                "Total count of missing required fields observed in requests.",
                ["endpoint", "field"]),
        }
        self._prom_hist = {
            "request_duration": _PromHistogram(
                f"{NAMESPACE}_request_duration_seconds",
                "End-to-end HTTP request latency in seconds.",
                ["endpoint", "method"],
                buckets=DEFAULT_LATENCY_BUCKETS),
            "inference_duration": _PromHistogram(
                f"{NAMESPACE}_model_inference_duration_seconds",
                "Time spent inside the model inference call, in seconds.",
                ["model_name", "model_version"],
                buckets=DEFAULT_LATENCY_BUCKETS),
            "confidence": _PromHistogram(
                f"{NAMESPACE}_model_confidence",
                "Reported model confidence per prediction (0..1).",
                ["model_name", "model_version"],
                buckets=DEFAULT_CONFIDENCE_BUCKETS),
        }
        self._prom_gauge = {
            "in_flight": _PromGauge(
                f"{NAMESPACE}_requests_in_flight",
                "Number of requests currently being served.",
                ["endpoint"]),
            "completeness": _PromGauge(
                f"{NAMESPACE}_data_quality_completeness_pct",
                "Percentage of required fields present in incoming requests.",
                ["endpoint"]),
        }

    # ---- Operational recording API ---------------------------------------
    def observe_request(
        self,
        *,
        endpoint: str,
        method: str,
        status: int,
        duration_seconds: float,
        error_type: str | None = None,
    ) -> None:
        labels = {"endpoint": endpoint, "method": method.upper(),
                  "status": str(status)}
        self.requests_total.inc(**labels)
        self.request_duration_seconds.observe(
            duration_seconds, endpoint=endpoint, method=method.upper())
        if status >= 400 or error_type is not None:
            err_labels = {**labels, "error_type": error_type or "unknown"}
            self.request_errors_total.inc(**err_labels)

    def track_inference(
        self,
        *,
        model_name: str,
        model_version: str,
        duration_seconds: float,
        confidence: float | None = None,
        endpoint: str = "/predict",
    ) -> None:
        self.model_predictions_total.inc(
            model_name=model_name, model_version=model_version, endpoint=endpoint)
        self.model_inference_duration_seconds.observe(
            duration_seconds, model_name=model_name, model_version=model_version)
        if confidence is not None:
            clamped = max(0.0, min(1.0, float(confidence)))
            self.model_confidence.observe(
                clamped, model_name=model_name, model_version=model_version)

    def observe_data_quality(
        self,
        *,
        endpoint: str,
        required_fields: Sequence[str],
        present_fields: Sequence[str],
    ) -> None:
        total = len(required_fields)
        present = sum(1 for f in required_fields if f in set(present_fields))
        if total == 0:
            return
        completeness = 100.0 * present / total
        self.data_quality_completeness_pct.set(
            completeness, endpoint=endpoint)
        missing = total - present
        if missing > 0:
            self.data_quality_missing_values_total.inc(
                amount=float(missing), endpoint=endpoint, field="_total")


# Module-level singleton, plus a context-manager for in-flight tracking.
REGISTRY = MetricRegistry()


class InFlightTracker:
    """Context manager that bumps the in-flight gauge for a request."""
    __slots__ = ("endpoint",)

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def __enter__(self) -> "InFlightTracker":
        REGISTRY.requests_in_flight.inc(1.0, endpoint=self.endpoint)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        REGISTRY.requests_in_flight.dec(1.0, endpoint=self.endpoint)


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------

def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


def render_prometheus() -> str:
    """Render all metrics in Prometheus text exposition format v0.0.4."""
    lines: list[str] = []
    reg = REGISTRY

    # Counters
    for metric in (reg.requests_total, reg.request_errors_total,
                   reg.model_predictions_total, reg.data_quality_missing_values_total):
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} counter")
        for labels, value in metric.collect():
            lines.append(f"{metric.name}{_fmt_labels(labels)} {value}")

    # Gauges
    for metric in (reg.requests_in_flight, reg.data_quality_completeness_pct):
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} gauge")
        for labels, value in metric.collect():
            lines.append(f"{metric.name}{_fmt_labels(labels)} {value}")

    # Histograms
    for metric in (reg.request_duration_seconds,
                   reg.model_inference_duration_seconds,
                   reg.model_confidence):
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} histogram")
        for labels, st in metric.collect():
            for b, count in zip(metric.buckets, st["buckets"]):
                b_label = "+Inf" if math.isinf(b) else str(b)
                bucket_labels = {**labels, "le": b_label}
                lines.append(
                    f"{metric.name}_bucket{_fmt_labels(bucket_labels)} {count}")
            lines.append(
                f"{metric.name}_count{_fmt_labels(labels)} {st['count']}")
            lines.append(
                f"{metric.name}_sum{_fmt_labels(labels)} {st['sum']:.6f}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Model-evaluation helpers (OFFLINE only)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Result of an offline evaluation against a labeled reference set."""
    model_name: str
    model_version: str
    n_samples: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    calibration_error: float | None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "n_samples": self.n_samples,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 6),
            "calibration_error": None if self.calibration_error is None
                                  else round(self.calibration_error, 6),
            "timestamp": self.timestamp,
        }


def _confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for binary 0/1 sequences."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        else: tn += 1
    return tp, fp, fn, tn


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def evaluate_model(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
    model_name: str = "medvision-v1",
    model_version: str = "unknown",
    min_samples: int = 50,
) -> EvaluationReport:
    """Compute offline model-quality metrics from a labeled reference set.

    IMPORTANT:
        This function must be called only with a labeled evaluation set
        whose ground truth is reliable. It MUST NOT be applied to live
        request traffic, which generally has no reliable labels.

    Raises:
        ValueError: if inputs are empty or length-mismatched.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if y_score is not None and len(y_score) != len(y_true):
        raise ValueError("y_score must have the same length as y_true")
    n = len(y_true)
    if n == 0:
        raise ValueError("Evaluation set is empty")

    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    roc_auc: float | None = None
    if y_score is not None and n >= 2 and len(set(y_true)) == 2:
        roc_auc = _roc_auc_score(y_true, y_score)

    cal_error: float | None = None
    if y_score is not None:
        cal_error = _expected_calibration_error(y_true, y_score, n_bins=10)

    return EvaluationReport(
        model_name=model_name, model_version=model_version,
        n_samples=n,
        precision=precision, recall=recall, f1=f1,
        roc_auc=roc_auc, calibration_error=cal_error,
    )


def _roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic. Returns 0.5 if undefined."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = 0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                wins += 0.5
    return _safe_div(wins, len(pos) * len(neg))


def _expected_calibration_error(
    y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10
) -> float:
    """ECE: sum |bin_acc - bin_conf| * (bin_size / total), over n_bins."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, s in enumerate(y_score)
               if (s > lo or i == 0) and s <= hi]
        if not idx:
            continue
        bin_acc = sum(1 for j in idx if y_true[j] == 1) / len(idx)
        bin_conf = sum(y_score[j] for j in idx) / len(idx)
        ece += abs(bin_acc - bin_conf) * (len(idx) / total)
    return ece


__all__ = [
    "DEFAULT_LATENCY_BUCKETS", "DEFAULT_CONFIDENCE_BUCKETS",
    "MetricRegistry", "REGISTRY", "InFlightTracker",
    "EvaluationReport", "evaluate_model", "render_prometheus",
]
