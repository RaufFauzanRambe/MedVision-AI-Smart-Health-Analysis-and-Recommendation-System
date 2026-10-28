"""
training_utils.py
=================
Shared helpers used across the MedVision-AI training module.

These helpers keep the rest of the modules small and focused:

- ``format_eta``, ``humanize_bytes`` — display helpers for progress bars.
- ``count_parameters`` — count trainable / total params of a model.
- ``move_to_device`` — recursively move tensors / lists / dicts to a device.
- ``AverageMeter`` — running mean / sum accumulator for losses & metrics.
- ``evaluate_metric`` — dispatch to sklearn / torchmetrics by name.

The module deliberately avoids importing torch at the top level so it
can be used in config-only contexts (e.g. CI gates that lint configs).
torch is imported lazily inside the functions that need it.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

NumberLike = Union[int, float]


# --------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------


def format_eta(seconds: NumberLike) -> str:
    """Format a duration in seconds as ``HH:MM:SS``."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def humanize_bytes(num: int, suffix: str = "B") -> str:
    """Render an integer byte count as ``1.4 GiB`` style string."""
    if num < 0:
        raise ValueError(f"Byte count must be non-negative, got {num}")
    units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]
    value = float(num)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:3.1f} {unit}{suffix}" if unit else f"{int(value)} {suffix}"
        value /= 1024.0
    return f"{value:3.1f} Yi{suffix}"  # pragma: no cover


# --------------------------------------------------------------------
# Model inspection
# --------------------------------------------------------------------


def count_parameters(model: Any, trainable_only: bool = True) -> Dict[str, int]:
    """Return ``{"total": ..., "trainable": ...}`` for a torch model."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("count_parameters requires torch") from exc
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_only:
        return {"total": int(total), "trainable": int(trainable)}
    return {"total": int(total), "trainable": int(trainable)}


def move_to_device(obj: Any, device: Any) -> Any:
    """Recursively move tensors / lists / dicts to ``device``."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


# --------------------------------------------------------------------
# Metric / loss accumulators
# --------------------------------------------------------------------


class AverageMeter:
    """Running average / sum accumulator with optional batch weighting.

    >>> am = AverageMeter()
    >>> am.update(0.5, n=10)
    >>> am.update(1.0, n=5)
    >>> am.avg
    0.6666666666666666
    """

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def update(self, value: NumberLike, n: int = 1) -> None:
        if n <= 0:
            return
        self.sum += float(value) * n
        self.count += n

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"AverageMeter(avg={self.avg:.6f}, n={self.count})"


# --------------------------------------------------------------------
# Metric dispatch
# --------------------------------------------------------------------


_SKLEARN_REGISTRY: Dict[str, str] = {
    "accuracy": "accuracy_score",
    "f1": "f1_score",
    "precision": "precision_score",
    "recall": "recall_score",
    "roc_auc": "roc_auc_score",
    "average_precision": "average_precision_score",
}


def evaluate_metric(
    name: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    **kwargs: Any,
) -> float:
    """Dispatch to sklearn by metric name. Returns a float.

    For healthcare use-cases we expose the four core classification
    metrics. For richer metrics (e.g. per-subgroup AUC), wrap your
    own evaluator and call it directly.
    """
    sklearn_name = _SKLEARN_REGISTRY.get(name)
    if not sklearn_name:
        raise ValueError(
            f"Unknown metric {name!r}; expected one of {sorted(_SKLEARN_REGISTRY)}"
        )
    from sklearn.metrics import get_scorer  # noqa: F401 (kept for clarity)

    import sklearn.metrics as sm

    fn: Callable[..., float] = getattr(sm, sklearn_name)  # type: ignore[assignment]
    value = fn(y_true, y_pred, **kwargs)
    return float(value)


def safe_min_max(values: Iterable[NumberLike]) -> Tuple[float, float]:
    """Return ``(min, max)`` for an iterable of numbers, tolerating empties."""
    items = [float(v) for v in values]
    if not items:
        return float("nan"), float("nan")
    return min(items), max(items)


__all__ = [
    "format_eta",
    "humanize_bytes",
    "count_parameters",
    "move_to_device",
    "AverageMeter",
    "evaluate_metric",
    "safe_min_max",
]
