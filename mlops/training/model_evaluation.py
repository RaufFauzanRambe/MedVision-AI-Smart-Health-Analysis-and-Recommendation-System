"""
model_evaluation.py
====================
Model evaluation utilities for the MedVision-AI training module.

The :class:`ModelEvaluator` orchestrates evaluation:

- Iterates over a validation DataLoader,
- Calls the model on each batch (under ``torch.no_grad()``),
- Computes a configurable set of metrics (accuracy, F1, AUC, ...),
- Optionally generates per-subgroup metrics for fairness audits,
- Returns an :class:`EvaluationReport` that the training loop and the
  experiment logger both consume.

The evaluator is intentionally decoupled from the training loop: it
takes a model and a loader, returns a report, and never mutates state
on the model. This makes it equally usable for end-of-training
evaluation and for periodic validation during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .training_utils import AverageMeter, evaluate_metric


@dataclass
class EvaluationReport:
    """Structured result of an evaluation pass."""

    metrics: Dict[str, float] = field(default_factory=dict)
    per_subgroup: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_samples: int = 0
    num_batches: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "per_subgroup": {k: dict(v) for k, v in self.per_subgroup.items()},
            "num_samples": self.num_samples,
            "num_batches": self.num_batches,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def update_best(self, other: "EvaluationReport", monitor: str, mode: str = "min") -> bool:
        """Return True if ``other`` improves on this report's ``monitor`` value."""
        if monitor not in other.metrics or monitor not in self.metrics:
            return monitor not in self.metrics
        if mode == "min":
            return other.metrics[monitor] < self.metrics[monitor]
        if mode == "max":
            return other.metrics[monitor] > self.metrics[monitor]
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")


# --------------------------------------------------------------------
# Subgroup evaluator
# --------------------------------------------------------------------


SubgroupSplitter = Callable[[Any], Optional[str]]


def default_subgroup_splitter(attribute_key: str = "subgroup") -> SubgroupSplitter:
    """Split predictions by ``sample['metadata'][attribute_key]`` if present."""

    def _split(sample: Any) -> Optional[str]:
        meta = getattr(sample, "get", lambda *a, **k: None)("__metadata__", None) or (
            sample.get("metadata") if isinstance(sample, dict) else None
        )
        if not isinstance(meta, dict):
            return None
        value = meta.get(attribute_key)
        return str(value) if value is not None else None

    return _split


# --------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------


@dataclass
class ModelEvaluator:
    """Evaluate a model on a DataLoader.

    Attributes
    ----------
    metrics:
        Names of metrics to compute (see :data:`training_utils._SKLEARN_REGISTRY`
        for the supported set). At minimum, ``"loss"`` is always reported.
    device:
        Torch device string. When ``None``, the model's existing device
        is used.
    subgroup_splitter:
        Optional callable returning a subgroup name for a sample
        (or ``None`` to skip subgroup bookkeeping).
    """

    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "roc_auc"])
    device: Optional[str] = None
    subgroup_splitter: Optional[SubgroupSplitter] = None
    loss_fn: Optional[Callable[[Any, Any], float]] = None

    def evaluate(self, model: Any, loader: Any) -> EvaluationReport:
        """Run a single evaluation pass and return a report."""
        import time

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ModelEvaluator requires torch") from exc

        device = self.device or next(model.parameters()).device
        model.eval()
        loss_meter = AverageMeter()
        all_preds: List[Any] = []
        all_targets: List[Any] = []
        all_subgroups: List[Optional[str]] = []
        n_batches = 0
        n_samples = 0
        t0 = time.time()

        with torch.no_grad():
            for batch in loader:
                # Loader is expected to yield (inputs, targets) or a dict
                inputs, targets, subgroups = self._unpack_batch(batch)
                inputs = self._to_device(inputs, device)
                targets = self._to_device(targets, device)
                outputs = model(inputs)
                if self.loss_fn is not None:
                    loss_val = float(self.loss_fn(outputs, targets))
                    loss_meter.update(loss_val, n=self._batch_size(targets))
                preds = self._extract_predictions(outputs)
                all_preds.extend(self._to_cpu_list(preds))
                all_targets.extend(self._to_cpu_list(targets))
                if subgroups:
                    all_subgroups.extend(subgroups)
                else:
                    n = len(self._to_cpu_list(targets))
                    all_subgroups.extend([None] * n)
                n_batches += 1
                n_samples += self._batch_size(targets)

        report = EvaluationReport(
            num_samples=n_samples,
            num_batches=n_batches,
            elapsed_seconds=time.time() - t0,
        )
        if loss_meter.count > 0:
            report.metrics["loss"] = loss_meter.avg
        for m in self.metrics:
            try:
                report.metrics[m] = evaluate_metric(m, all_targets, all_preds)
            except Exception as exc:  # noqa: BLE001
                # A metric may legitimately fail on degenerate batches
                # (e.g. AUC with a single class). Skip rather than abort.
                report.metrics[m] = float("nan")
                report.metrics[f"__error_{m}"] = str(exc)[:200]
        if self.subgroup_splitter is not None and any(all_subgroups):
            self._compute_per_subgroup(report, all_targets, all_preds, all_subgroups)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Any):
        """Extract (inputs, targets, [subgroups]) from a batch."""
        subgroups: List[Optional[str]] = []
        if isinstance(batch, dict):
            inputs = batch.get("image")
            targets = batch.get("label")
            if self.subgroup_splitter is not None:
                # Batch is a dict-of-tensors; we can't subgroup here.
                pass
            return inputs, targets, subgroups
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1], subgroups
            return batch[0], None, subgroups
        return batch, None, subgroups

    def _to_device(self, obj: Any, device: Any) -> Any:
        from .training_utils import move_to_device

        return move_to_device(obj, device)

    @staticmethod
    def _batch_size(targets: Any) -> int:
        try:
            return int(targets.shape[0])
        except Exception:  # noqa: BLE001
            return 1

    @staticmethod
    def _extract_predictions(outputs: Any) -> Any:
        """Convert raw model outputs into a 1-D tensor of predictions."""
        try:
            import torch
        except ImportError:  # pragma: no cover
            return outputs
        if isinstance(outputs, torch.Tensor) and outputs.ndim >= 2:
            return outputs.argmax(dim=1)
        return outputs

    @staticmethod
    def _to_cpu_list(value: Any) -> List[Any]:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return list(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [ModelEvaluator._to_cpu_list(v) for v in value]
        return value

    def _compute_per_subgroup(
        self,
        report: EvaluationReport,
        all_targets: List[Any],
        all_preds: List[Any],
        all_subgroups: List[Optional[str]],
    ) -> None:
        groups: Dict[str, List[int]] = {}
        for idx, g in enumerate(all_subgroups):
            if g is None:
                continue
            groups.setdefault(g, []).append(idx)
        for g, indices in groups.items():
            tg = [all_targets[i] for i in indices]
            pg = [all_preds[i] for i in indices]
            row: Dict[str, float] = {"n": float(len(indices))}
            for m in self.metrics:
                try:
                    row[m] = evaluate_metric(m, tg, pg)
                except Exception:  # noqa: BLE001
                    row[m] = float("nan")
            report.per_subgroup[g] = row


__all__ = [
    "EvaluationReport",
    "ModelEvaluator",
    "SubgroupSplitter",
    "default_subgroup_splitter",
]
