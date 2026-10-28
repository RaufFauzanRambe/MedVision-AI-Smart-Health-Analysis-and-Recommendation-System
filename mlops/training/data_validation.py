"""
data_validation.py
===================
Data quality validation for the MedVision-AI training module.

A common cause of training failures (and silent performance regressions)
is upstream data drift: a missing label, an unexpected image shape, a
class-imbalance explosion. This module exposes a small, framework-
agnostic validation layer that runs *before* training starts and can
also run periodically during training via a hook.

The :class:`DataValidator` accumulates a list of :class:`ValidationRule`
objects. Each rule is a callable ``(sample) -> Optional[str]`` that
returns an error message (or ``None`` if the sample is OK). The validator
collects all failures and produces a structured report.

Example
-------
    from mlops.training.data_validation import DataValidator, image_shape_rule

    validator = DataValidator(
        rules=[
            image_shape_rule(expected=(3, 224, 224)),
            label_in_range_rule(min_label=0, max_label=1),
        ],
        fail_fast=False,
    )
    report = validator.validate(samples)
    if not report.ok:
        report.raise_for_status()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ValidationRule = Callable[[Mapping[str, Any]], Optional[str]]


# --------------------------------------------------------------------
# Built-in rule factories
# --------------------------------------------------------------------


def image_shape_rule(expected: Tuple[int, ...], key: str = "image") -> ValidationRule:
    """Validate that ``sample[key]`` has shape ``expected``.

    Works with numpy arrays and torch tensors (anything exposing ``.shape``).
    """
    expected = tuple(expected)

    def _rule(sample: Mapping[str, Any]) -> Optional[str]:
        if key not in sample:
            return f"missing key '{key}'"
        value = sample[key]
        actual = tuple(getattr(value, "shape", None) or ())
        if actual != expected:
            return f"{key} shape mismatch: expected {expected}, got {actual}"
        return None

    return _rule


def label_in_range_rule(
    min_label: int = 0,
    max_label: int = 1,
    key: str = "label",
) -> ValidationRule:
    """Validate that integer labels are within ``[min_label, max_label]``."""

    def _rule(sample: Mapping[str, Any]) -> Optional[str]:
        if key not in sample:
            return f"missing key '{key}'"
        label = sample[key]
        try:
            value = int(label)
        except (TypeError, ValueError):
            return f"{key} is not an integer: {label!r}"
        if not (min_label <= value <= max_label):
            return f"{key}={value} out of range [{min_label}, {max_label}]"
        return None

    return _rule


def non_null_rule(keys: Sequence[str]) -> ValidationRule:
    """Validate that all ``keys`` are present and not None."""

    def _rule(sample: Mapping[str, Any]) -> Optional[str]:
        for k in keys:
            if k not in sample:
                return f"missing key '{k}'"
            if sample[k] is None:
                return f"'{k}' is None"
        return None

    return _rule


def dtype_rule(expected_dtype: str, key: str = "image") -> ValidationRule:
    """Validate that ``sample[key]`` has the expected dtype name."""

    def _rule(sample: Mapping[str, Any]) -> Optional[str]:
        if key not in sample:
            return f"missing key '{key}'"
        value = sample[key]
        dtype = getattr(getattr(value, "dtype", None), "name", None) or str(
            getattr(value, "dtype", "no dtype")
        )
        if expected_dtype not in dtype:
            return f"{key} dtype mismatch: expected ~{expected_dtype!r}, got {dtype!r}"
        return None

    return _rule


# --------------------------------------------------------------------
# Report
# --------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated result of running rules over a dataset."""

    total_samples: int = 0
    total_passed: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.total_samples == 0 else len(self.failures) / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total_samples": self.total_samples,
            "total_passed": self.total_passed,
            "total_failed": len(self.failures),
            "failure_rate": self.failure_rate,
            "failures": list(self.failures),
        }

    def raise_for_status(self) -> None:
        """Raise :class:`DataValidationError` if any sample failed."""
        if not self.ok:
            head = "; ".join(
                f"#{f['sample_idx']}: {f['message']}" for f in self.failures[:5]
            )
            tail = "" if len(self.failures) <= 5 else f" (+{len(self.failures) - 5} more)"
            raise DataValidationError(
                f"{len(self.failures)}/{self.total_samples} samples failed validation: {head}{tail}"
            )


class DataValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_for_status`."""


# --------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------


@dataclass
class DataValidator:
    """Run a list of :class:`ValidationRule` over a batch of samples.

    Attributes
    ----------
    rules:
        Ordered list of validation rules.
    fail_fast:
        When True, stops on the first failing sample. When False
        (the default), runs every rule on every sample and aggregates
        all failures.
    max_failures:
        Hard cap on the number of recorded failures (to avoid
        unbounded memory usage on broken datasets).
    """

    rules: List[ValidationRule] = field(default_factory=list)
    fail_fast: bool = False
    max_failures: int = 1000

    def validate(self, samples: Sequence[Mapping[str, Any]]) -> ValidationReport:
        """Apply every rule to every sample; return a :class:`ValidationReport`."""
        report = ValidationReport(total_samples=len(samples))
        for idx, sample in enumerate(samples):
            passed = True
            for rule in self.rules:
                msg = rule(sample)
                if msg is not None:
                    report.failures.append({"sample_idx": idx, "message": msg})
                    passed = False
                    if self.fail_fast:
                        return report
                    if len(report.failures) >= self.max_failures:
                        return report
                    break  # don't run remaining rules on this sample
            if passed:
                report.total_passed += 1
        return report

    def add_rule(self, rule: ValidationRule) -> "DataValidator":
        """Append a rule (returns self for chaining)."""
        self.rules.append(rule)
        return self


__all__ = [
    "ValidationRule",
    "ValidationReport",
    "DataValidationError",
    "DataValidator",
    "image_shape_rule",
    "label_in_range_rule",
    "non_null_rule",
    "dtype_rule",
]
