"""Tests for mlops.tracking.metrics_logger."""

from __future__ import annotations

import math

import pytest

from mlops.tracking.metrics_logger import (
    MetricValidationError,
    log_metric,
    log_metrics,
    summarize_metrics,
    validate_metric_name,
    validate_metric_value,
)


# ------------------------------------------------------------------
# validate_metric_name
# ------------------------------------------------------------------


def test_validate_metric_name_accepts_legal_names():
    for name in ["accuracy", "val/loss", "train.f1", "step 1", "a-b_c.d/e"]:
        assert validate_metric_name(name) == name


def test_validate_metric_name_rejects_empty():
    with pytest.raises(MetricValidationError):
        validate_metric_name("")


def test_validate_metric_name_rejects_leading_whitespace():
    with pytest.raises(MetricValidationError):
        validate_metric_name(" accuracy")


def test_validate_metric_name_rejects_illegal_chars():
    with pytest.raises(MetricValidationError):
        validate_metric_name("accuracy@val")


# ------------------------------------------------------------------
# validate_metric_value
# ------------------------------------------------------------------


def test_validate_metric_value_accepts_int_and_float():
    assert validate_metric_value(1) == 1.0
    assert validate_metric_value(0.9) == 0.9


def test_validate_metric_value_rejects_nan():
    with pytest.raises(MetricValidationError):
        validate_metric_value(float("nan"))


def test_validate_metric_value_rejects_inf():
    with pytest.raises(MetricValidationError):
        validate_metric_value(float("inf"))


def test_validate_metric_value_rejects_strings():
    with pytest.raises(MetricValidationError):
        validate_metric_value("0.95")


# ------------------------------------------------------------------
# log_metric / log_metrics (require mlflow + active run)
# ------------------------------------------------------------------


def test_log_metric_requires_active_run():
    # Without an active run, mlflow will raise. We just verify our
    # validation runs first (so the error is our error type, not
    # mlflow's).
    with pytest.raises(MetricValidationError):
        log_metric("bad@name", 0.5)


def test_log_metrics_validates_before_logging():
    bad_batch = {"accuracy": 0.9, "bad@name": 0.5}
    # Validation must fail before any mlflow call
    with pytest.raises(MetricValidationError):
        log_metrics(bad_batch)


# ------------------------------------------------------------------
# summarize_metrics
# ------------------------------------------------------------------


def test_summarize_metrics_basic():
    folds = [
        {"accuracy": 0.9, "f1": 0.8},
        {"accuracy": 0.92, "f1": 0.82},
        {"accuracy": 0.88, "f1": 0.78},
    ]
    summary = summarize_metrics(folds)
    assert set(summary.keys()) == {"accuracy", "f1"}
    assert math.isclose(summary["accuracy"]["mean"], 0.9, abs_tol=1e-9)
    assert summary["accuracy"]["n_folds"] == 3.0


def test_summarize_metrics_empty():
    assert summarize_metrics([]) == {}


def test_summarize_metrics_single_fold_has_zero_std():
    summary = summarize_metrics([{"accuracy": 0.9}])
    assert summary["accuracy"]["std"] == 0.0


def test_summarize_metrics_only_keeps_common_keys():
    folds = [
        {"accuracy": 0.9, "f1": 0.8},
        {"accuracy": 0.92},  # missing f1
    ]
    summary = summarize_metrics(folds)
    assert "accuracy" in summary
    assert "f1" not in summary
