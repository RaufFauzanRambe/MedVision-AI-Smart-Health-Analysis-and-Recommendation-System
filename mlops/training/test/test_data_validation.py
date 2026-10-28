"""Tests for mlops.training.data_validation."""

from __future__ import annotations

import numpy as np
import pytest

from mlops.training.data_validation import (
    DataValidationError,
    DataValidator,
    dtype_rule,
    image_shape_rule,
    label_in_range_rule,
    non_null_rule,
)


# --------------------------------------------------------------------
# Rule factories
# --------------------------------------------------------------------


def test_image_shape_rule_passes_on_match():
    sample = {"image": np.zeros((3, 224, 224), dtype=np.float32)}
    rule = image_shape_rule(expected=(3, 224, 224))
    assert rule(sample) is None


def test_image_shape_rule_flags_mismatch():
    sample = {"image": np.zeros((3, 256, 256), dtype=np.float32)}
    rule = image_shape_rule(expected=(3, 224, 224))
    msg = rule(sample)
    assert msg is not None
    assert "shape mismatch" in msg


def test_image_shape_rule_flags_missing_key():
    rule = image_shape_rule(expected=(3, 224, 224))
    msg = rule({})
    assert "missing key" in (msg or "")


def test_label_in_range_rule_accepts_legal_labels():
    rule = label_in_range_rule(min_label=0, max_label=1)
    assert rule({"label": 0}) is None
    assert rule({"label": 1}) is None


def test_label_in_range_rule_flags_out_of_range():
    rule = label_in_range_rule(min_label=0, max_label=1)
    msg = rule({"label": 5})
    assert "out of range" in (msg or "")


def test_label_in_range_rule_flags_non_integer():
    rule = label_in_range_rule(min_label=0, max_label=1)
    msg = rule({"label": "abc"})
    assert "not an integer" in (msg or "")


def test_non_null_rule_passes_when_all_present():
    rule = non_null_rule(keys=["image", "label"])
    assert rule({"image": np.zeros(1), "label": 0}) is None


def test_non_null_rule_flags_missing_key():
    rule = non_null_rule(keys=["image", "label"])
    assert "missing key" in (rule({"image": np.zeros(1)}) or "")


def test_non_null_rule_flags_none_value():
    rule = non_null_rule(keys=["image", "label"])
    assert "None" in (rule({"image": np.zeros(1), "label": None}) or "")


def test_dtype_rule_passes_on_match():
    sample = {"image": np.zeros(1, dtype=np.float32)}
    rule = dtype_rule(expected_dtype="float")
    assert rule(sample) is None


def test_dtype_rule_flags_mismatch():
    sample = {"image": np.zeros(1, dtype=np.uint8)}
    rule = dtype_rule(expected_dtype="float")
    assert "dtype mismatch" in (rule(sample) or "")


# --------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------


def test_validator_passes_clean_dataset():
    samples = [
        {"image": np.zeros((3, 224, 224), dtype=np.float32), "label": 0},
        {"image": np.zeros((3, 224, 224), dtype=np.float32), "label": 1},
    ]
    validator = DataValidator(
        rules=[image_shape_rule(expected=(3, 224, 224)), label_in_range_rule(0, 1)],
    )
    report = validator.validate(samples)
    assert report.ok
    assert report.total_samples == 2
    assert report.total_passed == 2
    assert report.failure_rate == 0.0


def test_validator_collects_all_failures_when_not_fail_fast():
    samples = [
        {"image": np.zeros((3, 224, 224)), "label": 0},  # OK
        {"image": np.zeros((3, 256, 256)), "label": 0},  # shape mismatch
        {"image": np.zeros((3, 224, 224)), "label": 9},  # label out of range
    ]
    validator = DataValidator(
        rules=[image_shape_rule(expected=(3, 224, 224)), label_in_range_rule(0, 1)],
        fail_fast=False,
    )
    report = validator.validate(samples)
    assert not report.ok
    assert len(report.failures) == 2
    assert report.total_passed == 1
    assert report.failure_rate == pytest.approx(2 / 3)


def test_validator_fail_fast_returns_first_failure():
    samples = [
        {"image": np.zeros((3, 256, 256)), "label": 0},
        {"image": np.zeros((3, 224, 224)), "label": 9},
    ]
    validator = DataValidator(
        rules=[image_shape_rule(expected=(3, 224, 224)), label_in_range_rule(0, 1)],
        fail_fast=True,
    )
    report = validator.validate(samples)
    assert len(report.failures) == 1


def test_validator_respects_max_failures():
    # Construct many failing samples
    samples = [
        {"image": np.zeros((3, 256, 256)), "label": 0}
    ] * 50
    validator = DataValidator(
        rules=[image_shape_rule(expected=(3, 224, 224))],
        max_failures=5,
    )
    report = validator.validate(samples)
    assert len(report.failures) == 5


def test_report_to_dict_shape():
    validator = DataValidator(rules=[image_shape_rule(expected=(3, 224, 224))])
    report = validator.validate([])
    d = report.to_dict()
    assert d["ok"] is True
    assert d["total_samples"] == 0
    assert d["total_failed"] == 0
    assert d["failure_rate"] == 0.0


def test_report_raise_for_status():
    validator = DataValidator(
        rules=[label_in_range_rule(0, 1)],
    )
    report = validator.validate([{"label": 9}])
    with pytest.raises(DataValidationError) as exc:
        report.raise_for_status()
    assert "1/1 samples failed validation" in str(exc.value)


def test_add_rule_chains():
    validator = DataValidator()
    assert validator.add_rule(label_in_range_rule(0, 1)) is validator
    assert len(validator.rules) == 1
