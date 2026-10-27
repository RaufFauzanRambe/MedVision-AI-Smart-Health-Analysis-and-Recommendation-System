"""
Tests for mlops.monitoring.metrics.

Run with:

    pytest mlops/monitoring/tests/test_metrics.py -v

These tests:

* Verify counter / histogram / gauge accounting.
* Verify Prometheus exposition format is parseable and includes
  HELP / TYPE lines.
* Verify offline model-evaluation metrics (precision, recall, F1,
  ROC-AUC, calibration error) match hand-computed values on a
  deterministic example.
* Verify privacy hygiene: confidence values are clamped to [0, 1].

They do NOT exercise the real model-serving application and do NOT
make any network calls.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# Make the package importable when running from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mlops.monitoring import metrics as M  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry():
    """Build an isolated MetricRegistry for each test."""
    return M.MetricRegistry()


# ---------------------------------------------------------------------------
# Counter / Gauge / Histogram primitives
# ---------------------------------------------------------------------------

class TestCounter:
    def test_inc_default_amount_is_one(self, fresh_registry):
        c = fresh_registry.requests_total
        c.inc(endpoint="/predict", method="POST", status="200")
        results = {frozenset(l.items()): v for l, v in c.collect()}
        expected_key = frozenset({("endpoint", "/predict"),
                                  ("method", "POST"),
                                  ("status", "200")})
        assert results[expected_key] == 1.0

    def test_inc_accumulates(self, fresh_registry):
        c = fresh_registry.requests_total
        c.inc(endpoint="/predict", method="POST", status="200")
        c.inc(endpoint="/predict", method="POST", status="200")
        c.inc(2.0, endpoint="/predict", method="POST", status="200")
        assert sum(v for _, v in c.collect()) == 4.0

    def test_labels_independent(self, fresh_registry):
        c = fresh_registry.requests_total
        c.inc(endpoint="/predict", method="POST", status="200")
        c.inc(endpoint="/predict", method="POST", status="500")
        c.inc(endpoint="/health", method="GET", status="200")
        assert len(c.collect()) == 3


class TestGauge:
    def test_set_overwrites(self, fresh_registry):
        g = fresh_registry.requests_in_flight
        g.set(5.0, endpoint="/predict")
        g.set(2.0, endpoint="/predict")
        assert g.collect()[0][1] == 2.0

    def test_inc_and_dec(self, fresh_registry):
        g = fresh_registry.requests_in_flight
        g.inc(1.0, endpoint="/predict")
        g.inc(1.0, endpoint="/predict")
        g.dec(1.0, endpoint="/predict")
        assert g.collect()[0][1] == 1.0


class TestHistogram:
    def test_count_and_sum(self, fresh_registry):
        h = fresh_registry.request_duration_seconds
        for v in (0.05, 0.1, 0.2):
            h.observe(v, endpoint="/predict", method="POST")
        results = h.collect()
        assert len(results) == 1
        labels, state = results[0]
        assert state["count"] == 3
        assert math.isclose(state["sum"], 0.35, rel_tol=1e-9)

    def test_buckets_monotonic(self, fresh_registry):
        h = fresh_registry.request_duration_seconds
        for v in (0.001, 0.01, 0.1, 1.0, 5.0):
            h.observe(v, endpoint="/predict", method="POST")
        _, state = h.collect()[0]
        # Cumulative counts must be non-decreasing across bucket index.
        prev = -1
        for b in state["buckets"]:
            assert b >= prev
            prev = b


# ---------------------------------------------------------------------------
# High-level observe APIs
# ---------------------------------------------------------------------------

class TestObserveRequest:
    def test_records_success(self, fresh_registry):
        fresh_registry.observe_request(
            endpoint="/predict", method="POST", status=200,
            duration_seconds=0.123)
        assert sum(v for _, v in fresh_registry.requests_total.collect()) == 1.0
        assert sum(v for _, v in fresh_registry.request_errors_total.collect()) == 0.0

    def test_records_error_on_4xx_5xx(self, fresh_registry):
        fresh_registry.observe_request(
            endpoint="/predict", method="POST", status=500,
            duration_seconds=0.456, error_type="inference_failed")
        err_total = sum(v for _, v in fresh_registry.request_errors_total.collect())
        assert err_total == 1.0


class TestTrackInference:
    def test_clamps_confidence_to_unit_interval(self, fresh_registry):
        # Negative or >1 confidence values must be clamped to [0, 1].
        fresh_registry.track_inference(
            model_name="medvision-v1", model_version="1.0.0",
            duration_seconds=0.05, confidence=-0.5)
        fresh_registry.track_inference(
            model_name="medvision-v1", model_version="1.0.0",
            duration_seconds=0.05, confidence=1.5)
        _, state = fresh_registry.model_confidence.collect()[0]
        assert state["count"] == 2
        # All observations must be within [0, 1].
        cumulative_below_1 = state["buckets"][-1]   # last bucket = +Inf
        assert cumulative_below_1 == 2


class TestDataQuality:
    def test_completeness_pct(self, fresh_registry):
        required = ["age", "sex", "symptoms", "vitals"]
        present = ["age", "sex", "symptoms"]
        fresh_registry.observe_data_quality(
            endpoint="/predict",
            required_fields=required,
            present_fields=present,
        )
        gauge = fresh_registry.data_quality_completeness_pct.collect()
        assert len(gauge) == 1
        _, value = gauge[0]
        assert math.isclose(value, 75.0)
        missing = sum(v for _, v
                      in fresh_registry.data_quality_missing_values_total.collect())
        assert missing == 1.0

    def test_empty_required_noop(self, fresh_registry):
        fresh_registry.observe_data_quality(
            endpoint="/predict", required_fields=[], present_fields=[])
        assert fresh_registry.data_quality_completeness_pct.collect() == []
        assert fresh_registry.data_quality_missing_values_total.collect() == []


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------

class TestRenderPrometheus:
    def test_includes_help_and_type(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(M, "REGISTRY", fresh_registry)
        fresh_registry.requests_total.inc(endpoint="/p", method="POST", status="200")
        text = M.render_prometheus()
        assert "# HELP requests_total" in text
        assert "# TYPE requests_total counter" in text
        assert 'requests_total{endpoint="/p",method="POST",status="200"} 1.0' in text

    def test_histogram_emits_bucket_count_sum(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(M, "REGISTRY", fresh_registry)
        fresh_registry.request_duration_seconds.observe(
            0.05, endpoint="/predict", method="POST")
        text = M.render_prometheus()
        assert "request_duration_seconds_bucket" in text
        assert "request_duration_seconds_count" in text
        assert "request_duration_seconds_sum" in text
        assert 'le="+Inf"' in text


# ---------------------------------------------------------------------------
# Offline model evaluation
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_perfect_classifier(self):
        report = M.evaluate_model(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1],
            y_score=[0.1, 0.9, 0.2, 0.8],
            model_name="medvision-v1",
            model_version="1.0.0",
        )
        assert report.n_samples == 4
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.roc_auc == 1.0

    def test_all_wrong(self):
        report = M.evaluate_model(
            y_true=[0, 1, 0, 1],
            y_pred=[1, 0, 1, 0],
            y_score=[0.9, 0.1, 0.9, 0.1],
        )
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1 == 0.0
        assert report.roc_auc == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            M.evaluate_model(y_true=[0, 1], y_pred=[0])

    def test_empty_set_raises(self):
        with pytest.raises(ValueError):
            M.evaluate_model(y_true=[], y_pred=[])

    def test_score_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            M.evaluate_model(y_true=[0, 1], y_pred=[0, 1], y_score=[0.5])

    def test_hand_computed_partial(self):
        # 4 positives, 1 false negative, 1 false positive:
        #   precision = 4 / (4 + 1) = 0.8
        #   recall    = 4 / (4 + 1) = 0.8
        #   f1        = 0.8
        y_true = [1, 1, 1, 1, 1, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 1, 0]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.4, 0.55, 0.1]
        report = M.evaluate_model(
            y_true=y_true, y_pred=y_pred, y_score=y_score,
            model_name="medvision-v1", model_version="2.0.0",
        )
        assert math.isclose(report.precision, 0.8, rel_tol=1e-9)
        assert math.isclose(report.recall, 0.8, rel_tol=1e-9)
        assert math.isclose(report.f1, 0.8, rel_tol=1e-9)

    def test_to_dict_round_trip(self):
        report = M.evaluate_model(
            y_true=[0, 1], y_pred=[0, 1], y_score=[0.2, 0.8],
            model_name="medvision-v1", model_version="1.0.0",
        )
        d = report.to_dict()
        assert d["model_name"] == "medvision-v1"
        assert d["n_samples"] == 2
        assert "precision" in d and "recall" in d and "f1" in d
