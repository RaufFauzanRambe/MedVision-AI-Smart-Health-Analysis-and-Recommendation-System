"""Tests for mlops.training.model_evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from mlops.training.model_evaluation import EvaluationReport, ModelEvaluator


# --------------------------------------------------------------------
# EvaluationReport unit tests (no torch required)
# --------------------------------------------------------------------


def test_evaluation_report_to_dict_shape():
    report = EvaluationReport(
        metrics={"accuracy": 0.9, "loss": 0.3},
        per_subgroup={"male": {"accuracy": 0.92}},
        num_samples=100,
        num_batches=4,
        elapsed_seconds=2.5,
    )
    d = report.to_dict()
    assert d["metrics"]["accuracy"] == 0.9
    assert d["per_subgroup"]["male"]["accuracy"] == 0.92
    assert d["num_samples"] == 100
    assert d["elapsed_seconds"] == 2.5


def test_report_update_best_min_mode():
    a = EvaluationReport(metrics={"val_loss": 0.5})
    b = EvaluationReport(metrics={"val_loss": 0.3})
    assert a.update_best(b, monitor="val_loss", mode="min") is True
    # Same or worse should not improve
    assert a.update_best(a, monitor="val_loss", mode="min") is False


def test_report_update_best_max_mode():
    a = EvaluationReport(metrics={"val_accuracy": 0.8})
    b = EvaluationReport(metrics={"val_accuracy": 0.9})
    assert a.update_best(b, monitor="val_accuracy", mode="max") is True


def test_report_update_best_invalid_mode():
    a = EvaluationReport(metrics={"val_loss": 0.5})
    b = EvaluationReport(metrics={"val_loss": 0.3})
    with pytest.raises(ValueError):
        a.update_best(b, monitor="val_loss", mode="bogus")


def test_report_update_best_when_monitor_missing_returns_true():
    """If self has no baseline, any new report with the metric counts as improvement."""
    a = EvaluationReport(metrics={})
    b = EvaluationReport(metrics={"val_loss": 0.3})
    assert a.update_best(b, monitor="val_loss", mode="min") is True


# --------------------------------------------------------------------
# Evaluator on a tiny torch model + sklearn-style targets
# --------------------------------------------------------------------


def _maybe_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError:
        return None, None


def _build_toy_model_and_loader():
    torch, nn = _maybe_torch()
    if torch is None:
        pytest.skip("torch not installed")

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    model = ToyModel()
    # 8 samples, 4 features each
    X = torch.randn(8, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = [(X, y)]
    return model, loader


def test_evaluator_runs_on_tiny_model():
    model, loader = _build_toy_model_and_loader()
    evaluator = ModelEvaluator(metrics=["accuracy"], device="cpu")
    report = evaluator.evaluate(model, loader)
    assert report.num_samples == 8
    assert report.num_batches == 1
    # accuracy must be in [0, 1]
    assert 0.0 <= report.metrics["accuracy"] <= 1.0
    # No loss_fn configured, so "loss" should be absent
    assert "loss" not in report.metrics


def test_evaluator_with_loss_fn():
    torch, _ = _maybe_torch()
    if torch is None:
        pytest.skip("torch not installed")

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    model = ToyModel()
    X = torch.randn(8, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = [(X, y)]
    evaluator = ModelEvaluator(
        metrics=["accuracy"],
        device="cpu",
        loss_fn=lambda out, t: torch.nn.functional.cross_entropy(out, t).item(),
    )
    report = evaluator.evaluate(model, loader)
    assert "loss" in report.metrics
    assert report.metrics["loss"] >= 0


def test_evaluator_handles_unknown_metric_gracefully():
    model, loader = _build_toy_model_and_loader()
    evaluator = ModelEvaluator(metrics=["unknown_metric"], device="cpu")
    report = evaluator.evaluate(model, loader)
    # Should not raise; metric should be NaN and an error entry captured
    assert np.isnan(report.metrics["unknown_metric"])
    assert "__error_unknown_metric" in report.metrics


def test_evaluator_per_subgroup_populated_when_splitter_set():
    """When a subgroup splitter is provided, the report should include per_subgroup."""
    model, loader = _build_toy_model_and_loader()

    def splitter(_sample):
        return None

    evaluator = ModelEvaluator(metrics=["accuracy"], device="cpu", subgroup_splitter=splitter)
    report = evaluator.evaluate(model, loader)
    # All subgroups are None, so per_subgroup should be empty
    assert report.per_subgroup == {}


def test_evaluator_dict_batch():
    """The evaluator should also accept batches as a dict of `image`/`label`."""
    torch, nn = _maybe_torch()
    if torch is None:
        pytest.skip("torch not installed")

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    model = ToyModel()
    X = torch.randn(8, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = [{"image": X, "label": y}]
    evaluator = ModelEvaluator(metrics=["accuracy"], device="cpu")
    report = evaluator.evaluate(model, loader)
    assert report.num_samples == 8
