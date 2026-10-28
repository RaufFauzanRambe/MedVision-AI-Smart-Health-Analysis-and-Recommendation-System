"""Tests for mlops.tracking.experiment_logger."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mlops.tracking.experiment_logger import ExperimentLogger, log_run
from mlops.tracking.mlflow_config import MLflowConfig, reset_mlflow_config


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch):
    """Provide an isolated MLflowConfig pointing at a temp dir."""
    config_yaml = tmp_path / "tracking.yaml"
    config_yaml.write_text(
        "tracking_uri: file:///" + str(tmp_path / "mlruns").replace("\\", "/") + "\n"
        "experiment_name: test_exp\n"
        "run_name_prefix: test\n"
        "artifacts_dir: " + str(tmp_path / "artifacts") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mlops.tracking.mlflow_config.DEFAULT_CONFIG_PATH", config_yaml
    )
    reset_mlflow_config()
    yield MLflowConfig.from_yaml(config_yaml)
    reset_mlflow_config()


def test_experiment_logger_starts_and_ends(tmp_config):
    """The context manager should call mlflow.start_run / end_run."""
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        # ExperimentLogger.start() calls into mlflow_config.setup(),
        # which itself calls mlflow.set_tracking_uri etc. We just need
        # the calls to be present, so we mock the whole mlflow module.
        mock_mlflow.start_run.return_value = None
        with ExperimentLogger(experiment="test_exp", run_name="unit_test") as run:
            assert run._active is True
        assert run._active is False
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")


def test_experiment_logger_marks_run_failed_on_exception(tmp_config):
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = None
        with pytest.raises(ValueError):
            with ExperimentLogger(run_name="bad_run"):
                raise ValueError("boom")
        mock_mlflow.end_run.assert_called_once_with(status="FAILED")


def test_log_run_helper_yields_logger(tmp_config):
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = None
        with log_run(run_name="helper") as run:
            assert isinstance(run, ExperimentLogger)
            assert run._active is True
        assert run._active is False


def test_experiment_logger_log_params_flattens_nested(tmp_config):
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = None
        with ExperimentLogger(run_name="params") as run:
            run.log_params({"optimizer": {"lr": 1e-3, "momentum": 0.9}})
        # The nested dict should have been flattened into dotted keys
        logged = mock_mlflow.log_param.call_args_list
        keys = {call.args[0] for call in logged}
        assert "optimizer.lr" in keys
        assert "optimizer.momentum" in keys


def test_experiment_logger_log_tags_merges_with_environment(tmp_config):
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = None
        with ExperimentLogger(run_name="tags") as run:
            run.log_tags({"model": "resnet50"})
        # set_tags should have been called at least once with a flat dict
        assert mock_mlflow.set_tags.called
        flat_tags = mock_mlflow.set_tags.call_args.args[0]
        assert flat_tags["model"] == "resnet50"


def test_start_idempotent_when_already_active(tmp_config):
    with patch("mlops.tracking.experiment_logger.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = None
        run = ExperimentLogger(run_name="dup").start()
        run.start()  # Should be a no-op
        assert mock_mlflow.start_run.call_count == 1
        run.end()


def test_end_is_noop_when_not_active(tmp_config):
    run = ExperimentLogger(run_name="never_started")
    run.end()  # Should not raise
