"""Tests for mlops.tracking.mlflow_config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mlops.tracking.mlflow_config import (
    DEFAULT_CONFIG_PATH,
    MLflowConfig,
    get_mlflow_config,
    reset_mlflow_config,
)


def test_from_dict_defaults():
    cfg = MLflowConfig()
    assert cfg.tracking_uri == "file:///tmp/mlruns"
    assert cfg.experiment_name == "medvision_default"


def test_from_dict_overrides():
    cfg = MLflowConfig.from_dict(
        {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test_exp",
            "tags": {"project": "MedVision-AI"},
            "autolog": True,
        }
    )
    assert cfg.tracking_uri == "http://localhost:5000"
    assert cfg.experiment_name == "test_exp"
    assert cfg.tags["project"] == "MedVision-AI"
    assert cfg.autolog is True


def test_from_yaml_file(tmp_path: Path):
    config_yaml = tmp_path / "tracking.yaml"
    config_yaml.write_text(
        "tracking_uri: file:///tmp/test_mlruns\n"
        "experiment_name: yaml_exp\n"
        "tags:\n  project: MedVision-AI\n"
        "  team: ai-platform\n",
        encoding="utf-8",
    )
    cfg = MLflowConfig.from_yaml(config_yaml)
    assert cfg.tracking_uri == "file:///tmp/test_mlruns"
    assert cfg.experiment_name == "yaml_exp"
    assert cfg.tags["team"] == "ai-platform"


def test_from_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MLflowConfig.from_yaml(tmp_path / "does_not_exist.yaml")


def test_get_mlflow_config_singleton(tmp_path: Path, monkeypatch):
    config_yaml = tmp_path / "tracking.yaml"
    config_yaml.write_text(
        "tracking_uri: file:///tmp/singleton_mlruns\nexperiment_name: singleton\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mlops.tracking.mlflow_config.DEFAULT_CONFIG_PATH", config_yaml
    )
    reset_mlflow_config()
    cfg1 = get_mlflow_config()
    cfg2 = get_mlflow_config()
    assert cfg1 is cfg2
    assert cfg1.experiment_name == "singleton"
    reset_mlflow_config()


def test_default_config_path_exists():
    # Sanity check that the bundled default config file is actually shipped.
    assert DEFAULT_CONFIG_PATH.exists(), (
        "configs/tracking.yaml must ship alongside mlflow_config.py"
    )


def test_setup_requires_mlflow(monkeypatch):
    # If mlflow is missing, setup() should raise a clear RuntimeError.
    import mlops.tracking.mlflow_config as mod

    monkeypatch.setattr(mod, "mlflow", None)
    cfg = MLflowConfig()
    with pytest.raises(RuntimeError):
        cfg.setup()
