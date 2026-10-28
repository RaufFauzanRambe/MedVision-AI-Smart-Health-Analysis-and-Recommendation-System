"""Tests for mlops.training.training_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops.training.training_config import (
    DatasetSpec,
    ModelSpec,
    OptimizerSpec,
    TrainingConfig,
)


# --------------------------------------------------------------------
# Default construction
# --------------------------------------------------------------------


def test_default_construction():
    cfg = TrainingConfig()
    assert cfg.experiment_name == "medvision_default"
    assert cfg.epochs == 30
    assert cfg.model.name == "resnet50"
    assert cfg.dataset.batch_size == 32
    assert cfg.optimizer.lr == 1e-3
    assert cfg.distributed.strategy == "single"
    assert cfg.reproducibility.seed == 42
    assert cfg.checkpoint.save_top_k == 3


def test_subspec_to_dict_round_trip():
    spec = ModelSpec(name="vit_base", num_classes=10)
    d = spec.to_dict()
    assert d["name"] == "vit_base"
    assert d["num_classes"] == 10


# --------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------


def test_from_yaml(tmp_path: Path):
    config_yaml = tmp_path / "test.yaml"
    config_yaml.write_text(
        "experiment_name: test_exp\n"
        "epochs: 5\n"
        "model:\n  name: efficientnet_b0\n  num_classes: 3\n"
        "dataset:\n  batch_size: 64\n"
        "optimizer:\n  lr: 0.0005\n",
        encoding="utf-8",
    )
    cfg = TrainingConfig.from_yaml(config_yaml)
    assert cfg.experiment_name == "test_exp"
    assert cfg.epochs == 5
    assert cfg.model.name == "efficientnet_b0"
    assert cfg.model.num_classes == 3
    assert cfg.dataset.batch_size == 64
    assert cfg.optimizer.lr == 0.0005


def test_from_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TrainingConfig.from_yaml(tmp_path / "missing.yaml")


def test_unknown_keys_go_to_extras():
    cfg = TrainingConfig.from_dict({"experiment_name": "x", "unknown_key": 42})
    assert cfg.extras["unknown_key"] == 42


# --------------------------------------------------------------------
# Merge / override
# --------------------------------------------------------------------


def test_merge_yaml_overrides_top_level(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("epochs: 10\nexperiment_name: base\n", encoding="utf-8")
    override = tmp_path / "override.yaml"
    override.write_text("epochs: 20\n", encoding="utf-8")
    cfg = TrainingConfig.from_yaml(base)
    cfg.merge_yaml(override)
    assert cfg.epochs == 20
    assert cfg.experiment_name == "base"  # not overridden


def test_merge_yaml_recursively_merges_subspecs(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n  name: resnet50\n  num_classes: 2\n"
        "dataset:\n  batch_size: 32\n",
        encoding="utf-8",
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        "model:\n  num_classes: 5\n"  # change only one field
        "dataset:\n  batch_size: 64\n",
        encoding="utf-8",
    )
    cfg = TrainingConfig.from_yaml(base)
    cfg.merge_yaml(override)
    assert cfg.model.name == "resnet50"  # preserved
    assert cfg.model.num_classes == 5    # overridden
    assert cfg.dataset.batch_size == 64   # overridden


def test_merge_dict_chaining():
    cfg = TrainingConfig()
    cfg.merge_dict({"epochs": 100, "model": {"num_classes": 7}})
    assert cfg.epochs == 100
    assert cfg.model.num_classes == 7


# --------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------


def test_to_dict_then_from_dict_round_trip():
    cfg = TrainingConfig(experiment_name="rt", epochs=7)
    cfg.dataset.batch_size = 99
    cfg2 = TrainingConfig.from_dict(cfg.to_dict())
    assert cfg2.experiment_name == "rt"
    assert cfg2.epochs == 7
    assert cfg2.dataset.batch_size == 99


def test_to_yaml_writes_file(tmp_path: Path):
    cfg = TrainingConfig(experiment_name="snap")
    out = cfg.to_yaml(tmp_path / "snapshot.yaml")
    assert out.exists()
    cfg2 = TrainingConfig.from_yaml(out)
    assert cfg2.experiment_name == "snap"


# --------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------


def test_resolve_paths(tmp_path: Path):
    cfg = TrainingConfig()
    cfg.dataset.train_path = "data/train"
    cfg.resolve_paths(tmp_path)
    assert Path(cfg.dataset.train_path).is_absolute()
    assert Path(cfg.dataset.train_path).exists() or "data/train" in cfg.dataset.train_path


def test_env_overrides():
    cfg = TrainingConfig(experiment_name="env_test")
    env = cfg.env_overrides()
    assert env["MEDVISION_EXPERIMENT_NAME"] == "env_test"
    assert env["MEDVISION_SEED"] == "42"


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


def test_distributed_spec_validation():
    with pytest.raises(TypeError):
        # Passing a string where a DistributedSpec is expected raises
        # when from_dict tries to unpack it as a Mapping.
        TrainingConfig.from_dict({"distributed": "bad_string"})
