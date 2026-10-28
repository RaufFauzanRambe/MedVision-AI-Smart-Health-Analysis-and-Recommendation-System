# MedVision-AI · MLOps Training

The `mlops/training/` package is the **training orchestration layer**
for the MedVision-AI Smart Health Analysis & Recommendation System.
It standardizes how every model is trained, validated, checkpointed,
and tuned — so that downstream serving, monitoring, and clinical
governance tooling can rely on a single contract.

> **Scope.** This module handles *training only*. Tracking, serving,
> and data pipelines live under sibling `mlops/` packages.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Distributed Training](#distributed-training)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Data Validation](#data-validation)
7. [Checkpointing](#checkpointing)
8. [Reproducibility](#reproducibility)
9. [Testing](#testing)
10. [File Layout](#file-layout)

---

## Architecture

```
                       ┌────────────────────┐
                       │   TrainingConfig   │  (dataclass tree)
                       └──────────┬─────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │              Trainer                  │
              │   (train_model.py)                    │
              └──┬────┬────┬────┬────┬────┬───────────┘
                 │    │    │    │    │    │
       ┌─────────┘    │    │    │    │    └───────────────┐
       ▼              ▼    ▼    ▼    ▼                    ▼
  seed_every-    Data-  Model-  Checkpoint-         Distributed-
  thing          Validator Evaluator Manager        Context
  (reproducibility)  (data_validation) (model_eval) (ckpt_mgr) (distributed)
                       │
                       └──► torch.utils.data.DataLoader
                       └──► torch.nn.Module
                       └──► torch.optim.Optimizer + scheduler
                       └──► mlops.tracking (optional MLflow run)
```

Every component is decoupled: the `Trainer` accepts a model factory
callable and consumes the rest of the module via injected instances
(`DataValidator`, `ModelEvaluator`, `CheckpointManager`). This makes
the trainer equally usable for classification, segmentation, and
detection pipelines by swapping only the factory.

---

## Quick Start

```python
from mlops.training.training_config import TrainingConfig
from mlops.training.train_model import Trainer, default_model_factory
from torch.utils.data import DataLoader

# 1. Load a config
cfg = TrainingConfig.from_yaml("configs/baseline.yaml")

# 2. Build a trainer (default factory uses a small MLP for smoke tests;
#    replace with your own factory in production)
trainer = Trainer(cfg, model_factory=default_model_factory)

# 3. Train
train_loader = DataLoader(your_train_dataset, batch_size=cfg.dataset.batch_size)
val_loader = DataLoader(your_val_dataset, batch_size=cfg.dataset.batch_size)
history = trainer.fit(train_loader, val_loader)

print(f"Best epoch: {history.best_epoch}")
print(f"Best metrics: {history.best_metrics}")
print(f"Total time: {history.total_seconds:.1f}s")
```

---

## Configuration

Three YAML files ship out of the box:

| File | Purpose |
|------|---------|
| `configs/default.yaml` | The base config — every key is set, no overrides. |
| `configs/baseline.yaml` | A short, fast, reproducible baseline for regression-blocking CI. |
| `configs/distributed.yaml` | Single-node, multi-GPU DDP overrides; layered on top of default. |

Layer configs at runtime with `merge_yaml`:

```python
cfg = TrainingConfig.from_yaml("configs/default.yaml")
cfg.merge_yaml("configs/distributed.yaml")  # apply overrides
```

`TrainingConfig` is a tree of dataclasses (`ModelSpec`, `DatasetSpec`,
`OptimizerSpec`, `DistributedSpec`, `CheckpointSpec`, `ReproducibilitySpec`)
so every leaf is type-validated. Unknown keys flow into `cfg.extras`.

---

## Distributed Training

```python
from mlops.training.distributed_training import DistributedContext, distributed
from mlops.training.training_config import DistributedSpec

spec = DistributedSpec(strategy="ddp", world_size=4, backend="nccl")

with distributed(spec) as ctx:
    model = ctx.wrap_model(model)
    # ... train ...
# process group is cleaned up on exit
```

For single-node multi-GPU runs, prefer launching with `torchrun`:

```bash
torchrun --nproc-per-node=4 -m medvision.train \
    --config mlops/training/configs/distributed.yaml
```

Supported strategies:

| Strategy | Use Case |
|----------|----------|
| `single` | Default. Single-GPU or CPU. No process group. |
| `ddp`    | `DistributedDataParallel` on a single node or across nodes. |
| `fsdp`   | `FullyShardedDataParallel` for models that don't fit on a single GPU. |

---

## Hyperparameter Tuning

```python
from mlops.training.hyperparameter_tuning import (
    HyperparameterTuner, suggest_learning_rate, suggest_batch_size,
)

def search(trial):
    return {
        "lr": suggest_learning_rate(trial, low=1e-5, high=1e-2),
        "batch_size": suggest_batch_size(trial),
    }

def train(trial, params):
    # Build a TrainingConfig, run a short training pass,
    # return {"val_accuracy": 0.91}
    ...

tuner = HyperparameterTuner(
    n_trials=20,
    direction="maximize",
    monitor_metric="val_accuracy",
    sampler="tpe",                # or "random"
    log_to_mlflow=True,           # nested MLflow runs
)
report = tuner.run(search, train)
print(report.best_params)
```

When Optuna is not installed, the tuner transparently falls back to
a random search using the same suggestion templates — useful for CI
where adding optuna is undesirable.

---

## Data Validation

A `DataValidator` runs a list of rules over samples before training
starts and on the first batch of every epoch (configurable). The
rules are tiny callables, and we ship a few common ones:

```python
from mlops.training.data_validation import (
    DataValidator, image_shape_rule, label_in_range_rule, non_null_rule,
)

validator = DataValidator(
    rules=[
        image_shape_rule(expected=(3, 224, 224)),
        label_in_range_rule(min_label=0, max_label=1),
        non_null_rule(keys=["image", "label"]),
    ],
    fail_fast=False,
)
report = validator.validate(samples)
report.raise_for_status()  # raises DataValidationError on failure
```

The validator is also exposed on the `Trainer` via `validator=...`.
Pass it in to enable pre-training sanity checks.

---

## Checkpointing

`CheckpointManager` saves every checkpoint with:

- Model state_dict
- Optimizer state_dict (optional, on by default)
- Scheduler state_dict (when provided)
- Current epoch & global step
- The full `TrainingConfig` snapshot
- User-provided extras (e.g. reproducibility state)

```python
from mlops.training.checkpoint_manager import CheckpointManager

mgr = CheckpointManager(
    checkpoint_dir="checkpoints",
    monitor_metric="val_loss",
    monitor_mode="min",
    save_top_k=3,
    save_every_n_epochs=1,
    keep_last_n_periodic=1,
)

# Save
path = mgr.save(
    epoch=10, global_step=1000,
    model=model, optimizer=optimizer, scheduler=scheduler,
    metrics={"val_loss": 0.42, "val_accuracy": 0.91},
    config=cfg.to_dict(),
)

# Resume
state = CheckpointManager.load(
    "checkpoints/ckpt_epoch_0010_best.pt",
    model=model, optimizer=optimizer, scheduler=scheduler,
)
# state.epoch == 10, state.global_step == 1000
```

Retention:

- **Top-K best** by `monitor_metric` are kept; non-best are pruned.
- **Periodic** checkpoints (every `save_every_n_epochs`) are kept on a
  separate rotation of size `keep_last_n_periodic`.

---

## Reproducibility

```python
from mlops.training.reproducibility import seed_everything

state = seed_everything(seed=42, deterministic=True, cudnn_benchmark=False)
print(state.to_dict())
# {
#   "seed": 42,
#   "deterministic": True,
#   "cudnn_benchmark": False,
#   "python_hash_seed": "42",
#   "torch_deterministic": True,
# }
```

`seed_everything` seeds:

- Python's `random` module
- NumPy
- PyTorch (CPU + all CUDA streams)
- `PYTHONHASHSEED` env var (so dict iteration is stable)

A `seed_worker` function is also exposed for use as
`DataLoader(worker_init_fn=...)`.

---

## Testing

Tests live under `tests/`. They use `pytest` and (where useful) skip
themselves gracefully when optional dependencies like `torch` are not
installed.

```bash
pytest mlops/training/tests/ -v
```

The suite covers:

| File | What it tests |
|------|----------------|
| `test_training_config.py` | YAML loading, deep merge, path resolution, env overrides. |
| `test_data_validation.py` | Each rule factory + the validator's collect/fail-fast behavior. |
| `test_model_evaluation.py` | Report logic + tiny torch model evaluation. |
| `test_checkpoint_manager.py` | Save/load round-trip, top-K pruning, best tracking. |

---

## File Layout

```
mlops/training/
├── README.md
├── requirements-training.txt
├── __init__.py
├── train_model.py            # Trainer + TrainingHistory
├── distributed_training.py   # DistributedContext, spawn_workers
├── hyperparameter_tuning.py  # HyperparameterTuner + suggestion templates
├── training_config.py        # TrainingConfig + sub-specs
├── data_validation.py        # DataValidator + rule factories
├── model_evaluation.py       # ModelEvaluator + EvaluationReport
├── checkpoint_manager.py     # CheckpointManager + CheckpointState
├── reproducibility.py        # seed_everything, seed_worker
├── training_utils.py         # AverageMeter, format_eta, ...
├── configs/
│   ├── default.yaml
│   ├── baseline.yaml
│   └── distributed.yaml
└── tests/
    ├── __init__.py
    ├── test_training_config.py
    ├── test_data_validation.py
    ├── test_model_evaluation.py
    └── test_checkpoint_manager.py
```
