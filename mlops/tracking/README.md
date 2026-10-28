# MedVision-AI · MLOps Tracking

The `mlops/tracking/` package is the **canonical MLflow integration layer**
for the MedVision-AI Smart Health Analysis & Recommendation System. It
standardizes how every training, evaluation, and inference run is logged,
tagged, and registered — so that downstream audit, fairness, and clinical
governance tooling can rely on a single contract.

> **Scope.** This module handles *tracking only*. Orchestration (Airflow /
> Prefect), feature stores, and serving live under sibling `mlops/`
> packages and are out of scope here.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Logging Recipes](#logging-recipes)
5. [Model Registry](#model-registry)
6. [Examples](#examples)
7. [Testing](#testing)
8. [File Layout](#file-layout)
9. [Operational Notes](#operational-notes)

---

## Architecture

```
                ┌────────────────────────────────────────┐
                │      run_experiment.py  (CLI entry)     │
                └───────────────────┬────────────────────┘
                                 ┌──▼───────────┐
                                 │  configs/    │  (YAML)
                                 │  schemas/    │  (JSON Schema)
                                 └───┬──────────┘
                                     │
   ┌─────────────────────────────────▼───────────────────────────────┐
   │                      ExperimentLogger                            │
   │   (context manager; tags, params, metrics, artifacts, autolog) │
   └────────┬───────────────┬───────────────┬──────────────┬─────────┘
            │               │               │              │
   ┌────────▼─────┐  ┌─────▼──────┐  ┌─────▼─────┐  ┌─────▼─────┐
   │ mlflow_config│  │ metrics_   │  │ artifact_ │  │ model_    │
   │              │  │ logger     │  │ logger    │  │ registry  │
   └──────────────┘  └───────────┘  └───────────┘  └───────────┘
            │
            └────► MLflow Tracking Server (local file store by default)
```

Every module depends on `mlflow_config.get_mlflow_config()` for the
shared `MLflowConfig` singleton, so configuration is loaded exactly
once per process.

---

## Quick Start

```python
from mlops.tracking.experiment_logger import ExperimentLogger

with ExperimentLogger(
    experiment="medvision-classification",
    run_name="baseline_v1",
    tags={"model": "resnet50", "dataset": "chest_xray_v3"},
) as run:
    run.log_params({"lr": 1e-4, "epochs": 30, "batch_size": 32})
    for epoch in range(30):
        train_loss = train_one_epoch(...)
        run.log_metric("train_loss", train_loss, step=epoch)
    run.log_artifact("model.pt", artifact_path="models")
```

That's it. MLflow is configured from `configs/tracking.yaml`, the run
gets default environment tags (Python version, hostname, UTC time) plus
your user tags, and `model.pt` is staged locally before being uploaded.

---

## Configuration

Two YAML files ship out of the box:

| File | Purpose |
|------|---------|
| `configs/tracking.yaml` | MLflow tracking URI, registry URI, default experiment name, run-name prefix, default tags, artifacts staging dir. |
| `configs/experiment_defaults.yaml` | Default experiment/run name, default params, default tags, autolog toggle, reproducibility seed, resource limits. |

Override the tracking config path at runtime:

```bash
export MEDVISION_TRACKING_CONFIG=/etc/medvision/tracking.yaml
```

Both files map 1:1 to fields on the `MLflowConfig` and
`ExperimentLogger` dataclasses respectively; see
`mlflow_config.py` for the authoritative field list.

---

## Logging Recipes

### Metrics with validation

`metrics_logger` validates names and values *before* calling into MLflow.
`NaN`/`inf` values are rejected, so they never silently corrupt the UI
dashboards. Batch logging is atomic — if any single metric in a batch
fails validation, the entire batch is aborted.

```python
from mlops.tracking.metrics_logger import log_metrics, summarize_metrics

log_metrics({"accuracy": 0.91, "f1": 0.83, "auc": 0.95}, step=0)

folds = [{"accuracy": 0.91}, {"accuracy": 0.89}, {"accuracy": 0.93}]
print(summarize_metrics(folds))
# {'accuracy': {'mean': 0.91, 'std': 0.0163, 'n_folds': 3.0}}
```

### Artifacts with deterministic hashing

```python
from mlops.tracking.artifact_logger import log_artifact

hash_hex = log_artifact(
    "model.pt",
    artifact_path="models",
    name="resnet50_epoch_30.pt",
    compute_hash=True,
)
```

The returned SHA256 hash lets downstream consumers detect when a model
artifact's content has not changed (so they can skip re-deployments).

### Staging an artifact locally without logging

```python
from mlops.tracking.artifact_logger import stage_artifact

staged = stage_artifact({"foo": "bar"}, artifacts_dir="/tmp/staging",
                       artifact_path="reports", name="summary.json")
```

Useful for inspection / debugging before committing to MLflow.

---

## Model Registry

The `ModelRegistry` facade wraps `MlflowClient` registry operations:

```python
from mlops.tracking.model_registry import ModelRegistry

reg = ModelRegistry()
reg.register_model(run_id="abc123", name="MedVisionClassifier", tags={"dataset": "v3"})
reg.transition_stage("MedVisionClassifier", version=1, stage="Production")
prod = reg.latest("MedVisionClassifier", stage="Production")
print(prod.to_dict())
```

Valid stages: `None`, `Staging`, `Production`, `Archived`.

---

## Examples

Two runnable examples live under `examples/`:

| File | Demonstrates |
|------|--------------|
| `basic_experiment.py` | Minimal context-managed run with a synthetic loss curve. |
| `classification_experiment.py` | 5-fold cross-validation on a synthetic sklearn dataset, with per-fold and aggregated metrics. |

Run either directly:

```bash
python -m mlops.tracking.examples.basic_experiment
python -m mlops.tracking.examples.classification_experiment
```

---

## Testing

Tests live under `tests/`. They use `pytest` and mock the `mlflow` module
so they can run without an MLflow server.

```bash
pytest mlops/tracking/tests/ -v
```

The suite covers:

- `test_mlflow_config.py` — YAML loading, singleton, missing-file handling.
- `test_metrics_logger.py` — name/value validation, NaN/inf rejection,
  cross-fold summarization.
- `test_experiment_logger.py` — start/end lifecycle, FAILED status on
  exception, param flattening, tag merging.

---

## File Layout

```
mlops/tracking/
├── README.md
├── requirements.txt
├── __init__.py
├── mlflow_config.py          # MLflowConfig singleton + setup()
├── experiment_logger.py      # ExperimentLogger context manager
├── run_experiment.py         # CLI entry point
├── metrics_logger.py         # Validated metric logging
├── artifact_logger.py        # Staging + hashing + MLflow log
├── model_registry.py         # Registry facade
├── tracking_utils.py         # Shared helpers (hashing, tags, env)
├── configs/
│   ├── tracking.yaml
│   └── experiment_defaults.yaml
├── schemas/
│   ├── experiment.schema.json
│   └── metrics.schema.json
├── examples/
│   ├── basic_experiment.py
│   └── classification_experiment.py
├── tests/
│   ├── test_experiment_logger.py
│   ├── test_mlflow_config.py
│   └── test_metrics_logger.py
└── logs/
    └── .gitkeep
```

---

## Operational Notes

- **Local file store by default.** The shipped `tracking.yaml` points at
  `file:///tmp/medvision/mlruns`. Change this to a tracking server URL
  in any shared environment.
- **Healthcare compliance.** Never log PHI (Protected Health Information)
  as a tag, param, or artifact. The `tags` dict in `tracking.yaml` and
  the `healthcare_metadata` block in `metrics.schema.json` are the only
  sanctioned places for subgroup identifiers, and only when pseudonymized.
- **Deterministic artifacts.** `safe_json_dump` writes JSON with
  `sort_keys=True` so that re-logging the same content produces
  byte-identical files — important for MLflow run diffing.
- **Autolog is opt-in.** Framework autologging (`mlflow.sklearn.autolog()`,
  `mlflow.pytorch.autolog()`) can produce large artifacts; it is
  disabled by default. Enable per-run via `ExperimentLogger(autolog=True)`.
