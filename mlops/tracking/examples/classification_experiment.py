"""
classification_experiment.py
============================
End-to-end example that runs a k-fold cross-validation on a small
synthetic classification problem and logs the results through the
MedVision-AI tracking stack.

The example is intentionally lightweight (sklearn + a tiny synthetic
dataset) so it runs in seconds on any laptop. Replace the dataset and
model in your own pipeline to see the same tracking flow over real
healthcare data.

Run it directly:

    python -m mlops.tracking.examples.classification_experiment
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..experiment_logger import ExperimentLogger  # type: ignore[no-name-in-module]
from ..metrics_logger import summarize_metrics  # type: ignore[no-name-in-module]

logger = logging.getLogger(__name__)


def _build_dataset(n_samples: int = 1000, n_features: int = 20, seed: int = 42):
    """Return a small synthetic binary-classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        random_state=seed,
    )
    return X, y


def main(n_splits: int = 5) -> None:
    X, y = _build_dataset()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []

    with ExperimentLogger(
        experiment="medvision_classification",
        run_name="rf_kfold",
        tags={"model": "random_forest", "dataset": "synthetic_binary"},
    ) as run:
        run.log_params(
            {
                "n_splits": n_splits,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "model": "RandomForestClassifier",
                "n_estimators": 100,
            }
        )

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            clf = RandomForestClassifier(n_estimators=100, random_state=42 + fold)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_va)[:, 1]
            preds = (proba >= 0.5).astype(int)
            fold_metrics.append(
                {
                    "accuracy": float(accuracy_score(y_va, preds)),
                    "f1": float(f1_score(y_va, preds)),
                    "auc": float(roc_auc_score(y_va, proba)),
                }
            )
            for name, value in fold_metrics[-1].items():
                run.log_metric(f"fold_{fold}_{name}", value)

        summary = summarize_metrics(fold_metrics)
        for metric, stats in summary.items():
            for stat, value in stats.items():
                run.log_metric(f"{metric}_{stat}", value)

        run.log_artifact(
            {"summary": summary, "fold_metrics": fold_metrics},
            artifact_path="reports",
            name="cv_summary.json",
        )
        logger.info("CV summary: %s", summary)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
