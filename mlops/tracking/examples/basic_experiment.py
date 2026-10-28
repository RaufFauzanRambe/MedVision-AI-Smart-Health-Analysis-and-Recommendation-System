"""
basic_experiment.py
===================
Minimal example showing how to use the MedVision-AI tracking stack
to log a single run with one parameter, one metric, and one artifact.

Run it directly:

    python -m mlops.tracking.examples.basic_experiment

The example does not require any real model code — it simulates
training by logging a fake loss curve and a small JSON artifact.
"""

from __future__ import annotations

import math

from ..experiment_logger import ExperimentLogger  # type: ignore[no-name-in-module]


def fake_loss_curve(steps: int, base: float = 2.0, decay: float = 0.05) -> list[float]:
    """Return a synthetic decaying loss curve suitable for a demo run."""
    return [base * math.exp(-decay * i) + 0.01 * (i % 5) for i in range(steps)]


def main() -> None:
    with ExperimentLogger(
        experiment="medvision_demo",
        run_name="basic_demo",
        tags={"model": "demo", "dataset": "synthetic"},
    ) as run:
        run.log_params({"lr": 0.01, "steps": 50, "optimizer": "sgd"})

        for step, loss in enumerate(fake_loss_curve(50)):
            run.log_metric("train_loss", float(loss), step=step)

        # Log a final scalar metric + an artifact
        run.log_metric("final_loss", fake_loss_curve(1)[-1])
        run.log_artifact(
            {"run_id": "demo", "final_loss": fake_loss_curve(1)[-1]},
            artifact_path="reports",
            name="summary.json",
        )


if __name__ == "__main__":  # pragma: no cover
    main()
