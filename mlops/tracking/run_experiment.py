"""
run_experiment.py
================
Command-line entry point for running a tracked MedVision-AI experiment.

This script is intentionally framework-agnostic: it loads configuration
from ``configs/experiment_defaults.yaml``, applies CLI overrides, sets
up MLflow via :mod:`mlflow_config`, and invokes a user-supplied
``--entrypoint`` callable (a dotted path like
``medvision.pipeline:train``).

The goal is to give every training run a consistent tracking setup
without forcing authors to repeat the boilerplate in their own scripts.

Usage
-----
    python -m mlops.tracking.run_experiment \
        --entrypoint medvision.pipeline:train \
        --params '{"lr": 1e-4, "epochs": 30}' \
        --tags '{"model": "resnet50"}'
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .experiment_logger import ExperimentLogger
from .mlflow_config import get_mlflow_config

logger = logging.getLogger(__name__)


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_json_arg(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {raw!r}: {exc}") from exc


def _resolve_entrypoint(dotted: str):
    if ":" not in dotted:
        raise argparse.ArgumentTypeError(
            "entrypoint must be 'module:attr', got %r" % dotted
        )
    module_name, attr = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Entrypoint {dotted} not found: {exc}") from exc


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tracked MedVision-AI experiment.",
    )
    parser.add_argument("--entrypoint", required=True, help="Dotted path module:attr")
    parser.add_argument("--config", help="Path to a YAML config file (defaults to experiment_defaults.yaml)")
    parser.add_argument("--experiment", help="MLflow experiment name (overrides config)")
    parser.add_argument("--run-name", help="Run name (auto-generated if omitted)")
    parser.add_argument("--params", help="JSON dict of params to log")
    parser.add_argument("--tags", help="JSON dict of tags to log")
    parser.add_argument("--autolog", action="store_true", help="Enable framework autologging")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and entrypoint, but do not run the experiment")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    # Logging setup -------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Resolve config -----------------------------------------------------
    default_path = Path(__file__).resolve().parent / "configs" / "experiment_defaults.yaml"
    config = _load_yaml(args.config) or _load_yaml(str(default_path))
    config["experiment"] = args.experiment or config.get("experiment")
    config["run_name"] = args.run_name or config.get("run_name")
    config["autolog"] = args.autolog or bool(config.get("autolog", False))

    # Resolve entrypoint -------------------------------------------------
    try:
        entrypoint = _resolve_entrypoint(args.entrypoint)
    except (ImportError, argparse.ArgumentTypeError) as exc:
        logger.error("Could not resolve entrypoint: %s", exc)
        return 2

    if args.dry_run:
        logger.info("[dry-run] Resolved entrypoint=%s, config=%s", args.entrypoint, config)
        return 0

    # Configure MLflow ---------------------------------------------------
    get_mlflow_config()

    # Build the run ------------------------------------------------------
    params = {**config.get("params", {}), **_load_json_arg(args.params)}
    tags = {**config.get("tags", {}), **_load_json_arg(args.tags)}

    with ExperimentLogger(
        experiment=config.get("experiment"),
        run_name=config.get("run_name"),
        tags=tags,
        autolog=config.get("autolog", False),
    ) as run:
        if params:
            run.log_params(params)
        try:
            entrypoint(run, params=params)
        except Exception:
            logger.exception("Entrypoint raised an exception; marking run as FAILED.")
            raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
