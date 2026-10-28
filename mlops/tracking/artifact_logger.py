"""
artifact_logger.py
==================
Helpers for persisting artifacts (model checkpoints, evaluation tables,
sample images, configuration dumps) into MLflow runs.

Logging artifacts directly to MLflow via ``mlflow.log_artifact`` works
for one-off files, but production runs typically want to:

1. Stage artifacts locally first (so they survive an MLflow upload
   failure and can be inspected post-mortem).
2. Hash each artifact so the same file can be re-logged as a no-op
   when content is unchanged.
3. Group artifacts into MLflow *artifact directories* so that
   downstream consumers can pull ``models/``, ``reports/``, etc.
   separately.

This module implements that staging + dedup flow on top of MLflow's
primitive API.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from .tracking_utils import compute_model_hash, safe_json_dump


ArtifactLike = Union[str, Path, Dict]


def stage_artifact(
    source: ArtifactLike,
    artifacts_dir: Union[str, Path],
    artifact_path: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Stage an artifact on local disk before logging to MLflow.

    Parameters
    ----------
    source:
        One of:
          * a path to an existing file or directory,
          * a dict (JSON-serializable) that will be written as
            ``<name>.json``.
    artifacts_dir:
        Root staging directory. The artifact will be placed under
        ``<artifacts_dir>/<artifact_path>/<name>``.
    artifact_path:
        Subdirectory inside the staging area (e.g. ``"models"``,
        ``"reports"``). Mirrors MLflow's ``artifact_path`` concept.
    name:
        Final filename. Required for dict inputs; optional for files
        (defaults to the source filename).

    Returns
    -------
    Path
        Absolute path to the staged artifact.
    """
    artifacts_dir = Path(artifacts_dir)
    target_dir = artifacts_dir / (artifact_path or "")
    target_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, dict):
        fname = name or "artifact.json"
        target = target_dir / fname
        return safe_json_dump(source, target)

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Artifact source not found: {src}")

    fname = name or src.name
    target = target_dir / fname
    if src.is_dir():
        shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, target)
    return target


def log_artifact(
    source: ArtifactLike,
    artifact_path: Optional[str] = None,
    name: Optional[str] = None,
    compute_hash: bool = False,
) -> Optional[str]:
    """Stage and log a single artifact to the current MLflow run.

    Parameters
    ----------
    source:
        File path, directory path, or JSON-serializable dict.
    artifact_path:
        Subdirectory inside the MLflow run artifact store.
    name:
        Final filename (required for dict sources).
    compute_hash:
        If True, compute and return a SHA256 of the staged artifact.
        Returned as a hex string. Hashes are not computed for dict
        artifacts (they are already deterministically serialized).

    Returns
    -------
    Optional[str]
        The SHA256 hash of the staged artifact, or ``None`` when
        ``compute_hash`` is False or the artifact is a dict.
    """
    import mlflow

    from .mlflow_config import get_mlflow_config

    cfg = get_mlflow_config()
    staged = stage_artifact(
        source=source,
        artifacts_dir=cfg.artifacts_dir,
        artifact_path=artifact_path,
        name=name,
    )
    mlflow.log_artifact(str(staged), artifact_path=artifact_path)

    if compute_hash and staged.is_file():
        return compute_model_hash(staged)
    return None


def log_artifacts_batch(
    items: Iterable[Dict],
) -> Dict[str, Optional[str]]:
    """Log a batch of artifacts.

    Each item is a dict with the same keys accepted by
    :func:`log_artifact` (``source``, ``artifact_path``, ``name``,
    ``compute_hash``). Returns a mapping of ``artifact_path/name``
    to the artifact hash (or ``None``).
    """
    out: Dict[str, Optional[str]] = {}
    for item in items:
        key = f"{item.get('artifact_path', '')}/{item.get('name', '')}".lstrip("/")
        out[key] = log_artifact(
            source=item["source"],
            artifact_path=item.get("artifact_path"),
            name=item.get("name"),
            compute_hash=item.get("compute_hash", False),
        )
    return out


def log_temporary_text(
    content: str,
    name: str,
    artifact_path: Optional[str] = None,
) -> None:
    """Write ``content`` to a temporary file and log it as an artifact.

    Convenience wrapper for ad-hoc text artifacts such as model cards,
    run summaries, or markdown reports.
    """
    import mlflow

    with tempfile.NamedTemporaryFile("w", suffix=Path(name).suffix, delete=False, encoding="utf-8") as fh:
        fh.write(content)
        tmp_path = fh.name
    try:
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


__all__ = [
    "stage_artifact",
    "log_artifact",
    "log_artifacts_batch",
    "log_temporary_text",
]
