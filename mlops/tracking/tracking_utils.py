"""
tracking_utils.py
=================
Shared helper functions used across the MedVision-AI MLOps tracking stack.

These utilities keep the rest of the modules small and focused: instead of
re-implementing timestamp formatting, hash computation, or tag flattening in
every file, each module calls into one of the helpers below.

Highlights
----------
- ``generate_run_name`` produces deterministic, human-readable run names.
- ``compute_model_hash`` produces a SHA256 fingerprint for a serialized
  model artifact so that downstream consumers can detect drift.
- ``flatten_tags`` turns nested dictionaries into MLflow's flat
  ``{"a.b": value}`` style tags.
- ``safe_json_dump`` writes JSON in a stable, sorted fashion so that
  re-logging the same artifact produces byte-identical files (which keeps
  MLflow run diffs meaningful).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_run_name(prefix: str = "medvision", suffix: Optional[str] = None) -> str:
    """Build a deterministic run name of the form ``<prefix>_<yyyymmdd-HHMMSS>[_<suffix>]``.

    Parameters
    ----------
    prefix:
        Short project/feature identifier, e.g. ``"medvision"``.
    suffix:
        Optional descriptor appended at the end (e.g. ``"baseline"``).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [prefix, ts]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def compute_model_hash(model_path: str | Path, chunk_size: int = 8192) -> str:
    """Compute a SHA256 fingerprint for a model artifact on disk.

    The function streams the file in chunks so that very large model files
    (e.g. multi-GB medical-imaging checkpoints) do not have to be loaded
    into memory.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_tags(tags: Mapping[str, Any], separator: str = ".") -> Dict[str, str]:
    """Flatten a nested mapping into ``{"a.b.c": value}`` style tags.

    MLflow tags must be flat strings; nested dictionaries are not supported.
    This helper traverses the input recursively and stringifies leaf values.
    """
    flat: Dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _walk(f"{prefix}{separator}{k}" if prefix else str(k), v)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                _walk(f"{prefix}{separator}{i}" if prefix else str(i), item)
        else:
            flat[prefix] = str(value)

    _walk("", tags)
    return flat


def collect_environment_tags() -> Dict[str, str]:
    """Return a dictionary of environment tags suitable for MLflow runs.

    Includes Python version, platform, hostname, and the current user
    (when available). Useful for reproducibility audits.
    """
    try:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    except Exception:
        user = "unknown"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": user,
        "utc_time": utc_now_iso(),
    }


def safe_json_dump(data: Any, path: str | Path) -> Path:
    """Write ``data`` to ``path`` as sorted, indented JSON.

    Using ``sort_keys=True`` and a fixed indent makes re-logged
    artifacts byte-identical, which simplifies MLflow run diffs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_iterable(value: Any) -> Iterable[Any]:
    """Wrap a scalar in a list so callers can always iterate."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


__all__ = [
    "utc_now_iso",
    "generate_run_name",
    "compute_model_hash",
    "flatten_tags",
    "collect_environment_tags",
    "safe_json_dump",
    "load_json",
    "ensure_iterable",
]
