"""
model_registry.py
=================
Thin wrapper around ``mlflow.tracking.MlflowClient`` for managing the
MedVision-AI model registry.

The model registry is the canonical source of truth for which model
versions are *Production*, *Staging*, or *Archived*. This module
encapsulates the most common registry workflows so that downstream
services (inference servers, CI gates, audit dashboards) can call a
single, well-typed function rather than reach into MLflow's client
directly.

Typical workflow:

    from mlops.tracking.model_registry import ModelRegistry

    reg = ModelRegistry()
    reg.register_model(
        run_id="abc123",
        name="MedVisionClassifier",
        tags={"dataset": "chest_xray_v3"},
    )
    reg.transition_stage("MedVisionClassifier", version=1, stage="Production")
    reg.latest("MedVisionClassifier", stage="Production")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .mlflow_config import get_mlflow_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelVersionInfo:
    """Lightweight view of a registered model version."""

    name: str
    version: int
    current_stage: str
    run_id: str
    run_link: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "current_stage": self.current_stage,
            "run_id": self.run_id,
            "run_link": self.run_link,
            "description": self.description,
        }


VALID_STAGES = {"None", "Staging", "Production", "Archived"}


class ModelRegistry:
    """A small facade over ``MlflowClient`` registry operations."""

    def __init__(self, config=None) -> None:
        self._config = config or get_mlflow_config()
        self._config.setup()
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient(
            tracking_uri=self._config.tracking_uri,
            registry_uri=self._config.registry_uri or self._config.tracking_uri,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        run_id: str,
        name: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
        wait_for_availability: bool = True,
    ) -> ModelVersionInfo:
        """Register a model artifact from a run into the registry.

        Parameters
        ----------
        run_id:
            The MLflow run ID that produced the artifact.
        name:
            Registered model name. Created if it does not yet exist.
        artifact_path:
            Directory within the run's artifacts containing the model.
        tags:
            Optional tags attached to the registered model.
        wait_for_availability:
            If True, block until the version reaches ``READY`` status.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            self._client.create_registered_model(name)
            logger.info("Created new registered model: %s", name)
        except Exception:
            # Already exists
            logger.debug("Registered model already exists: %s", name)
        version = self._client.create_model_version(
            name=name,
            source=model_uri,
            run_id=run_id,
            tags=tags,
            wait_for_model_completion=wait_for_availability,
        )
        return self._to_info(version)

    # ------------------------------------------------------------------
    # Staging transitions
    # ------------------------------------------------------------------

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> ModelVersionInfo:
        """Move a model version to ``stage`` (Staging/Production/Archived)."""
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage {stage!r}; expected one of {sorted(VALID_STAGES)}")
        version_info = self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        return self._to_info(version_info)

    def latest(self, name: str, stage: str = "Production") -> Optional[ModelVersionInfo]:
        """Return the latest model version for ``name`` at ``stage``."""
        versions = self._client.get_latest_versions(name, stages=[stage])
        if not versions:
            return None
        return self._to_info(versions[0])

    def list_versions(self, name: str) -> List[ModelVersionInfo]:
        """Return all registered versions of ``name``."""
        return [self._to_info(v) for v in self._client.search_model_versions(f"name='{name}'")]

    def delete_version(self, name: str, version: int) -> None:
        """Delete a single model version. Cannot be undone."""
        self._client.delete_model_version(name=name, version=version)

    def set_description(self, name: str, version: int, description: str) -> None:
        """Update the human-readable description of a model version."""
        self._client.update_model_version(
            name=name,
            version=version,
            description=description,
        )

    def set_tags(self, name: str, version: int, tags: Dict[str, str]) -> None:
        """Attach tags to a model version (e.g. ``{"approved_by": "..."}``)."""
        for k, v in tags.items():
            self._client.set_model_version_tag(name=name, version=version, key=k, value=v)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_info(self, mv) -> ModelVersionInfo:
        return ModelVersionInfo(
            name=mv.name,
            version=int(mv.version),
            current_stage=mv.current_stage or "None",
            run_id=getattr(mv, "run_id", "") or "",
            run_link=getattr(mv, "run_link", "") or "",
            description=getattr(mv, "description", "") or "",
        )


__all__ = ["ModelRegistry", "ModelVersionInfo", "VALID_STAGES"]
