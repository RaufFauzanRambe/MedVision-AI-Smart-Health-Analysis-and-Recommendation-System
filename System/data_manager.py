"""
MedVision-AI :: data_manager.py
===============================
Thread-safe JSON persistence layer for user profiles and daily health records.

Storage model (two human-readable JSON files inside ``config.DATA_DIR``):
    users.json           {user_id: {user_id, name, age, sex, ...}}
    health_records.json  {user_id: [record, record, ...]}  (records sorted by date)

Robustness features:
    * Atomic writes (write to a temp file, then ``os.replace``) so a crash can
      never leave a half-written store behind.
    * A re-entrant lock guards every read-modify-write cycle.
    * Corrupted stores raise :class:`DataStoreError` instead of being silently
      wiped.
    * ``seed_demo_data()`` ships four realistic sample users whose records
      intentionally cover the full clinical spectrum (healthy / borderline /
      high-risk / improving) so every subsystem can be exercised immediately.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config, validation

__all__ = ["DataStoreError", "DataManager"]


class DataStoreError(RuntimeError):
    """Raised when the JSON store is unreadable or structurally broken."""


class DataManager:
    """JSON-backed store for users and health records.

    Args:
        data_dir: optional directory override (defaults to ``config.DATA_DIR``,
            which itself honors the ``MEDVISION_DATA_DIR`` environment variable).
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(config.DATA_DIR)
        self.users_file = self.data_dir / "users.json"
        self.records_file = self.data_dir / "health_records.json"
        self._lock = threading.RLock()
        self._ensure_store()

    # ------------------------------------------------------------------
    # Low-level JSON plumbing
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        """Create the data directory and empty stores on first use."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.users_file.exists():
            self._atomic_write(self.users_file, {})
        if not self.records_file.exists():
            self._atomic_write(self.records_file, {})

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise DataStoreError(
                f"{path.name} is corrupted (invalid JSON: {exc}). "
                "Restore it from backup or delete it to start fresh."
            ) from exc
        except OSError as exc:
            raise DataStoreError(f"Cannot read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise DataStoreError(f"{path.name} must contain a JSON object")
        return data

    @staticmethod
    def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
        """Serialize to a temp file in the same dir, then atomically replace."""
        fd, tmp_path = tempfile.mkstemp(
            prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError as exc:
            # best effort cleanup so tmp files do not pile up
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise DataStoreError(f"Cannot write {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    def add_user(self, profile: Dict[str, Any],
                 allow_overwrite: bool = False) -> Dict[str, Any]:
        """Validate and persist a new user profile. Returns the stored dict."""
        clean, _warnings = validation.validate_user(profile)
        with self._lock:
            users = self._read_json(self.users_file)
            uid = clean["user_id"]
            if uid in users and not allow_overwrite:
                raise DataStoreError(
                    f"User '{uid}' already exists; use allow_overwrite=True "
                    "or update_user()."
                )
            clean.setdefault("created_at", date.today().isoformat())
            users[uid] = clean
            self._atomic_write(self.users_file, users)
        return clean

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._read_json(self.users_file).get(str(user_id))

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``updates`` into an existing profile (re-validated)."""
        with self._lock:
            users = self._read_json(self.users_file)
            if user_id not in users:
                raise DataStoreError(f"Unknown user '{user_id}'")
            merged = {**users[user_id], **updates, "user_id": user_id}
            clean, _warnings = validation.validate_user(merged)
            clean["created_at"] = users[user_id].get(
                "created_at", date.today().isoformat()
            )
            users[user_id] = clean
            self._atomic_write(self.users_file, users)
            return clean

    def delete_user(self, user_id: str, delete_records: bool = True) -> bool:
        """Remove a user (and by default all their health records)."""
        with self._lock:
            users = self._read_json(self.users_file)
            if user_id not in users:
                return False
            del users[user_id]
            self._atomic_write(self.users_file, users)
            if delete_records:
                records = self._read_json(self.records_file)
                records.pop(user_id, None)
                self._atomic_write(self.records_file, records)
            return True

    def list_users(self) -> List[Dict[str, Any]]:
        """All user profiles, ordered by user_id."""
        with self._lock:
            users = self._read_json(self.users_file)
        return [users[key] for key in sorted(users)]

    # ------------------------------------------------------------------
    # Health-record CRUD
    # ------------------------------------------------------------------

    def add_record(self, record: Dict[str, Any],
                   profile_height: Optional[float] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and append one health record.

        Args:
            record: raw record dict (must contain user_id + weight_kg).
            profile_height: when the record lacks ``height_cm``, this profile
                height is injected before validation (e.g. so BMI can be
                computed from the user profile).

        Returns:
            (stored_record, validation_warnings)

        Raises:
            DataStoreError: if the user does not exist.
            validation.ValidationError: if the record is invalid.
        """
        raw = dict(record)
        if raw.get("height_cm") is None and profile_height is not None:
            raw["height_cm"] = profile_height
        clean, warnings = validation.validate_record(raw)
        with self._lock:
            users = self._read_json(self.users_file)
            uid = clean["user_id"]
            if uid not in users:
                raise DataStoreError(
                    f"Unknown user '{uid}'; register the user first."
                )
            records = self._read_json(self.records_file)
            bucket = records.setdefault(uid, [])
            # Replace any same-day record: the newest measurement wins.
            bucket = [r for r in bucket if r.get("date") != clean["date"]]
            bucket.append(clean)
            bucket.sort(key=lambda r: r.get("date", ""))
            records[uid] = bucket
            self._atomic_write(self.records_file, records)
        return clean, warnings

    def get_records(self, user_id: str) -> List[Dict[str, Any]]:
        """All records for a user, oldest -> newest."""
        with self._lock:
            records = self._read_json(self.records_file)
            return list(records.get(str(user_id), []))

    def get_latest_record(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            records = self._read_json(self.records_file)
            bucket = records.get(str(user_id), [])
            return dict(bucket[-1]) if bucket else None

    def delete_record(self, user_id: str, record_date: str) -> bool:
        with self._lock:
            records = self._read_json(self.records_file)
            bucket = records.get(user_id, [])
            remaining = [r for r in bucket if r.get("date") != record_date]
            if len(remaining) == len(bucket):
                return False
            records[user_id] = remaining
            self._atomic_write(self.records_file, records)
            return True

    def count_records(self, user_id: Optional[str] = None) -> int:
        with self._lock:
            records = self._read_json(self.records_file)
            if user_id is None:
                return sum(len(v) for v in records.values())
            return len(records.get(str(user_id), []))

    # ------------------------------------------------------------------
    # Demo data
    # ------------------------------------------------------------------

    def seed_demo_data(self, force: bool = False) -> Dict[str, int]:
        """Load the built-in sample dataset (4 users, dated relative to today).

        Args:
            force: re-seed even when users already exist, replacing the demo
                users' data (non-demo users are left untouched).

        Returns:
            {"users_added": int, "records_added": int}
        """
        existing = {u["user_id"] for u in self.list_users()}
        if existing and not force:
            return {"users_added": 0, "records_added": 0}

        today = date.today()
        users_added = records_added = 0

        for user_id, profile, history in _build_demo_dataset(today):
            if user_id not in existing:
                self.add_user(profile)
                users_added += 1
            else:
                self.update_user(user_id, profile)
            for raw_record in history:
                stored, _w = self.add_record(raw_record)
                records_added += 1

        return {"users_added": users_added, "records_added": records_added}

    # ------------------------------------------------------------------
    # Bulk export
    # ------------------------------------------------------------------

    def export_all(self) -> Dict[str, Any]:
        """Snapshot of the full store (users + records), useful for backups."""
        with self._lock:
            return {
                "users": self._read_json(self.users_file),
                "records": self._read_json(self.records_file),
            }


# ---------------------------------------------------------------------------
# Built-in demo dataset
# ---------------------------------------------------------------------------
# Four personas:
#   u001  Sarah Chen      34 F  healthy & active          -> "Excellent"
#   u002  Marcus Rivera   52 M  hypertensive, prediabetic -> "Very High" risk
#   u003  Aisha Patel     28 F  borderline, sedentary     -> "Fair"
#   u004  David Kim       45 M  overweight, improving     -> trend demo

def _build_demo_dataset(today: date) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Return [(user_id, profile, [records...])], dates relative to *today*."""

    def days_ago(n: int) -> str:
        return (today - timedelta(days=n)).isoformat()

    sarah = (
        "u001",
        {
            "user_id": "u001", "name": "Sarah Chen", "age": 34, "sex": "female",
            "height_cm": 165.0, "email": "sarah.chen@example.com",
            "family_history": {"hypertension": False, "diabetes": False,
                               "heart_disease": False},
        },
        [
            {"user_id": "u001", "date": days_ago(60), "height_cm": 165.0,
             "weight_kg": 59.5, "systolic_bp": 114, "diastolic_bp": 74,
             "heart_rate": 66, "fasting_glucose": 90, "total_cholesterol": 182,
             "hdl_cholesterol": 64, "ldl_cholesterol": 96, "sleep_hours": 7.4,
             "steps": 10200, "water_intake_l": 2.3, "smoking": False,
             "alcohol_drinks_per_week": 2, "exercise_sessions_per_week": 4},
            {"user_id": "u001", "date": days_ago(0), "height_cm": 165.0,
             "weight_kg": 59.2, "systolic_bp": 112, "diastolic_bp": 73,
             "heart_rate": 65, "fasting_glucose": 89, "total_cholesterol": 178,
             "hdl_cholesterol": 66, "ldl_cholesterol": 94, "sleep_hours": 7.6,
             "steps": 11500, "water_intake_l": 2.4, "smoking": False,
             "alcohol_drinks_per_week": 2, "exercise_sessions_per_week": 5},
        ],
    )

    marcus = (
        "u002",
        {
            "user_id": "u002", "name": "Marcus Rivera", "age": 52, "sex": "male",
            "height_cm": 178.0, "email": "m.rivera@example.com",
            "family_history": {"hypertension": True, "diabetes": True,
                               "heart_disease": True},
        },
        [
            {"user_id": "u002", "date": days_ago(90), "height_cm": 178.0,
             "weight_kg": 93.5, "systolic_bp": 146, "diastolic_bp": 92,
             "heart_rate": 80, "fasting_glucose": 116, "total_cholesterol": 232,
             "hdl_cholesterol": 42, "ldl_cholesterol": 150, "sleep_hours": 5.9,
             "steps": 4600, "water_intake_l": 1.5, "smoking": True,
             "alcohol_drinks_per_week": 8, "exercise_sessions_per_week": 1},
            {"user_id": "u002", "date": days_ago(30), "height_cm": 178.0,
             "weight_kg": 94.2, "systolic_bp": 149, "diastolic_bp": 95,
             "heart_rate": 83, "fasting_glucose": 121, "total_cholesterol": 238,
             "hdl_cholesterol": 40, "ldl_cholesterol": 156, "sleep_hours": 5.6,
             "steps": 3900, "water_intake_l": 1.3, "smoking": True,
             "alcohol_drinks_per_week": 9, "exercise_sessions_per_week": 1},
            {"user_id": "u002", "date": days_ago(0), "height_cm": 178.0,
             "weight_kg": 92.1, "systolic_bp": 151, "diastolic_bp": 97,
             "heart_rate": 85, "fasting_glucose": 124, "total_cholesterol": 241,
             "hdl_cholesterol": 39, "ldl_cholesterol": 158, "sleep_hours": 5.4,
             "steps": 4100, "water_intake_l": 1.4, "smoking": True,
             "alcohol_drinks_per_week": 8, "exercise_sessions_per_week": 1},
        ],
    )

    aisha = (
        "u003",
        {
            "user_id": "u003", "name": "Aisha Patel", "age": 28, "sex": "female",
            "height_cm": 162.0,
            "family_history": {"diabetes": True},
        },
        [
            {"user_id": "u003", "date": days_ago(45), "height_cm": 162.0,
             "weight_kg": 67.4, "systolic_bp": 125, "diastolic_bp": 81,
             "heart_rate": 75, "fasting_glucose": 102, "total_cholesterol": 208,
             "hdl_cholesterol": 53, "ldl_cholesterol": 121, "sleep_hours": 6.1,
             "steps": 5400, "water_intake_l": 1.5, "smoking": False,
             "alcohol_drinks_per_week": 3, "exercise_sessions_per_week": 2},
            {"user_id": "u003", "date": days_ago(0), "height_cm": 162.0,
             "weight_kg": 66.8, "systolic_bp": 124, "diastolic_bp": 80,
             "heart_rate": 74, "fasting_glucose": 101, "total_cholesterol": 205,
             "hdl_cholesterol": 54, "ldl_cholesterol": 118, "sleep_hours": 6.3,
             "steps": 6200, "water_intake_l": 1.7, "smoking": False,
             "alcohol_drinks_per_week": 3, "exercise_sessions_per_week": 2},
        ],
    )

    david = (
        "u004",
        {
            "user_id": "u004", "name": "David Kim", "age": 45, "sex": "male",
            "height_cm": 180.0, "email": "david.kim@example.com",
            "family_history": {"hypertension": True, "heart_disease": False,
                               "diabetes": False},
        },
        [
            {"user_id": "u004", "date": days_ago(120), "height_cm": 180.0,
             "weight_kg": 97.8, "systolic_bp": 142, "diastolic_bp": 90,
             "heart_rate": 78, "fasting_glucose": 106, "total_cholesterol": 221,
             "hdl_cholesterol": 45, "ldl_cholesterol": 138, "sleep_hours": 6.5,
             "steps": 5200, "water_intake_l": 1.8, "smoking": False,
             "alcohol_drinks_per_week": 5, "exercise_sessions_per_week": 2},
            {"user_id": "u004", "date": days_ago(60), "height_cm": 180.0,
             "weight_kg": 94.6, "systolic_bp": 137, "diastolic_bp": 88,
             "heart_rate": 75, "fasting_glucose": 103, "total_cholesterol": 214,
             "hdl_cholesterol": 47, "ldl_cholesterol": 131, "sleep_hours": 6.9,
             "steps": 7100, "water_intake_l": 2.0, "smoking": False,
             "alcohol_drinks_per_week": 4, "exercise_sessions_per_week": 3},
            {"user_id": "u004", "date": days_ago(0), "height_cm": 180.0,
             "weight_kg": 92.0, "systolic_bp": 133, "diastolic_bp": 86,
             "heart_rate": 72, "fasting_glucose": 100, "total_cholesterol": 209,
             "hdl_cholesterol": 49, "ldl_cholesterol": 126, "sleep_hours": 7.2,
             "steps": 8600, "water_intake_l": 2.2, "smoking": False,
             "alcohol_drinks_per_week": 3, "exercise_sessions_per_week": 4},
        ],
    )

    return [sarah, marcus, aisha, david]
