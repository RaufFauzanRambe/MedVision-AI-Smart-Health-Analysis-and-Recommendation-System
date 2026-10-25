"""
MedVision-AI :: Smart Health Analysis and Recommendation System
================================================================

A dependency-light, educational Python package that:

  * validates and stores daily health records (JSON persistence),
  * classifies vitals/labs against clinical reference ranges,
  * computes a weighted composite health score (0-100),
  * estimates cardiometabolic risk with a hybrid engine
    (interpretable rule model + optional scikit-learn RandomForest),
  * generates prioritized, personalized recommendations and a 7-day plan.

--------------------------------------------------------------------------
MEDICAL DISCLAIMER
This software is an educational demonstration. It is NOT a medical device
and does NOT provide medical diagnosis, treatment, or advice. Always consult
a qualified healthcare professional regarding any medical condition or
before making changes to your diet, exercise, or medication.
--------------------------------------------------------------------------

Quick start::

    >>> from System import MedVisionSystem
    >>> system = MedVisionSystem()
    >>> system.seed_demo_data()
    {'users_added': 4, 'records_added': 10}
    >>> report = system.full_report("u002")
    >>> print(report["text_report"])

Or watch the whole showcase with the built-in demo::

    python -c "import System; System.run_demo()"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .data_manager import DataManager, DataStoreError
from .health_analyzer import HealthAnalyzer, generate_text_report
from .model_manager import ModelManager
from .recommendation_engine import RecommendationEngine
from .validation import ValidationError

__version__ = config.VERSION
__all__ = [
    "MedVisionSystem", "create_system", "run_demo",
    "DataManager", "DataStoreError", "HealthAnalyzer",
    "RecommendationEngine", "ModelManager", "ValidationError",
    "generate_text_report", "__version__",
]


class MedVisionSystem:
    """Facade wiring together storage, analysis, risk modeling and advice.

    This is the only class most callers need.

    Args:
        data_dir: optional override for the JSON data directory.
        model_file: optional override for the persisted model artifact.

    Example:
        >>> system = MedVisionSystem()
        >>> system.register_user("u100", "Jane Doe", age=40, sex="female",
        ...                      height_cm=168)
        >>> system.add_record("u100", weight_kg=70, systolic_bp=118,
        ...                   diastolic_bp=76, steps=9000)
        >>> result = system.analyze_user("u100")
        >>> result["health_score"]
        91.4
    """

    def __init__(self, data_dir: Optional[Path] = None,
                 model_file: Optional[Path] = None):
        self.data_manager = DataManager(data_dir=data_dir)
        self.model_manager = ModelManager(model_file=model_file)
        self.analyzer = HealthAnalyzer(model_manager=self.model_manager)
        self.recommender = RecommendationEngine()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def register_user(self, user_id: str, name: str, age: int, sex: str,
                      height_cm: Optional[float] = None,
                      family_history: Optional[Dict[str, bool]] = None,
                      **extra: Any) -> Dict[str, Any]:
        """Create a user profile (validated). Returns the stored profile."""
        profile: Dict[str, Any] = {
            "user_id": user_id, "name": name, "age": age, "sex": sex,
        }
        if height_cm is not None:
            profile["height_cm"] = height_cm
        if family_history is not None:
            profile["family_history"] = family_history
        profile.update(extra)
        return self.data_manager.add_user(profile)

    def list_users(self) -> List[Dict[str, Any]]:
        return self.data_manager.list_users()

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.data_manager.get_user(user_id)

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge validated updates into an existing profile."""
        return self.data_manager.update_user(user_id, updates)

    def delete_user(self, user_id: str, delete_records: bool = True) -> bool:
        """Remove a user (and by default their records). Returns True if gone."""
        return self.data_manager.delete_user(user_id, delete_records)

    def delete_record(self, user_id: str, record_date: str) -> bool:
        """Delete one record by ISO date. Returns True if removed."""
        return self.data_manager.delete_record(user_id, record_date)

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def add_record(self, user_id: str, **metrics: Any) -> Dict[str, Any]:
        """Validate and store one daily health record.

        Metrics are passed as keyword arguments, e.g.::

            system.add_record("u100", weight_kg=68, systolic_bp=115,
                              diastolic_bp=75, fasting_glucose=92,
                              steps=10500, sleep_hours=7.5)

        Height falls back to the profile value when omitted. Returns the
        stored record; non-fatal data oddities are listed in the returned
        ``_warnings`` key.
        """
        profile = self.data_manager.get_user(user_id)
        stored, warnings = self.data_manager.add_record(
            {"user_id": user_id, **metrics},
            profile_height=(profile or {}).get("height_cm"),
        )
        if warnings:
            stored["_warnings"] = warnings
        return stored

    def get_records(self, user_id: str) -> List[Dict[str, Any]]:
        return self.data_manager.get_records(user_id)

    # ------------------------------------------------------------------
    # Analysis & recommendations
    # ------------------------------------------------------------------

    def analyze_user(self, user_id: str,
                     record_date: Optional[str] = None) -> Dict[str, Any]:
        """Full analysis for a user's latest record (or a specific date)."""
        user = self.data_manager.get_user(user_id)
        if user is None:
            raise DataStoreError(f"Unknown user '{user_id}'")
        history = self.data_manager.get_records(user_id)
        if not history:
            raise DataStoreError(
                f"No health records for '{user_id}'; add a record first."
            )
        if record_date is None:
            record = history[-1]
        else:
            match = [r for r in history if r.get("date") == record_date]
            if not match:
                raise DataStoreError(
                    f"No record dated {record_date} for '{user_id}'"
                )
            record = match[0]
        return self.analyzer.analyze(record, user=user, history=history)

    def get_recommendations(self, user_id: str,
                            analysis: Optional[Dict[str, Any]] = None
                            ) -> List[Dict[str, Any]]:
        """Prioritized recommendations for the latest analysis."""
        if analysis is None:
            analysis = self.analyze_user(user_id)
        return self.recommender.generate(
            analysis, user=self.data_manager.get_user(user_id)
        )

    def full_report(self, user_id: str) -> Dict[str, Any]:
        """Everything at once: analysis + recommendations + weekly plan +
        a formatted plain-text report."""
        analysis = self.analyze_user(user_id)
        recommendations = self.get_recommendations(user_id, analysis)
        weekly_plan = self.recommender.build_weekly_plan(
            analysis, user=self.data_manager.get_user(user_id)
        )
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "weekly_plan": weekly_plan,
            "text_report": generate_text_report(analysis),
            "disclaimer": self.recommender.get_disclaimer(),
        }

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def train_model(self, augment: int = 12) -> Dict[str, Any]:
        """(Re)train the optional RandomForest over all stored records."""
        users = {u["user_id"]: u for u in self.data_manager.list_users()}
        pairs = []
        for uid, bucket in self.data_manager.export_all()["records"].items():
            user = users.get(uid, {})
            pairs.extend((record, user) for record in bucket)
        return self.model_manager.train(pairs, augment=augment)

    def model_status(self) -> Dict[str, Any]:
        return self.model_manager.get_status()

    def seed_demo_data(self, force: bool = False) -> Dict[str, int]:
        """Load the 4 built-in sample users (no-op if data already exists)."""
        return self.data_manager.seed_demo_data(force=force)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

def create_system(data_dir: Optional[Path] = None,
                  model_file: Optional[Path] = None) -> MedVisionSystem:
    """Factory returning a fully wired :class:`MedVisionSystem`."""
    return MedVisionSystem(data_dir=data_dir, model_file=model_file)


def run_demo() -> None:
    """End-to-end showcase: seed data, score users, show recommendations,
    train the model if possible, and print a high-risk user's report.

    Run with:  python -c "import System; System.run_demo()"
    """
    width = 68
    bar = "=" * width
    print(bar)
    print(f"{config.APP_FULL_NAME}".center(width))
    print(f"v{__version__} - educational demo, not a medical device".center(width))
    print(bar)

    system = create_system()
    seeded = system.seed_demo_data()
    print(f"\n[1] Demo dataset: {seeded['users_added']} user(s) added, "
          f"{seeded['records_added']} record(s) stored.")

    print("\n[2] Registered users:")
    for user in system.list_users():
        print(f"    {user['user_id']}  {user['name']:<15} "
              f"{user['age']:>3} y  {user['sex']}")

    print("\n[3] Composite health scores (latest record per user):")
    for user in system.list_users():
        try:
            analysis = system.analyze_user(user["user_id"])
            risk = analysis["risk_assessment"]
            print(f"    {user['name']:<15} score {analysis['health_score']:>5} / 100 "
                  f"({analysis['health_status']:<11}) | risk "
                  f"{risk['probability'] * 100:>3.0f}% ({risk['level']})")
        except DataStoreError as exc:
            print(f"    {user['name']:<15} skipped ({exc})")

    print("\n[4] Model status:", system.model_status())
    train_result = system.train_model()
    print("    Training result:", train_result)
    print("    Model status after training:", system.model_status())

    print(f"\n[5] Full report for the highest-risk demo user (u002 Marcus Rivera)")
    print(bar)
    report = system.full_report("u002")
    print(report["text_report"])
    print("\nTOP RECOMMENDATIONS")
    for i, rec in enumerate(report["recommendations"][:5], start=1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
        for action in rec["actions"][:2]:
            print(f"       - {action}")
    print("\n" + report["disclaimer"])
    print(bar)


def _package_banner() -> str:
    return (
        f"{config.APP_NAME} v{__version__} - {config.APP_FULL_NAME}\n"
        f"DISCLAIMER: {config.DISCLAIMER}"
    )


# Load and expose package metadata lazily (kept private to avoid confusion).
def info() -> Dict[str, Any]:
    """Return the parsed ``medvision_info.json`` metadata."""
    info_path = Path(__file__).parent / "medvision_info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"name": config.APP_NAME, "version": __version__}
