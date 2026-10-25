"""
MedVision-AI :: config.py
=========================
Central configuration for the MedVision-AI Smart Health Analysis and
Recommendation System.

This module owns:
    * Package identity / versioning (loaded from ``medvision_info.json``)
    * Filesystem locations (auto-created; overridable via environment variables)
    * Clinical reference ranges (BMI, blood pressure, glucose, lipids, ...)
    * Composite-score weights and ideal ranges used by the analysis engine
    * Risk-factor point values used by the rule-based risk estimator
    * Recommendation priority ordering and the global medical disclaimer

Everything here is dependency-free (Python standard library only).

Supported environment overrides:
    MEDVISION_DATA_DIR     where users.json / health_records.json are stored
    MEDVISION_MODELS_DIR   where trained model artifacts are stored
    MEDVISION_LOG_LEVEL    python logging level (default WARNING)
"""

import json
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Package identity
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
_INFO_FILE = BASE_DIR / "medvision_info.json"


def _load_info() -> dict:
    """Load medvision_info.json; fall back to sane defaults if unreadable."""
    try:
        with open(_INFO_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return {
            "name": "MedVision-AI",
            "full_name": "MedVision-AI Smart Health Analysis and Recommendation System",
            "version": "1.0.0",
            "disclaimer": (
                "Educational software demo. Not a medical device. "
                "Always consult a qualified healthcare professional."
            ),
        }


INFO = _load_info()

APP_NAME: str = INFO.get("name", "MedVision-AI")
APP_FULL_NAME: str = INFO.get(
    "full_name", "MedVision-AI Smart Health Analysis and Recommendation System"
)
VERSION: str = INFO.get("version", "1.0.0")
DISCLAIMER: str = INFO.get("disclaimer", "")

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("MEDVISION_DATA_DIR", BASE_DIR / "data"))
USERS_FILE = DATA_DIR / "users.json"
RECORDS_FILE = DATA_DIR / "health_records.json"

MODELS_DIR = Path(os.environ.get("MEDVISION_MODELS_DIR", BASE_DIR / "saved_models"))
MODEL_FILE = MODELS_DIR / "risk_model.pkl"

LOG_LEVEL = os.environ.get("MEDVISION_LOG_LEVEL", "WARNING").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a package-namespaced logger respecting MEDVISION_LOG_LEVEL."""
    logger = logging.getLogger(f"medvision.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
    return logger


# ---------------------------------------------------------------------------
# Clinical reference ranges
# ---------------------------------------------------------------------------

# -- WHO BMI classification (kg/m^2) ---------------------------------------
#    Lower-bound semantics: a category applies when bmi >= its lower bound.
BMI_CATEGORIES = [
    (0.0, "Underweight"),
    (18.5, "Normal weight"),
    (25.0, "Overweight"),
    (30.0, "Obesity Class I"),
    (35.0, "Obesity Class II"),
    (40.0, "Obesity Class III"),
]

# -- AHA/ACC 2017 adult blood-pressure staging (mmHg) -----------------------
#    (upper_bound_systolic, upper_bound_diastolic, label);
#    evaluated in order, first match wins.
BP_STAGES = [
    (120, 80, "Normal"),
    (130, 80, "Elevated"),
    (140, 90, "Stage 1 Hypertension"),
    (180, 120, "Stage 2 Hypertension"),
    (float("inf"), float("inf"), "Hypertensive Crisis"),
]
BP_CRISIS_SYSTOLIC = 180
BP_CRISIS_DIASTOLIC = 120

# -- ADA fasting-plasma-glucose interpretation (mg/dL) ----------------------
GLUCOSE_RANGES = [
    (70, "Low"),
    (100, "Normal"),
    (126, "Prediabetes"),
    (float("inf"), "Diabetes Range"),
]

# -- Lipid panel interpretation (mg/dL) ------------------------------------
CHOLESTEROL = {
    "total": [(200, "Desirable"), (240, "Borderline High"), (float("inf"), "High")],
    "ldl": [(100, "Optimal"), (130, "Near Optimal"), (160, "Borderline High"),
            (190, "High"), (float("inf"), "Very High")],
    # HDL: evaluated from BELOW (low HDL is bad). Sex-adjusted at runtime.
    "hdl_low_male": 40,     # < 40 mg/dL  -> Low (male)
    "hdl_low_female": 50,   # < 50 mg/dL  -> Low (female)
    "hdl_protective": 60,   # >= 60 mg/dL -> Protective
}

# -- Resting heart rate (bpm) ----------------------------------------------
HEART_RATE = {
    "bradycardia": 60,
    "normal_high": 100,
    "absolute_low": 40,     # below this -> concerning even for athletes
    "absolute_high": 150,   # above this -> severe tachycardia
}

# -- Sleep (hours/day, adults) ----------------------------------------------
SLEEP = {"min_recommended": 7.0, "max_recommended": 9.0}

# -- Physical activity -------------------------------------------------------
ACTIVITY = {
    "steps_ideal": 10000,
    "steps_sedentary": 5000,
    "exercise_ideal_per_week": 5,
    "exercise_min_per_week": 3,   # generally-accepted healthy minimum
}

# -- Hydration (liters/day) ---------------------------------------------------
HYDRATION = {"min_recommended": 2.0, "max_recommended": 3.5}

# -- Lifestyle limits ---------------------------------------------------------
LIFESTYLE = {
    "alcohol_limit_male": 14,      # drinks/week (US dietary guidance)
    "alcohol_limit_female": 7,
}

# ---------------------------------------------------------------------------
# Composite health score configuration
# ---------------------------------------------------------------------------
# Each category receives a subscore in [0, 100]. The composite score is the
# weighted mean of available subscores (weights of missing categories are
# renormalized, so partial records still produce a fair score).
#
# "ideal"  : value(s) inside this window score 100
# "hard"   : (low, high) anchors where the subscore reaches 0

SCORE_WEIGHTS = {
    "bmi": 0.12,
    "blood_pressure": 0.18,
    "glucose": 0.16,
    "cholesterol": 0.14,
    "heart_rate": 0.08,
    "sleep": 0.10,
    "activity": 0.12,
    "lifestyle": 0.10,
}

IDEAL_RANGES = {
    "bmi": {"ideal": (18.5, 24.9), "hard": (15.0, 40.0)},
    "systolic": {"ideal": (90.0, 119.0), "hard": (85.0, 180.0)},
    "diastolic": {"ideal": (60.0, 79.0), "hard": (55.0, 120.0)},
    "glucose": {"ideal": (70.0, 99.0), "hard": (50.0, 250.0)},
    "total_cholesterol": {"ideal": (120.0, 199.0), "hard": (80.0, 320.0)},
    "ldl": {"ideal": (40.0, 99.0), "hard": (30.0, 220.0)},
    "hdl": {"ideal": (60.0, 110.0), "hard": (30.0, 130.0)},
    "heart_rate": {"ideal": (60.0, 100.0), "hard": (42.0, 150.0)},
    "sleep": {"ideal": (7.0, 9.0), "hard": (3.0, 13.0)},
    "steps": {"ideal": (10000.0, 25000.0), "hard": (1000.0, 30000.0)},
    "exercise": {"ideal": (5.0, 14.0), "hard": (0.0, 21.0)},
    "water": {"ideal": (2.0, 3.5), "hard": (0.5, 6.0)},
}

# Composite status labels (threshold on the 0-100 scale, evaluated desc).
STATUS_LABELS = [
    (90, "Excellent"),
    (75, "Good"),
    (60, "Fair"),
    (40, "Poor"),
    (0, "Critical"),
]

# ---------------------------------------------------------------------------
# Rule-based risk estimator (points model)
# ---------------------------------------------------------------------------
# Cardiometabolic risk points; total points are squashed into a probability
# via a logistic curve configured by RISK_SLOPE / RISK_CENTER below.

RISK_POINTS = {
    "bp_stage2_or_crisis": 2.0,
    "bp_stage1": 1.0,
    "bp_elevated": 0.4,
    "glucose_diabetes_range": 2.0,
    "glucose_prediabetes": 1.0,
    "total_cholesterol_high": 1.0,
    "ldl_very_high": 1.0,
    "ldl_high": 0.6,
    "hdl_low": 0.8,
    "obese": 1.5,
    "overweight": 0.6,
    "current_smoker": 1.2,
    "sedentary": 0.6,           # < 5000 steps AND < 2 exercise sessions/week
    "sleep_deprived": 0.4,      # < 6 h/night
    "age_male_over_45": 1.0,
    "age_female_over_55": 1.0,
    "family_history_each": 0.6, # per condition (hypertension/diabetes/heart)
}

RISK_SLOPE = 0.35            # logistic steepness per point
RISK_CENTER = 4.0            # points at which probability == 0.5
RISK_PROB_FLOOR = 0.02
RISK_PROB_CEIL = 0.97

RISK_LEVELS = [              # (upper bound exclusive, label)
    (0.25, "Low"),
    (0.50, "Moderate"),
    (0.75, "High"),
    (float("inf"), "Very High"),
]

# ---------------------------------------------------------------------------
# Recommendation engine configuration
# ---------------------------------------------------------------------------

PRIORITY_ORDER = ["urgent", "high", "moderate", "low", "positive"]

RECOMMENDATION_CAP = 12      # max recommendations returned per analysis

__all__ = [
    "INFO", "APP_NAME", "APP_FULL_NAME", "VERSION", "DISCLAIMER",
    "BASE_DIR", "DATA_DIR", "USERS_FILE", "RECORDS_FILE",
    "MODELS_DIR", "MODEL_FILE", "LOG_LEVEL", "get_logger",
    "BMI_CATEGORIES", "BP_STAGES", "BP_CRISIS_SYSTOLIC", "BP_CRISIS_DIASTOLIC",
    "GLUCOSE_RANGES", "CHOLESTEROL", "HEART_RATE", "SLEEP", "ACTIVITY",
    "HYDRATION", "LIFESTYLE",
    "SCORE_WEIGHTS", "IDEAL_RANGES", "STATUS_LABELS",
    "RISK_POINTS", "RISK_SLOPE", "RISK_CENTER", "RISK_PROB_FLOOR",
    "RISK_PROB_CEIL", "RISK_LEVELS",
    "PRIORITY_ORDER", "RECOMMENDATION_CAP",
]
