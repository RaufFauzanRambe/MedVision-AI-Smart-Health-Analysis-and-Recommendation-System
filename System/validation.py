"""
MedVision-AI :: validation.py
=============================
Input sanitization and clinical-plausibility validation for user profiles and
daily health records.

Design goals:
    * Fail loudly on impossible values (e.g. systolic 900 mmHg) via
      :class:`ValidationError`, aggregating *all* problems in one pass.
    * Silently coerce benign variants ("75.0" -> 75.0, " MALE " -> "male").
    * Emit soft *warnings* for values that are physiologically possible but
      suspicious (e.g. LDL + HDL exceeding total cholesterol).

All functions return a cleaned copy of the input; the originals are never
mutated, so callers can safely persist the returned dict.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config

__all__ = ["ValidationError", "validate_user", "validate_record"]

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_DATE_FORMAT = "%Y-%m-%d"

_VALID_SEXES = ("male", "female", "other")


class ValidationError(ValueError):
    """Raised when input data fails clinical or structural validation.

    Attributes:
        errors: list of human-readable problem descriptions (may span fields).
        warnings: list of non-fatal observations for the same payload.
    """

    def __init__(self, errors: List[str], warnings: Optional[List[str]] = None):
        self.errors = list(errors)
        self.warnings = list(warnings or [])
        detail = "; ".join(self.errors) if self.errors else "invalid input"
        super().__init__(f"Validation failed ({len(self.errors)} error(s)): {detail}")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"Validation failed with {len(self.errors)} error(s):"]
        lines += [f"  - {e}" for e in self.errors]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level coercion helpers
# ---------------------------------------------------------------------------

def _coerce_float(field: str, value: Any, errors: List[str]) -> Optional[float]:
    """Convert value to float if at all possible; record an error otherwise."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"'{field}' must be a number, got {value!r}")
        return None


def _coerce_int(field: str, value: Any, errors: List[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        errors.append(f"'{field}' must be an integer, got {value!r}")
        return None
    if not as_float.is_integer():
        errors.append(f"'{field}' must be a whole number, got {value!r}")
        return None
    return int(as_float)


def _coerce_bool(field: str, value: Any, errors: List[str]) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "y", "1"):
            return True
        if low in ("false", "no", "n", "0"):
            return False
    errors.append(f"'{field}' must be a boolean, got {value!r}")
    return None


def _check_range(field: str, value: Optional[float], low: float, high: float,
                 errors: List[str]) -> Optional[float]:
    if value is None:
        return None
    if not (low <= value <= high):
        errors.append(
            f"'{field}' = {value:g} is outside the plausible range "
            f"[{low:g}, {high:g}]"
        )
        return None
    return value


# ---------------------------------------------------------------------------
# User-profile validation
# ---------------------------------------------------------------------------

def validate_user(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and normalize a user profile dict.

    Required keys: ``user_id``, ``name``, ``age``, ``sex``.
    Optional keys: ``height_cm``, ``email``, ``family_history`` (dict with
    optional boolean keys ``hypertension``, ``diabetes``, ``heart_disease``).

    Returns:
        (clean_user_dict, warnings)

    Raises:
        ValidationError: if any field is structurally invalid.
    """
    if not isinstance(data, dict):
        raise ValidationError(["user profile must be a dict"])

    errors: List[str] = []
    warnings: List[str] = []
    clean: Dict[str, Any] = {}

    # -- user_id -----------------------------------------------------------
    user_id = data.get("user_id")
    if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id.strip()):
        errors.append(
            "'user_id' must be 1-64 chars of letters/digits/'_'/'-', "
            f"got {user_id!r}"
        )
    else:
        clean["user_id"] = user_id.strip()

    # -- name ---------------------------------------------------------------
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("'name' must be a non-empty string")
    else:
        clean["name"] = name.strip()[:120]

    # -- age ------------------------------------------------------------------
    age = _coerce_int("age", data.get("age"), errors)
    if age is not None:
        if 1 <= age <= 120:
            clean["age"] = age
        else:
            errors.append(f"'age' = {age} is outside the plausible range [1, 120]")

    # -- sex ---------------------------------------------------------------
    sex = data.get("sex")
    if isinstance(sex, str):
        norm = sex.strip().lower()
        if norm in _VALID_SEXES:
            clean["sex"] = norm
        else:
            errors.append(
                f"'sex' must be one of {_VALID_SEXES}, got {sex!r}"
            )
    else:
        errors.append(f"'sex' must be a string, one of {_VALID_SEXES}")

    # -- optional height ------------------------------------------------------
    height = _coerce_float("height_cm", data.get("height_cm"), errors)
    if height is not None:
        clean["height_cm"] = _check_range(
            "height_cm", height, 50, 250, errors
        )

    # -- optional email (very light check; only shape, not deliverability) ----
    email = data.get("email")
    if email:
        if isinstance(email, str) and "@" in email and 3 <= len(email) <= 254:
            clean["email"] = email.strip()
        else:
            errors.append(f"'email' does not look like an email address: {email!r}")

    # -- optional family history ------------------------------------------------
    fh = data.get("family_history")
    if fh is not None:
        if not isinstance(fh, dict):
            errors.append("'family_history' must be a dict of booleans")
        else:
            clean_fh: Dict[str, bool] = {}
            for key in ("hypertension", "diabetes", "heart_disease"):
                if key in fh and fh[key] is not None:
                    val = _coerce_bool(f"family_history.{key}", fh[key], errors)
                    if val is not None:
                        clean_fh[key] = val
            clean["family_history"] = clean_fh

    if errors:
        raise ValidationError(errors, warnings)
    return clean, warnings


# ---------------------------------------------------------------------------
# Health-record validation
# ---------------------------------------------------------------------------

# field -> (low, high, coercion, required)
_FIELD_SPECS: Dict[str, Tuple[float, float, str, bool]] = {
    "height_cm":                  (50, 250, "float", False),  # may fall back to profile height
    "weight_kg":                  (2, 500, "float", True),
    "systolic_bp":                (60, 260, "float", False),
    "diastolic_bp":               (40, 160, "float", False),
    "heart_rate":                 (30, 220, "float", False),
    "fasting_glucose":            (20, 600, "float", False),
    "total_cholesterol":          (50, 500, "float", False),
    "hdl_cholesterol":            (10, 120, "float", False),
    "ldl_cholesterol":            (20, 350, "float", False),
    "sleep_hours":                (0, 24, "float", False),
    "steps":                      (0, 200000, "int", False),
    "water_intake_l":             (0, 15, "float", False),
    "alcohol_drinks_per_week":    (0, 100, "int", False),
    "exercise_sessions_per_week": (0, 30, "int", False),
}


def _normalize_date(raw: Any, errors: List[str]) -> str:
    """Accept ISO 'YYYY-MM-DD' (or datetime obj) and return the ISO string."""
    if raw is None:
        return date.today().strftime(_DATE_FORMAT)
    if isinstance(raw, (date, datetime)):
        return raw.strftime(_DATE_FORMAT)
    if isinstance(raw, str):
        try:
            return datetime.strptime(raw.strip(), _DATE_FORMAT).strftime(_DATE_FORMAT)
        except ValueError:
            errors.append(
                f"'date' must use the format YYYY-MM-DD, got {raw!r}"
            )
            return ""
    errors.append(f"'date' must be a YYYY-MM-DD string, got {raw!r}")
    return ""


def validate_record(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and normalize one daily health record.

    Required keys: ``user_id`` + ``height_cm`` + ``weight_kg``.
    ``date`` defaults to today when omitted. All other vital/lab/lifestyle
    fields are optional but range-checked when present (see ``_FIELD_SPECS``).

    Returns:
        (clean_record_dict, warnings)

    Raises:
        ValidationError: if any field is structurally or clinically invalid.
    """
    if not isinstance(data, dict):
        raise ValidationError(["health record must be a dict"])

    errors: List[str] = []
    warnings: List[str] = []
    clean: Dict[str, Any] = {}

    # -- identity & date ------------------------------------------------------
    user_id = data.get("user_id")
    if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id.strip()):
        errors.append(f"'user_id' must be a valid user id string, got {user_id!r}")
    else:
        clean["user_id"] = user_id.strip()

    iso_date = _normalize_date(data.get("date"), errors)
    if iso_date:
        clean["date"] = iso_date

    # -- per-field range checks -------------------------------------------------
    for field, (low, high, kind, required) in _FIELD_SPECS.items():
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                errors.append(f"required field '{field}' is missing")
            continue
        if kind == "int":
            value = _coerce_int(field, value, errors)
        else:
            value = _coerce_float(field, value, errors)
        value = _check_range(field, value, low, high, errors)
        if value is not None:
            clean[field] = value

    # -- cross-field: diastolic must be below systolic ----------------------------
    sbp, dbp = clean.get("systolic_bp"), clean.get("diastolic_bp")
    if sbp is not None and dbp is not None and dbp >= sbp:
        errors.append(
            f"'diastolic_bp' ({dbp:g}) must be lower than 'systolic_bp' ({sbp:g})"
        )

    # -- cross-field: lipid panel internal consistency (soft warning) -----------
    tc, hdl, ldl = (
        clean.get("total_cholesterol"),
        clean.get("hdl_cholesterol"),
        clean.get("ldl_cholesterol"),
    )
    if None not in (tc, hdl, ldl) and hdl + ldl > tc + 1e-6:
        warnings.append(
            "hdl_cholesterol + ldl_cholesterol exceeds total_cholesterol; "
            "the lipid panel looks internally inconsistent"
        )

    # -- lifestyle booleans ----------------------------------------------------
    smoking = data.get("smoking")
    if smoking is not None:
        val = _coerce_bool("smoking", smoking, errors)
        if val is not None:
            clean["smoking"] = val

    if errors:
        raise ValidationError(errors, warnings)
    return clean, warnings
