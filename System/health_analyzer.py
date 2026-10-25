"""
MedVision-AI :: health_analyzer.py
==================================
The rule-based clinical analysis engine.

For each daily health record the analyzer produces:

    1. Categorical interpretations
       BMI class (WHO), blood-pressure stage (AHA/ACC 2017), fasting-glucose
       band (ADA), lipid-panel status, resting heart-rate status, sleep,
       activity and hydration assessments.

    2. A weighted composite health score (0-100)
       Eight category subscores (bmi, blood_pressure, glucose, cholesterol,
       heart_rate, sleep, activity, lifestyle) are computed from piecewise
       linear scoring windows and combined with ``config.SCORE_WEIGHTS``.
       Categories missing from the record are skipped and the remaining
       weights are renormalized, so partial records still score fairly.

    3. Risk flags, hybrid risk assessment and longitudinal trends.

The class has no I/O side effects and is safe to reuse across threads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .model_manager import ModelManager

__all__ = ["HealthAnalyzer", "generate_text_report"]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_metric(value: float, ideal: Tuple[float, float],
                 hard: Tuple[float, float]) -> float:
    """Piecewise-linear score in [0, 100] for one metric.

    Inside ``ideal`` -> 100. Outside, the score decays linearly toward the
    ``hard`` anchors, reaching 0 at (or beyond) them. Works for both
    too-low and too-high deviations, taking the better of the two sides.
    """
    ideal_lo, ideal_hi = ideal
    hard_lo, hard_hi = hard
    if ideal_lo <= value <= ideal_hi:
        return 100.0
    if value < ideal_lo:
        span = max(ideal_lo - hard_lo, 1e-9)
        side = _clamp((value - hard_lo) / span) * 100.0
    else:
        span = max(hard_hi - ideal_hi, 1e-9)
        side = _clamp((hard_hi - value) / span) * 100.0
    return round(side, 1)


def _classify_bmi(bmi: float) -> str:
    """WHO band via lower-bound lookup (last matching bound wins)."""
    label = config.BMI_CATEGORIES[0][1]
    for lower, name in config.BMI_CATEGORIES:
        if bmi >= lower:
            label = name
    return label


def _classify_bp(sbp: float, dbp: float) -> str:
    if sbp >= config.BP_CRISIS_SYSTOLIC or dbp >= config.BP_CRISIS_DIASTOLIC:
        return "Hypertensive Crisis"
    for upper_s, upper_d, label in config.BP_STAGES:
        if sbp < upper_s and dbp < upper_d:
            return label
    return "Stage 2 Hypertension"  # unreachable safeguard


def _classify_glucose(mg_dl: float) -> str:
    for upper, label in config.GLUCOSE_RANGES:
        if mg_dl < upper:
            return label
    return "Diabetes Range"


def _classify_total_chol(mg_dl: float) -> str:
    for upper, label in config.CHOLESTEROL["total"]:
        if mg_dl < upper:
            return label
    return "High"


def _classify_ldl(mg_dl: float) -> str:
    for upper, label in config.CHOLESTEROL["ldl"]:
        if mg_dl < upper:
            return label
    return "Very High"


def _classify_hdl(mg_dl: float, sex: Optional[str]) -> str:
    if mg_dl >= config.CHOLESTEROL["hdl_protective"]:
        return "Protective"
    low_cut = config.CHOLESTEROL["hdl_low_female"] if sex == "female" \
        else config.CHOLESTEROL["hdl_low_male"]
    if mg_dl < low_cut:
        return "Low"
    return "Average"


def _classify_heart_rate(bpm: float) -> str:
    hr = config.HEART_RATE
    if bpm < hr["absolute_low"]:
        return "Severe Bradycardia"
    if bpm < hr["bradycardia"]:
        return "Bradycardia (may be normal for athletes)"
    if bpm <= hr["normal_high"]:
        return "Normal"
    if bpm <= hr["absolute_high"]:
        return "Tachycardia"
    return "Severe Tachycardia"


class HealthAnalyzer:
    """Analyzes one health record into interpretations, score, risk, trends.

    Args:
        model_manager: optional shared :class:`ModelManager`. When supplied,
            risk prediction runs in *hybrid* mode (sklearn model when trained,
            rule-based estimator otherwise). When omitted, an internal
            fallback manager is created so results are always producible.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, record: Dict[str, Any],
                user: Optional[Dict[str, Any]] = None,
                history: Optional[List[Dict[str, Any]]] = None
                ) -> Dict[str, Any]:
        """Produce the full analysis payload for one record.

        Args:
            record: a validated health-record dict.
            user: optional profile dict (enables sex/age personalization and
                family-history risk factors).
            history: optional full record list (oldest -> newest) for trend
                analysis; ``record`` itself need not be the last element.

        Returns:
            Dict with keys: ``bmi``, ``blood_pressure``, ``glucose``,
            ``cholesterol``, ``heart_rate``, ``sleep``, ``activity``,
            ``hydration``, ``subscores``, ``health_score``, ``health_status``,
            ``risk_flags``, ``risk_assessment``, ``trends``, ``notes``.
        """
        user = user or {}
        sex = user.get("sex")
        notes: List[str] = []

        # ---- BMI ---------------------------------------------------------
        bmi_value = self._compute_bmi(record, user, notes)
        bmi_block: Dict[str, Any] = {"value": bmi_value}
        if bmi_value is not None:
            bmi_block["category"] = _classify_bmi(bmi_value)

        # ---- Blood pressure ----------------------------------------------
        sbp, dbp = record.get("systolic_bp"), record.get("diastolic_bp")
        bp_block: Dict[str, Any] = {"systolic": sbp, "diastolic": dbp}
        if sbp is not None and dbp is not None:
            stage = _classify_bp(sbp, dbp)
            bp_block["stage"] = stage
            if stage == "Hypertensive Crisis":
                notes.append(
                    "Blood pressure is at hypertensive-crisis level "
                    "(>= 180/120). Seek immediate medical care."
                )

        # ---- Glucose -------------------------------------------------------
        glucose = record.get("fasting_glucose")
        glucose_block: Dict[str, Any] = {"value": glucose}
        if glucose is not None:
            glucose_block["status"] = _classify_glucose(glucose)

        # ---- Lipids ---------------------------------------------------------
        tc = record.get("total_cholesterol")
        hdl = record.get("hdl_cholesterol")
        ldl = record.get("ldl_cholesterol")
        lipid_block: Dict[str, Any] = {
            "total": tc, "hdl": hdl, "ldl": ldl, "status": None, "details": []
        }
        if any(v is not None for v in (tc, hdl, ldl)):
            details: List[str] = []
            if tc is not None:
                details.append(f"Total: {_classify_total_chol(tc)}")
            if ldl is not None:
                details.append(f"LDL: {_classify_ldl(ldl)}")
            if hdl is not None:
                details.append(f"HDL: {_classify_hdl(hdl, sex)}")
            lipid_block["details"] = details
            lipid_block["status"] = self._overall_lipid_status(tc, hdl, ldl, sex)

        # ---- Heart rate / sleep / activity / hydration ----------------------
        hr = record.get("heart_rate")
        hr_block: Dict[str, Any] = {"value": hr}
        if hr is not None:
            hr_block["status"] = _classify_heart_rate(hr)

        sleep_h = record.get("sleep_hours")
        sleep_block: Dict[str, Any] = {"value": sleep_h}
        if sleep_h is not None:
            if sleep_h < config.SLEEP["min_recommended"]:
                sleep_block["status"] = "Below recommended range"
            elif sleep_h > config.SLEEP["max_recommended"]:
                sleep_block["status"] = "Above recommended range"
            else:
                sleep_block["status"] = "Healthy range"

        steps = record.get("steps")
        ex_sessions = record.get("exercise_sessions_per_week")
        activity_block: Dict[str, Any] = {"steps": steps,
                                          "exercise_sessions_per_week": ex_sessions}
        if steps is not None:
            if steps >= config.ACTIVITY["steps_ideal"]:
                activity_block["status"] = "Highly active"
            elif steps >= config.ACTIVITY["steps_sedentary"]:
                activity_block["status"] = "Moderately active"
            else:
                activity_block["status"] = "Sedentary"

        water = record.get("water_intake_l")
        hydration_block: Dict[str, Any] = {"value": water}
        if water is not None:
            if water < config.HYDRATION["min_recommended"]:
                hydration_block["status"] = "Low"
            elif water > config.HYDRATION["max_recommended"]:
                hydration_block["status"] = "High"
            else:
                hydration_block["status"] = "Healthy range"

        # ---- Composite score -------------------------------------------------
        subscores = self._subscores(record, user, bmi_value)
        score, status = self._composite_score(subscores)

        # ---- Risk flags / assessment / trends ---------------------------------
        risk_flags = self._risk_flags(bmi_block, bp_block, glucose_block,
                                      lipid_block, hr_block, sleep_block,
                                      activity_block)
        risk = self.model_manager.predict_risk(record, user)

        trends = self._trends(record, history or [])

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "record": dict(record),
            "user": {
                "user_id": user.get("user_id"),
                "name": user.get("name"),
                "age": user.get("age"),
                "sex": sex,
            },
            "bmi": bmi_block,
            "blood_pressure": bp_block,
            "glucose": glucose_block,
            "cholesterol": lipid_block,
            "heart_rate": hr_block,
            "sleep": sleep_block,
            "activity": activity_block,
            "hydration": hydration_block,
            "subscores": subscores,
            "health_score": score,
            "health_status": status,
            "risk_flags": risk_flags,
            "risk_assessment": risk,
            "trends": trends,
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bmi(record: Dict[str, Any], user: Dict[str, Any],
                     notes: List[str]) -> Optional[float]:
        height = record.get("height_cm") or user.get("height_cm")
        weight = record.get("weight_kg")
        if height and weight:
            try:
                meters = float(height) / 100.0
                return round(float(weight) / (meters * meters), 1)
            except (TypeError, ValueError, ZeroDivisionError):
                notes.append("BMI could not be computed from the given "
                             "height/weight values.")
        elif not weight:
            notes.append("Weight missing; BMI-based scoring skipped.")
        return None

    @staticmethod
    def _overall_lipid_status(tc, hdl, ldl, sex) -> str:
        """Blend the lipid panel into one coarse status label."""
        statuses: List[str] = []
        if tc is not None:
            statuses.append(_classify_total_chol(tc))
        if ldl is not None:
            statuses.append(_classify_ldl(ldl))
        if hdl is not None:
            statuses.append(_classify_hdl(hdl, sex))
        if not statuses:
            return "Unknown"
        bad = {"High", "Very High", "Low"}
        mid = {"Borderline High", "Near Optimal", "Average"}
        if any(s in bad for s in statuses):
            return "Needs attention"
        if any(s in mid for s in statuses):
            return "Borderline"
        return "Healthy"

    def _subscores(self, record: Dict[str, Any], user: Dict[str, Any],
                   bmi_value: Optional[float]) -> Dict[str, float]:
        """Compute all available category subscores in [0, 100]."""
        ir = config.IDEAL_RANGES
        out: Dict[str, float] = {}
        sex = user.get("sex")

        if bmi_value is not None:
            out["bmi"] = score_metric(bmi_value, ir["bmi"]["ideal"],
                                      ir["bmi"]["hard"])

        sbp, dbp = record.get("systolic_bp"), record.get("diastolic_bp")
        if sbp is not None and dbp is not None:
            out["blood_pressure"] = round(min(
                score_metric(sbp, ir["systolic"]["ideal"], ir["systolic"]["hard"]),
                score_metric(dbp, ir["diastolic"]["ideal"], ir["diastolic"]["hard"]),
            ), 1)

        glucose = record.get("fasting_glucose")
        if glucose is not None:
            out["glucose"] = score_metric(glucose, ir["glucose"]["ideal"],
                                          ir["glucose"]["hard"])

        tc = record.get("total_cholesterol")
        hdl = record.get("hdl_cholesterol")
        ldl = record.get("ldl_cholesterol")
        lipid_parts: List[float] = []
        if tc is not None:
            lipid_parts.append(score_metric(tc, ir["total_cholesterol"]["ideal"],
                                            ir["total_cholesterol"]["hard"]))
        if ldl is not None:
            lipid_parts.append(score_metric(ldl, ir["ldl"]["ideal"],
                                            ir["ldl"]["hard"]))
        if hdl is not None:
            # low HDL is bad -> score rises with HDL up to the ideal window
            lipid_parts.append(score_metric(hdl, ir["hdl"]["ideal"],
                                            ir["hdl"]["hard"]))
        if lipid_parts:
            out["cholesterol"] = round(min(lipid_parts), 1)

        hr = record.get("heart_rate")
        if hr is not None:
            out["heart_rate"] = score_metric(hr, ir["heart_rate"]["ideal"],
                                             ir["heart_rate"]["hard"])

        sleep_h = record.get("sleep_hours")
        if sleep_h is not None:
            out["sleep"] = score_metric(sleep_h, ir["sleep"]["ideal"],
                                        ir["sleep"]["hard"])

        steps = record.get("steps")
        ex = record.get("exercise_sessions_per_week")
        act_parts: List[float] = []
        if steps is not None:
            act_parts.append(score_metric(steps, ir["steps"]["ideal"],
                                          ir["steps"]["hard"]))
        if ex is not None:
            act_parts.append(score_metric(ex, ir["exercise"]["ideal"],
                                          ir["exercise"]["hard"]))
        if act_parts:
            out["activity"] = round(sum(act_parts) / len(act_parts), 1)

        life_parts: List[float] = []
        water = record.get("water_intake_l")
        if water is not None:
            life_parts.append(score_metric(water, ir["water"]["ideal"],
                                           ir["water"]["hard"]))
        if record.get("smoking"):
            life_parts.append(20.0)      # heavy penalty for current smoking
        elif record.get("smoking") is not None:
            life_parts.append(100.0)
        alcohol = record.get("alcohol_drinks_per_week")
        if alcohol is not None:
            limit = config.LIFESTYLE["alcohol_limit_female"] if sex == "female" \
                else config.LIFESTYLE["alcohol_limit_male"]
            life_parts.append(100.0 if alcohol <= limit else
                              max(0.0, 100.0 - 8.0 * (alcohol - limit)))
        if life_parts:
            out["lifestyle"] = round(sum(life_parts) / len(life_parts), 1)

        return out

    @staticmethod
    def _composite_score(subscores: Dict[str, float]
                         ) -> Tuple[float, str]:
        """Weighted mean of available subscores with renormalized weights."""
        if not subscores:
            return 0.0, "Insufficient data"
        total_weight = sum(
            config.SCORE_WEIGHTS[cat] for cat in subscores
            if cat in config.SCORE_WEIGHTS
        )
        if total_weight <= 0:
            weighted = sum(subscores.values()) / len(subscores)
        else:
            weighted = sum(
                config.SCORE_WEIGHTS.get(cat, 0.0) * val
                for cat, val in subscores.items()
            ) / total_weight
        score = round(_clamp(weighted), 1)
        status = next(label for floor, label in config.STATUS_LABELS
                      if score >= floor)
        return score, status

    @staticmethod
    def _risk_flags(bmi_block, bp_block, glucose_block, lipid_block,
                    hr_block, sleep_block, activity_block) -> List[Dict[str, str]]:
        flags: List[Dict[str, str]] = []

        if bmi_block.get("category") in ("Obesity Class I",
                                         "Obesity Class II",
                                         "Obesity Class III"):
            flags.append({"metric": "bmi", "severity": "warning",
                          "message": f"BMI {bmi_block['value']} "
                                     f"({bmi_block['category']})"})
        if bp_block.get("stage") in ("Stage 1 Hypertension",
                                     "Stage 2 Hypertension",
                                     "Hypertensive Crisis"):
            flags.append({"metric": "blood_pressure",
                          "severity": "urgent" if bp_block["stage"] in
                          ("Stage 2 Hypertension", "Hypertensive Crisis")
                          else "warning",
                          "message": f"Blood pressure {bp_block['systolic']:g}/"
                                     f"{bp_block['diastolic']:g} "
                                     f"({bp_block['stage']})"})
        if glucose_block.get("status") in ("Prediabetes", "Diabetes Range",
                                           "Low"):
            flags.append({"metric": "glucose",
                          "severity": "urgent" if glucose_block["status"] ==
                          "Diabetes Range" else "warning",
                          "message": f"Fasting glucose {glucose_block['value']:g} "
                                     f"mg/dL ({glucose_block['status']})"})
        if lipid_block.get("status") == "Needs attention":
            flags.append({"metric": "cholesterol", "severity": "warning",
                          "message": "Lipid panel "
                                     + "; ".join(lipid_block.get("details", []))})
        hr_status = hr_block.get("status", "")
        if hr_status.startswith(("Bradycardia", "Tachycardia", "Severe")):
            flags.append({"metric": "heart_rate",
                          "severity": "urgent" if hr_status.startswith("Severe")
                          else "warning",
                          "message": f"Resting heart rate {hr_block['value']:g} "
                                     f"bpm ({hr_status})"})
        if sleep_block.get("status") == "Below recommended range":
            flags.append({"metric": "sleep", "severity": "info",
                          "message": f"Sleep {sleep_block['value']:g} h "
                                     "(below the 7-9 h adult recommendation)"})
        if activity_block.get("status") == "Sedentary":
            flags.append({"metric": "activity", "severity": "info",
                          "message": f"Only {activity_block['steps']:g} steps - "
                                     "sedentary activity level"})

        severity_rank = {"urgent": 0, "warning": 1, "info": 2}
        flags.sort(key=lambda f: severity_rank.get(f["severity"], 3))
        return flags

    @staticmethod
    def _trends(record: Dict[str, Any],
                history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare *record* against the previous record in *history*."""
        dated = sorted(
            [r for r in history if r.get("date")],
            key=lambda r: r["date"],
        )
        earlier = None
        this_date = record.get("date")
        for r in dated:
            if r.get("date") == this_date:
                break
            earlier = r  # keep the latest record strictly before this one
        if earlier is None:
            return [{"status": "insufficient_history",
                     "message": "No earlier record available; trend analysis "
                                "needs at least two records."}]

        watched = {
            "weight_kg": "lower_is_better", "systolic_bp": "lower_is_better",
            "diastolic_bp": "lower_is_better", "fasting_glucose": "lower_is_better",
            "total_cholesterol": "lower_is_better", "ldl_cholesterol": "lower_is_better",
            "hdl_cholesterol": "higher_is_better", "heart_rate": "lower_is_better",
            "sleep_hours": "closer_to_ideal", "steps": "higher_is_better",
        }
        trends: List[Dict[str, Any]] = []
        for field, direction in watched.items():
            now_val, prev_val = record.get(field), earlier.get(field)
            if now_val is None or prev_val is None:
                continue
            delta = round(float(now_val) - float(prev_val), 2)
            if abs(delta) < 1e-9:
                trend = "stable"
            elif direction == "higher_is_better":
                trend = "improving" if delta > 0 else "declining"
            elif direction == "lower_is_better":
                trend = "improving" if delta < 0 else "declining"
            else:  # closer_to_ideal for sleep (7-9 h)
                ideal_mid = 8.0
                trend = "improving" if abs(now_val - ideal_mid) < abs(prev_val - ideal_mid) \
                    else "declining"
            trends.append({
                "metric": field, "previous": prev_val, "current": now_val,
                "delta": delta, "trend": trend,
                "previous_date": earlier.get("date"),
            })
        return trends


# ---------------------------------------------------------------------------
# Presentation helper
# ---------------------------------------------------------------------------

def generate_text_report(analysis: Dict[str, Any],
                         disclaimer: Optional[str] = None) -> str:
    """Render an analysis payload as a fixed-width plain-text report."""
    width = 68
    user = analysis.get("user", {})
    record = analysis.get("record", {})
    lines: List[str] = []
    bar = "=" * width
    thin = "-" * width

    lines.append(bar)
    lines.append(f"{config.APP_NAME.upper()}  |  HEALTH ANALYSIS REPORT".center(width))
    lines.append(bar)
    lines.append(f"Patient : {user.get('name', 'Unknown')} "
                 f"({user.get('user_id', '?')})   "
                 f"Age {user.get('age', '?')}   Sex {user.get('sex', '?')}")
    lines.append(f"Date    : {record.get('date', 'n/a')}   "
                 f"Generated: {analysis.get('generated_at', '')}")
    lines.append(thin)

    lines.append("METRICS")
    bmi = analysis.get("bmi", {})
    if bmi.get("value") is not None:
        lines.append(f"  BMI              {bmi['value']:>6}   {bmi.get('category', '')}")
    bp = analysis.get("blood_pressure", {})
    if bp.get("systolic") is not None:
        lines.append(f"  Blood pressure   {bp['systolic']:g}/{bp['diastolic']:g}   "
                     f"{bp.get('stage', '')}")
    glc = analysis.get("glucose", {})
    if glc.get("value") is not None:
        lines.append(f"  Fasting glucose  {glc['value']:g} mg/dL   "
                     f"{glc.get('status', '')}")
    lip = analysis.get("cholesterol", {})
    if lip.get("status") and lip["status"] != "Unknown":
        lines.append(f"  Lipid panel      total {lip.get('total', '-')} / "
                     f"HDL {lip.get('hdl', '-')} / LDL {lip.get('ldl', '-')}   "
                     f"{lip['status']}")
    hr = analysis.get("heart_rate", {})
    if hr.get("value") is not None:
        lines.append(f"  Resting HR       {hr['value']:g} bpm   "
                     f"{hr.get('status', '')}")
    slp = analysis.get("sleep", {})
    if slp.get("value") is not None:
        lines.append(f"  Sleep            {slp['value']:g} h   "
                     f"{slp.get('status', '')}")
    act = analysis.get("activity", {})
    if act.get("steps") is not None:
        lines.append(f"  Steps            {act['steps']:g}   "
                     f"{act.get('status', '')}")
    hyd = analysis.get("hydration", {})
    if hyd.get("value") is not None:
        lines.append(f"  Water intake     {hyd['value']:g} L   "
                     f"{hyd.get('status', '')}")
    lines.append(thin)

    lines.append(f"HEALTH SCORE : {analysis.get('health_score', 0):g} / 100   "
                 f"({analysis.get('health_status', '')})")
    subs = analysis.get("subscores", {})
    if subs:
        pretty = " | ".join(f"{k} {v:g}" for k, v in sorted(subs.items()))
        lines.append(f"  Subscores: {pretty}")
    lines.append(thin)

    risk = analysis.get("risk_assessment", {})
    lines.append(f"RISK ASSESSMENT ({risk.get('method', 'n/a')})")
    lines.append(f"  10-y cardiometabolic risk proxy: "
                 f"{risk.get('probability', 0) * 100:.0f}%  "
                 f"({risk.get('level', '')})")
    for factor in risk.get("factors", []):
        lines.append(f"  - {factor}")
    lines.append(thin)

    trends = analysis.get("trends", [])
    if trends and trends[0].get("status") != "insufficient_history":
        lines.append("TRENDS vs previous record")
        for t in trends:
            arrow = {"improving": "improving", "declining": "declining",
                     "stable": "stable"}.get(t["trend"], t["trend"])
            lines.append(f"  {t['metric']:<22} {t['previous']} -> {t['current']}  "
                         f"(delta {t['delta']:+g}, {arrow})")
    else:
        lines.append("TRENDS: not enough history yet.")

    if analysis.get("notes"):
        lines.append(thin)
        lines.append("NOTES")
        lines += [f"  * {n}" for n in analysis["notes"]]

    lines.append(thin)
    lines.append("This report is NOT medical advice. Consult a qualified")
    lines.append("healthcare professional for diagnosis and treatment.")
    lines.append(bar)
    if disclaimer and disclaimer not in "\n".join(lines):
        lines.append(disclaimer)
    return "\n".join(lines)
