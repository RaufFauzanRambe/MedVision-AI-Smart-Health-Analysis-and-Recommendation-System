"""
MedVision-AI :: recommendation_engine.py
========================================
Turns an analysis payload into prioritized, personalized recommendations and
a simple 7-day wellness plan.

Output contract - every recommendation is a dict:
    {
      "category":  "medical_attention" | "nutrition" | "physical_activity"
                 | "sleep" | "hydration" | "lifestyle" | "monitoring",
      "priority":  "urgent" | "high" | "moderate" | "low" | "positive",
      "title":     short headline,
      "detail":    why this matters (1-3 sentences),
      "actions":   [concrete, checkable action items]
    }

The list is sorted urgent-first and capped at ``config.RECOMMENDATION_CAP``.
Positive reinforcement is deliberately included so healthy users do not get
an empty (or naggy) response. Every generated set ends with the safety note
"recommendations are educational, consult a professional" - enforced by
``get_disclaimer()`` which callers should always display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import config

__all__ = ["RecommendationEngine"]


class RecommendationEngine:
    """Rule-based generator of personalized health recommendations."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, analysis: Dict[str, Any],
                 user: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Produce the sorted, capped recommendation list for one analysis."""
        user = user or {}
        recs: List[Dict[str, Any]] = []
        sex = (user.get("sex") or "other").lower()

        # 1. hard clinical findings first
        recs += self._medical_attention(analysis)
        # 2. domain advice driven by the same findings
        recs += self._nutrition(analysis, sex)
        recs += self._physical_activity(analysis)
        recs += self._sleep(analysis)
        recs += self._hydration(analysis)
        recs += self._lifestyle(analysis, sex)
        recs += self._monitoring(analysis)
        # 3. positive reinforcement
        recs += self._positives(analysis)

        priority = {p: i for i, p in enumerate(config.PRIORITY_ORDER)}
        recs.sort(key=lambda r: priority.get(r["priority"], 99))
        return recs[: config.RECOMMENDATION_CAP]

    def build_weekly_plan(self, analysis: Dict[str, Any],
                          user: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """A gentle 7-day focus rotation personalized by activity level."""
        user = user or {}
        activity = (analysis.get("activity") or {}).get("status", "")
        bp_stage = (analysis.get("blood_pressure") or {}).get("stage", "Normal")
        high_intensity_ok = activity == "Highly active" and \
            bp_stage in ("Normal", "Elevated")

        base = [
            ("Monday", "Cardio day",
             f"{30 if high_intensity_ok else 15} min brisk walk or cycling"),
            ("Tuesday", "Strength day",
             "Bodyweight basics: squats, wall push-ups, planks (2 x 10)"),
            ("Wednesday", "Mobility day",
             "10-15 min stretching or yoga before bed"),
            ("Thursday", "Cardio day",
             f"{30 if high_intensity_ok else 20} min moderate walk"),
            ("Friday", "Strength day",
             "Light resistance band workout (20 min)"),
            ("Saturday", "Active fun",
             "A hobby that moves you: hiking, dancing, or sports"),
            ("Sunday", "Recovery day",
             "Easy stroll + prepare healthy meals for the week"),
        ]
        plan = [{"day": d, "focus": f, "activity": a} for d, f, a in base]

        sleep_short = (analysis.get("sleep") or {}).get("status") == \
            "Below recommended range"
        if sleep_short:
            plan[2]["activity"] += "; lights out 30 min earlier"
            plan[6]["activity"] += "; wind-down routine tonight"
        return plan

    @staticmethod
    def get_disclaimer() -> str:
        """The safety note that must accompany any rendered recommendation."""
        return (
            "These recommendations are generated for educational purposes and "
            "are not medical advice. Please consult a qualified healthcare "
            "professional before changing your diet, exercise routine, or "
            "medications."
        )

    # ------------------------------------------------------------------
    # Generators (each returns a list of recommendation dicts)
    # ------------------------------------------------------------------

    def _medical_attention(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        bp = analysis.get("blood_pressure", {})
        stage = bp.get("stage")

        if stage == "Hypertensive Crisis":
            recs.append({
                "category": "medical_attention", "priority": "urgent",
                "title": "Seek immediate medical care",
                "detail": "A blood pressure of 180/120 mmHg or higher is a "
                          "hypertensive crisis and can damage organs within "
                          "hours. Do not wait for a routine appointment.",
                "actions": [
                    "Contact emergency services or go to the nearest ER now",
                    "Do not take extra BP medication unless a doctor says so",
                    "Have someone stay with you until you are evaluated",
                ],
            })
        elif stage == "Stage 2 Hypertension":
            recs.append({
                "category": "medical_attention", "priority": "urgent",
                "title": "Book a physician visit within days",
                "detail": f"Blood pressure {bp['systolic']:g}/{bp['diastolic']:g} "
                          "is in the stage-2 hypertension range. Repeated "
                          "readings at this level usually warrant medication "
                          "evaluation and a workup for underlying causes.",
                "actions": [
                    "Schedule a doctor appointment this week",
                    "Measure BP twice daily and log the readings",
                    "Bring this report to the consultation",
                ],
            })

        glucose = analysis.get("glucose", {})
        if glucose.get("status") == "Diabetes Range":
            recs.append({
                "category": "medical_attention", "priority": "urgent",
                "title": "Fasting glucose is in the diabetes range",
                "detail": f"A fasting value of {glucose['value']:g} mg/dL meets "
                          "the ADA threshold for diabetes. Diagnosis requires "
                          "confirmation, so professional follow-up should "
                          "happen quickly.",
                "actions": [
                    "Request an appointment for confirmatory testing "
                    "(repeat fasting glucose, HbA1c)",
                    "Ask about diabetes education and nutrition referral",
                    "Watch for thirst, frequent urination, or blurred vision",
                ],
            })
        elif glucose.get("status") == "Prediabetes":
            recs.append({
                "category": "medical_attention", "priority": "high",
                "title": "Prediabetes range - act early",
                "detail": f"Fasting glucose {glucose['value']:g} mg/dL falls in "
                          "the prediabetes window. Lifestyle change at this "
                          "stage can delay or prevent progression to type-2 "
                          "diabetes.",
                "actions": [
                    "Ask your clinician about an HbA1c test",
                    "Cut sugary drinks entirely this month",
                    "Add one 10-minute walk after your largest meal",
                ],
            })

        lipids = analysis.get("cholesterol", {})
        if lipids.get("status") == "Needs attention":
            ldl = lipids.get("ldl")
            recs.append({
                "category": "medical_attention", "priority": "high",
                "title": "Discuss your lipid panel with a clinician",
                "detail": "One or more lipid fractions are outside the "
                          "desirable range "
                          f"({'; '.join(lipids.get('details', []))}). "
                          "Persistent elevations contribute to "
                          "cardiovascular risk over years.",
                "actions": [
                    "Share this report at your next check-up",
                    "Request a full lipid panel if values are older than 6 months"
                ]
                + (["Ask about treatment thresholds given your overall risk"]
                   if (ldl or 0) >= 160 else []),
            })

        hr = analysis.get("heart_rate", {})
        if str(hr.get("status", "")).startswith(("Severe", "Tachycardia")):
            recs.append({
                "category": "medical_attention", "priority": "high",
                "title": "Resting heart rate needs review",
                "detail": f"Resting HR {hr['value']:g} bpm ({hr['status']}) "
                          "is above the usual 60-100 range. Caffeine, stress, "
                          "fever, dehydration and thyroid issues are common "
                          "drivers.",
                "actions": [
                    "Re-measure at true rest on several days",
                    "Mention palpitations or dizziness to a clinician",
                ],
            })

        return recs

    def _nutrition(self, analysis: Dict[str, Any], sex: str) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        bmi_cat = (analysis.get("bmi") or {}).get("category")
        needs_bp_focus = (analysis.get("blood_pressure") or {}).get("stage") in (
            "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension")
        lipids = (analysis.get("cholesterol") or {}).get("status")

        if bmi_cat == "Underweight":
            recs.append({
                "category": "nutrition", "priority": "moderate",
                "title": "Gently increase energy intake",
                "detail": "A BMI below 18.5 suggests under-nutrition. Aim for "
                          "calorie-dense whole foods rather than empty "
                          "snacks to rebuild reserves safely.",
                "actions": [
                    "Add nuts, avocado, olive oil or full-fat yogurt daily",
                    "Eat 4-5 smaller meals instead of 2-3 large ones",
                    "Consider a dietitian consult if weight keeps dropping",
                ],
            })
        elif bmi_cat in ("Overweight", "Obesity Class I", "Obesity Class II",
                         "Obesity Class III"):
            recs.append({
                "category": "nutrition", "priority": "moderate",
                "title": "Sustainable weight-management plan",
                "detail": f"BMI places you in the '{bmi_cat}' range. A modest "
                          "0.5-1 kg per week loss improves blood pressure, "
                          "glucose and lipids simultaneously.",
                "actions": [
                    "Target a 300-500 kcal daily deficit, not crash dieting",
                    "Half of every plate: non-starchy vegetables",
                    "Limit fried food and sugary desserts to once a week",
                ],
            })

        if needs_bp_focus:
            recs.append({
                "category": "nutrition", "priority": "high",
                "title": "DASH-style eating for blood pressure",
                "detail": "The DASH pattern (rich in vegetables, fruits, "
                          "low-fat dairy, and potassium; low in sodium) can "
                          "lower systolic pressure by 5-11 mmHg in many "
                          "adults.",
                "actions": [
                    "Keep sodium under 1,500-2,300 mg/day; check labels",
                    "Swap processed snacks for bananas, beans, and greens",
                    "Season with herbs and spices instead of salt",
                ],
            })

        if lipids == "Needs attention" or lipids == "Borderline":
            recs.append({
                "category": "nutrition", "priority": "moderate",
                "title": "Cholesterol-friendly food swaps",
                "detail": "Replacing saturated fats with unsaturated sources "
                          "and adding soluble fiber measurably lowers LDL "
                          "cholesterol within weeks.",
                "actions": [
                    "Use olive or canola oil instead of butter",
                    "Add oats, beans, apples or barley daily (soluble fiber)",
                    "Fish twice a week; reduce red and processed meats",
                ],
            })

        glucose = (analysis.get("glucose") or {}).get("status")
        if glucose in ("Prediabetes", "Diabetes Range"):
            recs.append({
                "category": "nutrition", "priority": "high",
                "title": "Steady-carb, low-glycemic meals",
                "detail": "Distributing carbohydrates evenly and choosing "
                          "low-glycemic sources reduces glucose spikes and "
                          "insulin demand.",
                "actions": [
                    "Choose whole grains over white bread/rice/pasta",
                    "Pair carbs with protein or healthy fat",
                    "Replace juice and soda with water or unsweetened tea",
                ],
            })
        return recs

    def _physical_activity(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        act = analysis.get("activity", {})
        status = act.get("status")
        steps = act.get("steps")

        if status == "Sedentary" and steps is not None:
            recs.append({
                "category": "physical_activity", "priority": "high",
                "title": "Break out of the sedentary zone",
                "detail": f"At {steps:g} daily steps you are well below the "
                          "commonly used 7,000-10,000 step benchmarks. Even "
                          "small increases cut cardiometabolic risk.",
                "actions": [
                    "Add ~1,000 steps per week until you reach 8,000-10,000",
                    "Walk 5 minutes every hour you sit",
                    "Two 15-minute walks daily beat one impossible hour",
                ],
            })
        elif status == "Moderately active":
            recs.append({
                "category": "physical_activity", "priority": "low",
                "title": "Push activity a little further",
                "detail": "You are moving at a moderate level; nudging toward "
                          "10,000 steps adds meaningful cardio protection.",
                "actions": [
                    "One extra 15-20 minute walk on most days",
                    "Try a new activity class to keep it interesting",
                ],
            })

        ex = act.get("exercise_sessions_per_week")
        if ex is not None and ex < config.ACTIVITY["exercise_min_per_week"]:
            recs.append({
                "category": "physical_activity", "priority": "moderate",
                "title": "Reach 3+ structured sessions weekly",
                "detail": "Guidelines recommend at least 150 minutes of "
                          "moderate aerobic activity per week plus two "
                          "strength sessions. You are currently below that "
                          "floor.",
                "actions": [
                    "Schedule 3 calendar-blocked workouts (30 min each)",
                    "Start with brisk walking + bodyweight strength",
                    "Track sessions in this system weekly",
                ],
            })
        return recs

    def _sleep(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        sleep = analysis.get("sleep", {})
        if sleep.get("status") == "Below recommended range":
            recs.append({
                "category": "sleep", "priority": "moderate",
                "title": "Protect a 7-9 hour sleep window",
                "detail": f"Records show {sleep['value']:g} h of sleep. Short "
                          "sleep raises blood pressure, glucose and appetite "
                          "hormones - it quietly undoes diet and exercise "
                          "gains.",
                "actions": [
                    "Fix a consistent bedtime and wake time (incl. weekends)",
                    "No screens 30-60 minutes before bed",
                    "Caffeine cutoff at least 8 hours before sleep",
                ],
            })
        elif sleep.get("status") == "Above recommended range":
            recs.append({
                "category": "sleep", "priority": "low",
                "title": "Check unusually long sleep",
                "detail": "Regularly sleeping well beyond 9 hours is worth "
                          "discussing with a clinician, especially if paired "
                          "with daytime fatigue.",
                "actions": [
                    "Note sleep quality and daytime energy for a week",
                    "Mention it at your next check-up",
                ],
            })
        return recs

    def _hydration(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        hyd = analysis.get("hydration", {})
        if hyd.get("status") == "Low":
            recs.append({
                "category": "hydration", "priority": "low",
                "title": "Drink more water through the day",
                "detail": f"Logged intake is {hyd['value']:g} L. Mild chronic "
                          "dehydration can raise resting heart rate and "
                          "impair concentration.",
                "actions": [
                    "Keep a filled bottle visible at your desk",
                    "A glass of water with every meal and snack",
                    "Aim for roughly 2-3 L unless your doctor restricts fluids",
                ],
            })
        return recs

    def _lifestyle(self, analysis: Dict[str, Any], sex: str) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        record = analysis.get("record", {})

        if record.get("smoking"):
            recs.append({
                "category": "lifestyle", "priority": "urgent",
                "title": "Smoking cessation is the single highest-impact step",
                "detail": "Smoking multiplies cardiovascular risk on top of "
                          "your other readings. Quitting yields benefits "
                          "within weeks, not years.",
                "actions": [
                    "Set a quit date within the next 14 days",
                    "Ask a clinician about nicotine replacement or medication",
                    "Use a quitline or support program (e.g., 1-800-QUIT-NOW "
                    "in the US)",
                ],
            })

        alcohol = record.get("alcohol_drinks_per_week")
        limit = config.LIFESTYLE["alcohol_limit_female"] if sex == "female" \
            else config.LIFESTYLE["alcohol_limit_male"]
        if alcohol is not None and alcohol > limit:
            recs.append({
                "category": "lifestyle", "priority": "moderate",
                "title": "Bring alcohol within recommended limits",
                "detail": f"Logged {alcohol:g} drinks/week exceeds the common "
                          f"guideline of up to {limit:g}/week for your "
                          "profile; alcohol directly raises blood pressure "
                          "and triglycerides.",
                "actions": [
                    "Plan 3+ alcohol-free days per week",
                    "Alternate each alcoholic drink with a glass of water",
                    "Watch portion size - one 'drink' is smaller than most pours",
                ],
            })
        return recs

    def _monitoring(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        flags = analysis.get("risk_flags", [])
        if any(f["severity"] in ("urgent", "warning") for f in flags):
            recs.append({
                "category": "monitoring", "priority": "high",
                "title": "Re-measure flagged metrics weekly",
                "detail": "Trends matter more than single readings. Weekly "
                          "check-ins make improvement (or drift) visible "
                          "early and keep you accountable.",
                "actions": [
                    "Log the flagged metrics here at least once a week",
                    "Review the trend arrows after 4 weeks of consistent data",
                    "Share the log with your healthcare provider",
                ],
            })
        return recs

    def _positives(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reinforce what is already going well."""
        recs: List[Dict[str, Any]] = []
        subs = analysis.get("subscores", {})
        wins = sorted(
            ((cat, val) for cat, val in subs.items() if val >= 85),
            key=lambda kv: kv[1], reverse=True,
        )
        if wins:
            labels = {
                "bmi": "healthy body weight", "blood_pressure": "excellent "
                "blood pressure", "glucose": "healthy glucose control",
                "cholesterol": "healthy cholesterol", "heart_rate": "good "
                "resting heart rate", "sleep": "good sleep habits",
                "activity": "strong activity level", "lifestyle": "healthy "
                "lifestyle choices",
            }
            recs.append({
                "category": "monitoring", "priority": "positive",
                "title": "Keep doing what works",
                "detail": "These areas already score 85+ and are worth "
                          "protecting: "
                          + ", ".join(labels.get(c, c) for c, _v in wins) + ".",
                "actions": [
                    "Maintain your current routine in these areas",
                    "Re-run analysis monthly to confirm stability",
                ],
            })
        return recs
