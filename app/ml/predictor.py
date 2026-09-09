"""
Bloom ML Predictor Service

Fixes vs bloom_v2 broken version:
  1. REMOVED duplicate score_pcos_risk() call (line ~186) that referenced
     undefined `cycle_length` → NameError silently caught by dashboard
     try/except → everyone saw the same _EMPTY_PREDICTION placeholder.
  2. REMOVED dead first fertile-window block (lines ~147-151) that was
     immediately overwritten by the correct block below it.
  3. score_pcos_risk() now called exactly once with correct args.
  4. DAYS_UNTIL_MIN/MAX (0, 60) used for output clipping — not PHYS_MIN/MAX
     (14, 60) which are cycle-length bounds, not days-remaining bounds.
"""

import os, pickle, logging
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple, List

from app.ml.feature_spec import (
    build_feature_vector, IRREGULAR_THRESHOLD, DAYS_UNTIL_MIN, DAYS_UNTIL_MAX,
)

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_ml", "bloom_model.pkl")

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _bundle = pickle.load(f)
            logger.info(f"✅ Bloom model loaded (held-out MAE: ±{_bundle.get('mae','?')} days)")
        except Exception as e:
            logger.warning(f"⚠️  Model not loaded: {e}")
            _bundle = {}
    return _bundle


def score_pcos_risk(
    avg_cycle_length: float,
    cycle_variation: float,
    bmi: float,
    symptoms: List[str],
    stress: str,
) -> Tuple[str, float, List[str]]:
    """
    Return (risk_level, score_0_to_1, reasons).
    risk_level: 'low' | 'moderate' | 'high'
    """
    score   = 0.0
    reasons = []

    if avg_cycle_length > 35:
        score += 0.35
        reasons.append(f"Average cycle length {avg_cycle_length:.0f} days (>35)")
    elif avg_cycle_length > 32:
        score += 0.15
        reasons.append(f"Slightly long average cycle ({avg_cycle_length:.0f} days)")

    if cycle_variation > 7:
        score += 0.25
        reasons.append(f"High cycle variation (±{cycle_variation:.1f} days)")
    elif cycle_variation > 4:
        score += 0.10

    if bmi > 30:
        score += 0.15
        reasons.append(f"BMI {bmi:.1f} (obesity is a PCOS risk factor)")
    elif bmi < 17:
        score += 0.10
        reasons.append(f"Low BMI {bmi:.1f} can disrupt cycles")

    pcos_symptoms = {"acne", "hair loss", "fatigue", "bloating", "mood swings"}
    matched = pcos_symptoms & set(symptoms)
    if len(matched) >= 2:
        score += 0.15
        reasons.append(f"Multiple PCOS-linked symptoms: {', '.join(sorted(matched))}")
    elif len(matched) == 1:
        score += 0.05

    if stress == "high":
        score += 0.10
        reasons.append("Chronic high stress disrupts the HPG axis")

    score = min(score, 1.0)
    level = "high" if score >= 0.5 else ("moderate" if score >= 0.25 else "low")
    return level, round(score, 2), reasons


def predict(
    age: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "none",
    symptom: str = "none",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: float = 28.0,
    cycle_variation: float = 2.0,
    symptoms: Optional[List[str]] = None,
) -> dict:
    """
    Run the ML model and return a rich prediction dict.
    Falls back to heuristic if model is unavailable.

    `avg_previous` is the user's rolling historical average cycle length —
    the only legitimate proxy for expected cycle length at prediction time.
    There is no `cycle_length` argument: the current cycle's real length
    isn't known until it ends, so it can't be an input.
    """
    bundle = _load()
    feats  = build_feature_vector(
        age, days_since_last, mood, flow, symptom,
        stress, sleep, exercise, bmi,
        avg_previous, cycle_variation,
    )
    is_irregular = avg_previous > IRREGULAR_THRESHOLD

    if bundle and "model" in bundle:
        scaler = bundle["scaler"]
        Xs     = scaler.transform([feats])

        mdl  = bundle["irr_model"] if is_irregular else bundle["model"]
        # FIX: clip to DAYS_UNTIL bounds (0-60), not PHYS_MIN/MAX (14-60)
        # PHYS_MIN=14 would wrongly floor all near-term predictions at "14 days away"
        pred = float(np.clip(mdl.predict(Xs)[0], DAYS_UNTIL_MIN, DAYS_UNTIL_MAX))
        lo   = float(np.clip(bundle["q10"].predict(Xs)[0], DAYS_UNTIL_MIN, pred))
        hi   = float(np.clip(bundle["q90"].predict(Xs)[0], pred, DAYS_UNTIL_MAX))
        mae  = bundle.get("mae", 3.0)
    else:
        # Heuristic fallback — still useful, just less accurate
        pred = max(DAYS_UNTIL_MIN, float(avg_previous) - float(days_since_last))
        lo   = max(DAYS_UNTIL_MIN, pred - 5)
        hi   = min(DAYS_UNTIL_MAX, pred + 5)
        mae  = 5.0

    days_until = max(0, int(round(pred)))
    next_date  = date.today() + timedelta(days=days_until)

    # ── Cycle phase ───────────────────────────────────────────────────────────
    # Cap cycle_day so it never exceeds avg_previous
    # (edge case: user hasn't logged in a very long time)
    cycle_day     = min(int(days_since_last) + 1, int(avg_previous))
    ovulation_day = int(avg_previous) - 14

    if cycle_day <= 5:
        phase = "menstrual"
    elif cycle_day <= max(6, ovulation_day - 2):
        phase = "follicular"
    elif cycle_day <= ovulation_day + 3:
        phase = "ovulation"
    else:
        phase = "luteal"

    # ── Fertile window ────────────────────────────────────────────────────────
    # Ovulation ≈ predicted next period minus 14 days
    # Fertile window = 5 days before ovulation through 1 day after
    predicted_ovulation = next_date - timedelta(days=14)
    fertile_start_dt    = predicted_ovulation - timedelta(days=5)
    fertile_end_dt      = predicted_ovulation + timedelta(days=1)

    today = date.today()
    if fertile_start_dt < fertile_end_dt and fertile_end_dt >= today:
        fertile_start = max(fertile_start_dt, today).isoformat()
        fertile_end   = fertile_end_dt.isoformat()
    else:
        fertile_start = None
        fertile_end   = None

    # ── PCOS risk ─────────────────────────────────────────────────────────────
    # FIX: called exactly ONCE — previous version called it twice,
    # second call referenced undefined `cycle_length` → NameError
    pcos_level, pcos_score, pcos_reasons = score_pcos_risk(
        avg_previous, cycle_variation, bmi, symptoms or [], stress,
    )

    return {
        "days_until":       days_until,
        "next_period_date": next_date.isoformat(),
        "confidence_lo":    max(0, int(round(lo))),
        "confidence_hi":    int(round(hi)),
        "mae_days":         mae,
        "is_irregular":     is_irregular,
        "cycle_phase":      phase,
        "cycle_day":        cycle_day,
        "fertile_start":    fertile_start,
        "fertile_end":      fertile_end,
        "pcos_risk_level":  pcos_level,
        "pcos_risk_score":  pcos_score,
        "pcos_reasons":     pcos_reasons,
    }