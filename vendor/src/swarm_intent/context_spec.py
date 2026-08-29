"""
FROZEN interface between the STGT swarm model and the LLM interpretation layer.

This module holds the exact narrative vocabulary that ``build_tactical_context()``
(``inference.py``) is allowed to emit for each qualitative field: the finite set of
words a downstream LLM (Groq baseline or fine-tuned Qwen adapter) has ever been
prompted/trained to see. STGT's regression outputs may change scale or units when
the model is retrained (see ``calibration.py``) — but as long as whatever calibrator
sits between STGT's raw numbers and the narrative maps onto these SAME words, the
prompt format, the evaluator's vocabulary (``llm/prompts.py`` INTENT_FAMILIES /
THREAT_FAMILIES / ACTION_FAMILIES), and any already-fine-tuned adapter weights all
stay valid without any change on the LLM side of this boundary.

Do not add, remove, or reword a value here casually — every string is something a
fine-tuned model may have specific learned associations with, and something the
evaluator or a saved eval/demo JSON may already depend on verbatim. If the
vocabulary itself must change, that's a deliberate decision to make explicitly, not
a side effect of a calibration refit.

Every value below was copied verbatim out of the current ``inference.py`` —
nothing here is new or invented.
"""
from __future__ import annotations

# --- Velocity trend (inference.py build_tactical_context, local var "vel_trend") ---
VELOCITY_ACCELERATING = "accelerating"
VELOCITY_DECELERATING = "decelerating"
VELOCITY_STEADY = "steady"
VELOCITY_TREND_VALUES = (VELOCITY_ACCELERATING, VELOCITY_DECELERATING, VELOCITY_STEADY)

# --- Spread dynamics (inference.py build_tactical_context, local var "approach_summary") ---
SPREAD_CONVERGING = "converging (drones closing in)"
SPREAD_DISPERSING = "dispersing (drones spreading out)"
SPREAD_STABLE = "stable spread"
SPREAD_DYNAMICS_VALUES = (SPREAD_CONVERGING, SPREAD_DISPERSING, SPREAD_STABLE)

# --- Stability trend (inference.py build_tactical_context, local var "stab_trend") ---
STABILITY_DEGRADING = "degrading"
STABILITY_IMPROVING = "improving"
STABILITY_HOLDING = "holding"
STABILITY_TREND_VALUES = (STABILITY_DEGRADING, STABILITY_IMPROVING, STABILITY_HOLDING)

# --- Role differentiation (inference.py build_tactical_context, local var "role_flag") ---
ROLE_DIFFERENTIATION_PRESENT = "present"
ROLE_DIFFERENTIATION_NOT_PROMINENT = "not prominent"
ROLE_DIFFERENTIATION_VALUES = (ROLE_DIFFERENTIATION_PRESENT, ROLE_DIFFERENTIATION_NOT_PROMINENT)

# --- Classifier confidence phrasing ---
# Unlike the four fields above, confidence has NO categorical word-choice in the
# current code — build_tactical_context reports it as a raw number plus a raw count,
# never as a label (there is no "high confidence" / "low confidence" branch). The
# template below is the exact phrasing (verbatim from inference.py); there is no
# enum to freeze here because none exists yet. If a future change adds a categorical
# confidence label, give it a _VALUES tuple here the same way as the four fields above.
CONFIDENCE_LINE_TEMPLATE = "Classifier confidence: mean={mean_conf:.2f} ({low_conf} low-confidence windows)"

# --- Transition / no-transition phrasing (not one of the 5 calibrated fields, but
# part of the same frozen text interface — included for completeness since
# build_tactical_context emits it unconditionally on every call) ---
NO_TRANSITIONS_DETECTED = "No formation transitions detected."
