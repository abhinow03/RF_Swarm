"""
Build a supervised fine-tuning (SFT) dataset for the tactical-reasoning LLM.

Strategy (teacher-distillation + rule-based label cleaning):
  1. Sample many diverse swarm scenarios (formation pairs, speeds, stabilities,
     approach rates, drone counts, noise levels) -- stratified by RULES threat
     tier (AUDIT.md V5 Phase 1 step 1 fallback stratification; see
     STRATA_TARGETS below), not uniformly across all 49 pairs.
  2. For each, synthesise the SAME tactical-context + key-window prompt the
     real pipeline produces (so the fine-tuned model sees in-distribution input).
  3. Get a GOLD assessment. By default we call a strong TEACHER model (NVIDIA
     NIM-hosted Nemotron 3 Super 120B, AUDIT.md V5 Phase 1 step 0 -- Groq was
     retired as the teacher provider this phase) to draft it, then OVERRIDE
     threat_level / likely_intent / recommended_action with the canonical
     domain rule for that scenario. This removes teacher noise and makes the
     targets internally consistent.
  4. Write chat-format JSONL: [{"role":"user",...},{"role":"assistant",...}].

Why distill + clean rather than pure rules: the teacher supplies fluent,
varied `situation_summary` / `threat_reasoning` / `follow_up_watch` text, while
the rules guarantee the decision fields are correct and consistent. That is the
behaviour we actually want the small model to learn.

Run WITHOUT a teacher (--no-teacher) to get rule-only templated targets — lower
quality prose but zero API cost, useful for a first smoke test.

Usage:
    export NVIDIA_API_KEY=...
    python llm_finetuning/build_sft_dataset.py --out data/sft_train.jsonl
    # --n overrides the total row count; default is sum(STRATA_TARGETS) = 12000.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# Make the src/ package importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.config import BASE_FORMATIONS
from swarm_intent.inference import build_llm_prompt, OUTPUT_SCHEMA  # noqa: F401
from swarm_intent import context_spec as spec

# ---------------------------------------------------------------------------
# Empirical percentile breakpoints of the REAL STGT regression-label
# distributions (AUDIT.md sec V, swarm_data/_reg_distribution_analysis.npz,
# n=5879 real training sequences from the teammate's retrained checkpoint).
# 1%-step empirical CDF (101 points/field) -- close enough to true empirical
# sampling for narrative-text purposes, and small enough to commit as a
# literal so synth_context() has NO runtime dependency on swarm_data/ (that
# folder is gitignored, teammate-provided, not present on a fresh clone/CI).
# Regenerate via scripts/export_real_reg_percentiles.py if the underlying
# distribution changes (e.g. after a further STGT retrain).
# ---------------------------------------------------------------------------
REAL_REG_PERCENTILES = {
    'velocity_physical': [2.8423, 3.1661, 3.2645, 3.3477, 3.4076, 3.4851, 3.5319, 3.5955, 3.647, 3.6994, 3.7513, 3.7985, 3.8434, 3.8945, 3.9413, 3.9938, 4.0383, 4.0796, 4.1308, 4.1803, 4.2212, 4.2735, 4.3325, 4.3733, 4.4115, 4.4582, 4.4963, 4.5535, 4.5983, 4.6517, 4.6969, 4.7488, 4.7965, 4.8474, 4.8935, 4.9368, 4.9903, 5.0386, 5.0849, 5.1244, 5.1746, 5.2159, 5.2659, 5.3153, 5.3636, 5.4158, 5.4609, 5.5077, 5.5435, 5.5871, 5.6242, 5.6788, 5.7227, 5.7755, 5.8331, 5.8842, 5.9305, 5.9744, 6.0229, 6.0649, 6.1064, 6.1574, 6.2161, 6.259, 6.3084, 6.3512, 6.4002, 6.4505, 6.5018, 6.5599, 6.6008, 6.6504, 6.7077, 6.7551, 6.8001, 6.8444, 6.9072, 6.9561, 7.0139, 7.0632, 7.1137, 7.1593, 7.2034, 7.2559, 7.3121, 7.3623, 7.4214, 7.4636, 7.5131, 7.5743, 7.6337, 7.6858, 7.7346, 7.784, 7.8377, 7.9038, 7.9792, 8.0658, 8.1777, 8.3862, 9.2241],
    'approach_rate': [-0.5792, -0.3677, -0.3215, -0.2883, -0.2639, -0.2498, -0.2377, -0.229, -0.2199, -0.2127, -0.2044, -0.1981, -0.1899, -0.1835, -0.1775, -0.1704, -0.1644, -0.1586, -0.1536, -0.1473, -0.1428, -0.1379, -0.1333, -0.1286, -0.1248, -0.119, -0.1138, -0.1099, -0.1061, -0.1027, -0.0991, -0.095, -0.0909, -0.0875, -0.0844, -0.0808, -0.0774, -0.0734, -0.0704, -0.0674, -0.0647, -0.0619, -0.0582, -0.0555, -0.0525, -0.0499, -0.0471, -0.0442, -0.0413, -0.039, -0.0362, -0.033, -0.0302, -0.0275, -0.0253, -0.0226, -0.0202, -0.0174, -0.0148, -0.0123, -0.0094, -0.007, -0.0035, 0.0001, 0.0033, 0.0062, 0.0088, 0.011, 0.0133, 0.016, 0.0185, 0.0217, 0.0245, 0.0276, 0.0303, 0.0334, 0.036, 0.0392, 0.0429, 0.0462, 0.0498, 0.0534, 0.0573, 0.0612, 0.066, 0.0703, 0.0747, 0.0796, 0.0853, 0.0905, 0.0971, 0.1053, 0.1131, 0.1241, 0.136, 0.1507, 0.1681, 0.1914, 0.2272, 0.2755, 0.4558],
    'stability': [0.3429, 0.5052, 0.5433, 0.5635, 0.5753, 0.5827, 0.5908, 0.5986, 0.6055, 0.6128, 0.6181, 0.6247, 0.6308, 0.6369, 0.6443, 0.6504, 0.6589, 0.6652, 0.672, 0.6776, 0.683, 0.6898, 0.6946, 0.7009, 0.7074, 0.7141, 0.7207, 0.726, 0.7324, 0.7393, 0.746, 0.7519, 0.7577, 0.7617, 0.7661, 0.7703, 0.7746, 0.7799, 0.7858, 0.7902, 0.7934, 0.7964, 0.8002, 0.8048, 0.8088, 0.8127, 0.8163, 0.8202, 0.8249, 0.8294, 0.8335, 0.8374, 0.8409, 0.8433, 0.8464, 0.8497, 0.8525, 0.8547, 0.8568, 0.8591, 0.8609, 0.8632, 0.8646, 0.8661, 0.8681, 0.8699, 0.8722, 0.8742, 0.8764, 0.8787, 0.8804, 0.8824, 0.8852, 0.8882, 0.8909, 0.8933, 0.8963, 0.8984, 0.9001, 0.9026, 0.9048, 0.9068, 0.9088, 0.9107, 0.9129, 0.9147, 0.9165, 0.9181, 0.9207, 0.9234, 0.9259, 0.9284, 0.931, 0.9335, 0.9357, 0.9379, 0.9406, 0.9426, 0.9447, 0.9495, 0.9693],
    'delta_v_physical': [-2.6399, -1.5944, -1.3288, -1.166, -1.0591, -0.9816, -0.9, -0.8495, -0.7964, -0.7468, -0.7087, -0.6679, -0.628, -0.5935, -0.5563, -0.5327, -0.5044, -0.4801, -0.4544, -0.4352, -0.4146, -0.3935, -0.3747, -0.3527, -0.3383, -0.317, -0.2922, -0.2711, -0.2537, -0.2354, -0.2224, -0.2094, -0.19, -0.1728, -0.1564, -0.1422, -0.1268, -0.1127, -0.1011, -0.0857, -0.0729, -0.0632, -0.0538, -0.0444, -0.0367, -0.0282, -0.0194, -0.0138, -0.0074, -0.0005, 0.0063, 0.0134, 0.0204, 0.0261, 0.0334, 0.0396, 0.0458, 0.0543, 0.064, 0.0749, 0.0835, 0.0954, 0.1052, 0.1201, 0.1344, 0.1461, 0.1615, 0.1797, 0.1984, 0.2142, 0.2324, 0.2493, 0.2646, 0.283, 0.302, 0.3203, 0.3438, 0.3633, 0.3834, 0.4049, 0.428, 0.4524, 0.4753, 0.5011, 0.5301, 0.5583, 0.5871, 0.6122, 0.6394, 0.6782, 0.7199, 0.7638, 0.8092, 0.883, 0.9373, 0.9971, 1.0875, 1.1772, 1.3174, 1.5762, 2.7365],
    'delta_stability': [-0.7918, -0.5187, -0.4694, -0.4325, -0.4049, -0.3759, -0.3522, -0.3322, -0.3166, -0.302, -0.2922, -0.2826, -0.273, -0.2644, -0.2526, -0.2433, -0.2349, -0.2237, -0.2147, -0.2054, -0.197, -0.1879, -0.1802, -0.1727, -0.1669, -0.1594, -0.1515, -0.1455, -0.1393, -0.1332, -0.1285, -0.1241, -0.1199, -0.1171, -0.1132, -0.1097, -0.1053, -0.1019, -0.099, -0.0956, -0.0929, -0.0892, -0.0863, -0.0821, -0.0781, -0.0746, -0.0701, -0.066, -0.0617, -0.0579, -0.0521, -0.0479, -0.0438, -0.0376, -0.0324, -0.0254, -0.0183, -0.0093, -0.001, 0.0093, 0.018, 0.0285, 0.0355, 0.0427, 0.048, 0.053, 0.0592, 0.0655, 0.0696, 0.0748, 0.0783, 0.0832, 0.0885, 0.0929, 0.0967, 0.1007, 0.1045, 0.1083, 0.1124, 0.1171, 0.1213, 0.1258, 0.1325, 0.1408, 0.1525, 0.1614, 0.1717, 0.1792, 0.1864, 0.1926, 0.1985, 0.2073, 0.2173, 0.2293, 0.2392, 0.2565, 0.2734, 0.2958, 0.3317, 0.3736, 0.4914],
}


def _sample_real(rng: random.Random, field: str) -> float:
    """Bootstrap-sample one point from REAL_REG_PERCENTILES[field]'s empirical
    CDF -- a real STGT-population value, not a hand-picked uniform range."""
    breakpoints = REAL_REG_PERCENTILES[field]
    return rng.choice(breakpoints)


# ---------------------------------------------------------------------------
# Canonical domain rules: scenario -> (threat, intent, action). This is the
# ground-truth decision logic the model must internalise. EDIT THESE with your
# domain expert — they define what "correct" means for the whole project.
# ---------------------------------------------------------------------------
RULES = {
    # (from, to) : (threat_level, likely_intent, recommended_action)

    # --- steady-state ---
    ("v_shape",      "v_shape"):      ("medium",   "surveillance",        "increase_surveillance"),
    ("encirclement", "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "column"):       ("low",      "patrol",              "monitor"),
    ("diamond",      "diamond"):      ("low",      "patrol",              "monitor"),
    ("dispersed",    "dispersed"):    ("low",      "surveillance",        "monitor"),
    ("converging",   "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "shield"):       ("medium",   "defensive",           "monitor"),

    # --- to converging ---
    ("v_shape",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("encirclement", "converging"):   ("critical", "encircle",            "deploy_countermeasure"),
    ("column",       "converging"):   ("high",     "approach",            "alert_operator"),
    ("diamond",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("dispersed",    "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "converging"):   ("medium",   "approach",            "increase_surveillance"),

    # --- to encirclement ---
    ("v_shape",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("diamond",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("dispersed",    "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("converging",   "encirclement"): ("critical", "encircle",            "deploy_countermeasure"),
    ("shield",       "encirclement"): ("medium",   "encircle",            "increase_surveillance"),

    # --- from encirclement (de-escalating) ---
    ("encirclement", "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("encirclement", "column"):       ("low",      "withdraw",            "monitor"),
    ("encirclement", "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("encirclement", "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("encirclement", "shield"):       ("medium",   "defensive",           "monitor"),

    # --- from converging (de-escalating) ---
    ("converging",   "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("converging",   "column"):       ("low",      "patrol",              "monitor"),
    ("converging",   "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("converging",   "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("converging",   "shield"):       ("medium",   "defensive",           "monitor"),

    # --- v_shape remaining ---
    ("v_shape",      "column"):       ("low",      "transit",             "monitor"),
    ("v_shape",      "diamond"):      ("medium",   "defensive_transition","monitor"),
    ("v_shape",      "dispersed"):    ("low",      "area_search",         "monitor"),
    ("v_shape",      "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- column remaining ---
    ("column",       "v_shape"):      ("high",     "attack_preparation",  "alert_operator"),
    ("column",       "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("column",       "dispersed"):    ("low",      "area_search",         "monitor"),
    ("column",       "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- diamond remaining ---
    ("diamond",      "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("diamond",      "column"):       ("low",      "transit",             "monitor"),
    ("diamond",      "dispersed"):    ("medium",   "area_search",         "monitor"),
    ("diamond",      "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- dispersed remaining ---
    ("dispersed",    "v_shape"):      ("high",     "rally",               "alert_operator"),
    ("dispersed",    "column"):       ("low",      "transit",             "monitor"),
    ("dispersed",    "diamond"):      ("medium",   "consolidate",         "increase_surveillance"),
    ("dispersed",    "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- shield remaining ---
    ("shield",       "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("shield",       "column"):       ("medium",   "reposition",          "monitor"),
    ("shield",       "diamond"):      ("medium",   "defensive",           "monitor"),
    ("shield",       "dispersed"):    ("low",      "surveillance",        "monitor"),
}
DEFAULT_RULE = ("medium", "reposition", "monitor")  # ponytail: safety net only, should never fire now

# ---------------------------------------------------------------------------
# AUDIT.md V5 Phase 1 step 1: fallback stratification. RULES has only 2
# distinct critical pairs (below the ~5-pair halt-gate threshold; see
# docs/V5_LOG.md), so a uniform 3,000 rows/tier would force 1,500 rows onto
# each of the 2 critical pairs alone -- far more repetition than the other
# tiers' pairs get. Instead: cap critical at 600 rows/pair (2 x 600 = 1,200)
# and redistribute the 1,800-row shortfall proportionally (+600 each) across
# the other three tiers. RULES itself is untouched (no critical-pair
# extension applied -- that needs Dr. Patil sign-off, not taken here).
# ---------------------------------------------------------------------------
STRATA_TARGETS = {"low": 3600, "medium": 3600, "high": 3600, "critical": 1200}


def build_stratified_pairs(rng: random.Random, targets: dict[str, int]) -> list[tuple]:
    """Returns a shuffled list of (form_a, form_b) pairs, length sum(targets.values()),
    with exactly `targets[tier]` rows drawn from tier `tier`'s pairs and, within
    a tier, counts spread as evenly as possible across that tier's pairs (so no
    single pair dominates a tier's rows)."""
    tier_pairs: dict[str, list[tuple]] = {}
    for pair, (threat, _intent, _action) in RULES.items():
        tier_pairs.setdefault(threat, []).append(pair)

    out = []
    for tier, target in targets.items():
        pairs = sorted(tier_pairs.get(tier, []))  # sorted: deterministic before shuffle
        if not pairs:
            continue
        base, remainder = divmod(target, len(pairs))
        counts = [base + (1 if i < remainder else 0) for i in range(len(pairs))]
        rng.shuffle(counts)  # so the +1 remainder doesn't always land on the same pairs
        for pair, count in zip(pairs, counts):
            out.extend([pair] * count)
    rng.shuffle(out)
    return out


def synth_context(form_a: str, form_b: str, rng: random.Random) -> tuple[str, list]:
    """Fabricate a realistic tactical-context string + key windows for a scenario.

    Independent of a trained checkpoint so you can build SFT data immediately.

    RECALIBRATED to the real STGT population (AUDIT.md sec V/W/continued):
    centroid_velocity, approach_rate, delta_v and stability are now bootstrap-
    sampled from REAL_REG_PERCENTILES (real STGT regression labels, n=5879,
    sec V) instead of hand-picked uniform ranges. Previously this function used
    e.g. approach ~U(-1.5,1.5) and delta_v ~U(-1.0,2.0), sampling ranges 4-1000x
    wider than the real (then normalised-space, pre-retrain) pipeline values —
    see the superseded sec F/F2 finding. Post retrain + this fix, the sampled
    proportions of converging/dispersing/stable now track the real ~29.7%/9.7%/
    60.6% split (previously ~70%/20%/10%, see the before/after table this
    session's commit reports) instead of an arbitrary uniform artifact.

    stab_early/stab_late are derived from two INDEPENDENT real-population draws
    (a mean-stability value and a delta-stability value, both from their own
    real marginal distributions) rather than a true joint real (early, late)
    pair — REAL_REG_PERCENTILES only stores marginal 1%-step empirical CDFs per
    field (chosen so this function has no runtime dependency on the gitignored
    swarm_data/ folder), not the underlying joint samples. This preserves
    realistic marginal ranges and realistic swing magnitudes but not their true
    joint correlation; deemed close enough for narrative-text purposes.

    Threshold logic below matches calibration.py's AbsoluteCalibrator exactly (same
    +-0.5 velocity, +-0.1 approach, +-0.1 stability-delta cutoffs) -- this function
    used to implement a narrower BINARY version of two of these three thresholds
    (spread_dynamics had no "dispersing" branch at all; stability_trend compared a
    single scalar against one cutoff instead of an early/late delta, so "improving"
    was unreachable), and never rendered "Role differentiation: ..." into the context
    text at all despite build_tactical_context() always emitting it in production —
    two real train/serve mismatches (AUDIT.md sec X/Z-adjacent). Fixed here to reach
    every context_spec.py value; RULES and its (form_a, form_b) decision logic are
    untouched -- these are narrative-grammar fixes only.
    """
    transitioning = form_a != form_b
    mean_conf = round(rng.uniform(0.7, 0.98), 2)
    mean_stab_draw = _sample_real(rng, "stability")
    delta_stab_draw = _sample_real(rng, "delta_stability")
    stab_early = round(min(1.0, max(0.0, mean_stab_draw - delta_stab_draw / 2)), 2)
    stab_late = round(min(1.0, max(0.0, mean_stab_draw + delta_stab_draw / 2)), 2)
    mean_stab = round((stab_early + stab_late) / 2, 2)
    approach = round(_sample_real(rng, "approach_rate"), 3)
    delta_v = round(_sample_real(rng, "delta_v_physical"), 2)
    vel_trend = (spec.VELOCITY_ACCELERATING if delta_v > 0.5
                else spec.VELOCITY_DECELERATING if delta_v < -0.5 else spec.VELOCITY_STEADY)
    if stab_late < stab_early - 0.1:
        stab_trend = spec.STABILITY_DEGRADING
    elif stab_late > stab_early + 0.1:
        stab_trend = spec.STABILITY_IMPROVING
    else:
        stab_trend = spec.STABILITY_HOLDING
    if approach < -0.1:
        spread_trend = spec.SPREAD_CONVERGING
    elif approach > 0.1:
        spread_trend = spec.SPREAD_DISPERSING
    else:
        spread_trend = spec.SPREAD_STABLE
    role_present = rng.random() < 0.3  # matches production's minority-case framing (a
    # role split only shows up when one drone strays >2x the group's median distance)
    role_str = spec.ROLE_DIFFERENTIATION_PRESENT if role_present else spec.ROLE_DIFFERENTIATION_NOT_PROMINENT
    dominant = form_a
    history = f"{form_a} -> transitioning -> {form_b}" if transitioning else form_a

    ctx = "\n".join([
        f"Observation window: 0.0s - 60.0s (9 overlapping windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {history}",
        (f"Transition detected at t=20.0s: {form_a} -> {form_b}"
         if transitioning else "No formation transitions detected."),
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_trend} (mean={mean_stab:.2f})",
        f"Spread dynamics: {spread_trend} (mean approach_rate={approach:.3f})",
        f"Role differentiation: {role_str}",
        f"Classifier confidence: mean={mean_conf:.2f}",
    ])
    key_windows = [
        {"t": "0.0-25.0s", "formation": form_a, "confidence": mean_conf,
         "velocity": round(_sample_real(rng, "velocity_physical"), 3), "approach": approach,
         "stability": stab_early, "from": None, "to": None, "role_differentiation": role_present},
        {"t": "35.0-60.0s", "formation": "transitioning" if transitioning else form_a,
         "confidence": mean_conf, "velocity": round(_sample_real(rng, "velocity_physical"), 3),
         "approach": approach, "stability": stab_late,
         "from": form_a if transitioning else None,
         "to": form_b if transitioning else None, "role_differentiation": role_present},
    ]
    return ctx, key_windows


def build_teacher_prompt(form_a: str, form_b: str, ctx: str, prompt: Optional[str] = None) -> str:
    """The teacher must see the same schema-bearing prompt the student trains
    on -- otherwise its JSON comes back with different keys and every
    draft.get() silently falls back to the template (the bug that produced a
    fully-templated v2 dataset). Only the teacher sees the canonical rule
    labels (the student prompt in the dataset stays label-free) -- without
    this, a teacher that disagrees with RULES writes threat_reasoning arguing
    for its own prediction, which then contradicts the overridden
    threat_level in the saved row."""
    threat, intent, action = RULES.get((form_a, form_b), DEFAULT_RULE)
    base = prompt or f"Given this UAV swarm tactical context, write the JSON assessment.\n\n{ctx}"
    return (
        f"{base}\n\n"
        f"GROUND TRUTH from the canonical rule engine — use exactly these values, "
        f'do not deviate: threat_level="{threat}", likely_intent="{intent}", '
        f'recommended_action="{action}". Write situation_summary, threat_reasoning, '
        f"key_indicators and follow_up_watch so they genuinely support this assessment."
    )


def finalize_assessment(form_a: str, form_b: str, draft: Optional[dict]) -> tuple[dict, bool]:
    """Overrides the decision fields with the canonical rule (clean labels),
    filling in templated prose wherever the teacher draft is missing/failed.
    Returns (assessment, used_teacher)."""
    threat, intent, action = RULES.get((form_a, form_b), DEFAULT_RULE)
    if not isinstance(draft, dict) or "error" in draft:
        draft = {}
    used_teacher = bool(draft.get("situation_summary"))
    return {
        "situation_summary": draft.get("situation_summary",
            f"The swarm is in a {form_a} formation"
            + (f" transitioning to {form_b}." if form_a != form_b else ", holding steady.")),
        "threat_level": threat,
        "threat_reasoning": draft.get("threat_reasoning",
            f"{form_a}->{form_b} dynamics with the observed approach rate indicate a {threat} threat."),
        "likely_intent": intent,
        "recommended_action": action,
        "confidence_in_assessment": "high",
        "key_indicators": draft.get("key_indicators",
            [f"{form_a}->{form_b} transition", "approach rate", "classifier confidence"]),
        "follow_up_watch": draft.get("follow_up_watch",
            "Monitor formation and approach rate over the next window."),
    }, used_teacher


def gold_assessment(form_a, form_b, ctx, teacher=None, prompt=None) -> tuple[dict, bool]:
    """Unbatched convenience wrapper (kept for --no-stratify/single-item call
    sites and any external importers) -- one teacher.complete() call, then
    finalize_assessment(). main()'s generation loop calls the two halves
    separately so it can batch build_teacher_prompt()'s output through
    teacher.complete_batch() instead."""
    draft = teacher.complete(build_teacher_prompt(form_a, form_b, ctx, prompt)) if teacher is not None else {}
    return finalize_assessment(form_a, form_b, draft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="total row count. Default: sum(STRATA_TARGETS) = 12000 "
                         "under stratified sampling, 600 under --no-stratify.")
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-teacher", action="store_true",
                    help="skip the NVIDIA teacher; use templated targets only")
    ap.add_argument("--teacher-model", default=None,
                    help="NVIDIA NIM model id for the teacher (default: NvidiaClient's "
                         "default, nvidia/nemotron-3-super-120b-a12b).")
    ap.add_argument("--no-stratify", action="store_true",
                    help="sample pairs uniformly (old behaviour) instead of the "
                         "RULES-threat-tier stratification (AUDIT.md V5 Phase 1 step 1)")
    ap.add_argument("--append", action="store_true",
                    help="add to the existing --out dataset instead of overwriting it "
                         "(loads train+val, re-shuffles, re-splits). For accumulating "
                         "teacher rows across daily quota windows / teacher models.")
    ap.add_argument("--teacher-only", action="store_true",
                    help="drop rows where the teacher fell back to templated prose")
    ap.add_argument("--max-teacher-fails", type=int, default=20,
                    help="stop after this many consecutive teacher fallbacks (daily "
                         "quota exhausted) instead of generating template filler; 0 = off")
    ap.add_argument("--seed", type=int, default=None,
                    help="default 42, but randomized under --append so re-runs don't "
                         "regenerate duplicate scenarios")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="teacher API calls in flight at once (thread pool, "
                         "NvidiaClient/GroqClient generate_batch). 1 = fully serial. "
                         "Keep within whatever concurrent-request limit the provider allows.")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = 42 if not args.append else random.SystemRandom().randrange(1 << 30)
    if args.n is None:
        args.n = sum(STRATA_TARGETS.values()) if not args.no_stratify else 600

    teacher = None
    if not args.no_teacher:
        from swarm_intent.llm.client import NvidiaClient
        # max_tokens=3072: reasoning teachers (qwen3 family) spend hundreds of
        # <think> tokens before the JSON; 1024 truncates mid-answer. Harmless
        # cap for non-reasoning teachers, which finish well under it.
        teacher = (NvidiaClient(model=args.teacher_model, max_tokens=3072)
                   if args.teacher_model
                   else NvidiaClient(max_tokens=3072))  # reads NVIDIA_API_KEY

    rng = random.Random(args.seed)
    if args.no_stratify:
        pairs = list(RULES.keys()) + [(a, b) for a in BASE_FORMATIONS for b in BASE_FORMATIONS]
        pair_sequence = [rng.choice(pairs) for _ in range(args.n)]
    else:
        # scale STRATA_TARGETS proportionally if --n overrides the 12000 default
        scale = args.n / sum(STRATA_TARGETS.values())
        targets = {tier: round(target * scale) for tier, target in STRATA_TARGETS.items()}
        pair_sequence = build_stratified_pairs(rng, targets)
    args.n = len(pair_sequence)  # rounding in the stratified path can shift the exact total
    rows = []
    n_teacher_ok = 0
    n_gen = 0
    consec_fails = 0
    t0 = time.monotonic()
    stop_early = False
    chunk_size = max(1, args.concurrency)
    for chunk_start in range(0, len(pair_sequence), chunk_size):
        chunk = pair_sequence[chunk_start:chunk_start + chunk_size]
        # Cheap, CPU-only per-item work stays sequential; only the teacher
        # network calls are batched/concurrent.
        chunk_ctx, chunk_prompts = [], []
        for form_a, form_b in chunk:
            ctx, key_windows = synth_context(form_a, form_b, rng)
            prompt = build_llm_prompt(
                predictions=[{**kw, "time_start_s": 0, "time_end_s": 0,
                              "formation_type": kw["formation"], "centroid_velocity": kw["velocity"],
                              "approach_rate": kw["approach"], "formation_stability": kw["stability"],
                              "formation_confidence": kw["confidence"],
                              "role_differentiation": kw["role_differentiation"],
                              "transition_from": kw["from"], "transition_to": kw["to"]} for kw in key_windows],
                tactical_context=ctx, summary={})
            chunk_ctx.append(ctx)
            chunk_prompts.append(prompt)

        if teacher is not None:
            teacher_prompts = [build_teacher_prompt(fa, fb, ctx, prompt) for (fa, fb), ctx, prompt
                               in zip(chunk, chunk_ctx, chunk_prompts)]
            drafts = teacher.complete_batch(teacher_prompts, batch_size=chunk_size)
        else:
            drafts = [None] * len(chunk)

        for (form_a, form_b), prompt, draft in zip(chunk, chunk_prompts, drafts):
            gold, used_teacher = finalize_assessment(form_a, form_b, draft)
            n_gen += 1
            n_teacher_ok += used_teacher
            if teacher is not None:
                consec_fails = 0 if used_teacher else consec_fails + 1
                if args.max_teacher_fails and consec_fails >= args.max_teacher_fails:
                    print(f"STOPPING EARLY: {consec_fails} consecutive teacher fallbacks — "
                          "daily token quota likely exhausted. Re-run later with --append, "
                          "or switch quota pools with --teacher-model.", flush=True)
                    stop_early = True
                    break
            if args.teacher_only and not used_teacher:
                continue
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold, indent=2)},
            ]})

        elapsed = time.monotonic() - t0
        eta = elapsed / n_gen * (args.n - n_gen) if n_gen else 0.0
        print(f"{n_gen}/{args.n} generated ({len(rows)} kept, {n_teacher_ok} with "
              f"teacher prose) [{elapsed/60:.1f} min elapsed, ~{eta/60:.1f} min left]",
              flush=True)
        if stop_early:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    val_path = args.out.replace(".jsonl", "_val.jsonl")

    if args.append:
        def load_rows(path):
            if not os.path.exists(path):
                return []
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        existing = load_rows(args.out) + load_rows(val_path)
        print(f"--append: {len(existing)} existing + {len(rows)} new rows")
        rows = existing + rows
        # same scenario generated twice (e.g. an accidental same-seed re-run)
        # would leak between train and val after the re-split — drop dupes
        seen, uniq = set(), []
        for r in rows:
            key = r["messages"][0]["content"]
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        if len(uniq) < len(rows):
            print(f"  dropped {len(rows) - len(uniq)} duplicate scenarios")
        rows = uniq

    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val, train = rows[:n_val], rows[n_val:]
    with open(args.out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    total = time.monotonic() - t0
    print(f"Wrote {len(train)} train -> {args.out} and {len(val)} val -> {val_path} "
          f"in {total/60:.1f} min ({total/max(n_gen, 1):.1f} s/example)")
    if teacher is not None:
        pct = 100.0 * n_teacher_ok / max(n_gen, 1)
        print(f"Teacher prose used in {n_teacher_ok}/{n_gen} generated examples ({pct:.0f}%)")
        if pct < 50 and not args.teacher_only:
            print("WARNING: most examples fell back to templated prose — "
                  "check the teacher model/key before training on this file.")


if __name__ == "__main__":
    main()
