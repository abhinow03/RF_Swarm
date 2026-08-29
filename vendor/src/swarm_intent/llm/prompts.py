"""
Semantic vocabularies, matching helpers, evaluation test cases, and the judge
prompt. Migrated from the eval cell so they live in one importable place.
"""
from __future__ import annotations

INTENT_FAMILIES = {
    # NOTE: "attack" (a former entry here) was removed — RULES (build_sft_dataset.py)
    # never emits it, so it was unreachable dead vocabulary. If a future RULES change
    # adds a genuine attack-in-progress intent distinct from attack_preparation, restore
    # a family for it here with the corresponding RULES value.
    "approach":     {"approach", "advancing", "closing", "moving toward", "converging"},
    "surveillance": {"surveillance", "observe", "monitor", "scout", "reconnaissance", "recon"},
    "patrol":       {"patrol", "loiter", "holding", "circling", "maintaining"},
    "reposition":   {"repositioning", "reposition", "maneuver", "relocating", "reorganising", "reorganizing"},
    "defensive":    {"defensive", "defend", "shield", "protect", "guarding"},
    "encircle":     {"encirclement", "encircle", "surround", "flanking", "flank"},
    # Families below cover the remaining RULES intents (build_sft_dataset.py),
    # so rule-trained answers are never scored as hallucinations.
    "withdraw":     {"withdraw", "withdrawing", "retreat", "retreating", "disengage"},
    "regroup":      {"regroup", "regrouping", "reform", "reforming"},
    "consolidate":  {"consolidate", "consolidating", "tightening"},
    "transit":      {"transit", "transiting", "en route", "traveling", "travelling"},
    "defensive_transition": {"defensive_transition", "defensive transition"},
    "area_search":  {"area_search", "area search", "search", "sweep", "sweeping"},
    "attack_preparation": {"attack_preparation", "attack preparation", "preparing to attack"},
    "rally":        {"rally", "rallying", "massing", "mustering"},
    # Legal per OUTPUT_SCHEMA but not a RULES output — a correct abstention under
    # degraded/ambiguous input. Without this family, is_hallucination() flagged every
    # correct "unknown" response as a hallucination.
    "unknown":      {"unknown", "unclear", "indeterminate", "insufficient data"},
}
THREAT_FAMILIES = {
    "low":      {"low", "minimal", "none"},
    "medium":   {"medium", "moderate", "elevated"},
    "high":     {"high", "severe", "serious"},
    "critical": {"critical", "extreme", "imminent"},
    # Added so threat_level has the same schema-legal abstention value
    # likely_intent already has (see "unknown" above) -- previously abstention
    # rows had to assert a fake threat_level (build_abstain_rows.py hardcoded
    # "medium"; baselines.py's _abstain() already emitted "unknown" here despite
    # it not being schema-legal, see its own comment). Same token set as
    # ABSTENTION_TOKENS/INTENT_FAMILIES["unknown"].
    "unknown":  {"unknown", "unclear", "indeterminate", "insufficient data"},
}
# Covers every RULES-emitted action plus "intercept", which is schema-legal
# (OUTPUT_SCHEMA in inference.py) but never appears in RULES.
ACTION_FAMILIES = {
    "monitor":               {"monitor", "monitoring", "observe", "watch", "no action needed",
                               "continue monitoring", "routine monitoring"},
    "increase_surveillance": {"increase_surveillance", "increase surveillance",
                               "heightened surveillance", "closer watch", "step up surveillance",
                               "enhanced monitoring"},
    "alert_operator":        {"alert_operator", "alert operator", "notify operator",
                               "alert the operator", "flag to operator", "escalate to operator"},
    "deploy_countermeasure": {"deploy_countermeasure", "deploy countermeasure", "countermeasure",
                               "deploy countermeasures", "engage countermeasures",
                               "activate countermeasure"},
    "intercept":             {"intercept", "intercepting", "interception", "engage", "neutralize",
                               "neutralise"},
}
ALL_VALID_INTENT_TOKENS = {t for fam in INTENT_FAMILIES.values() for t in fam}
ALL_VALID_THREAT_TOKENS = {t for fam in THREAT_FAMILIES.values() for t in fam}
ALL_VALID_ACTION_TOKENS = {t for fam in ACTION_FAMILIES.values() for t in fam}

# Single source of truth for "this response is abstaining", used by evaluate_llm to
# exclude abstained responses from accuracy/hallucination scoring (see evaluate.py's
# module docstring) and by baselines.py's rules_lookup to signal "could not extract
# a formation pair, will not guess". Same token set as INTENT_FAMILIES["unknown"].
ABSTENTION_TOKENS = {"unknown", "unclear", "indeterminate", "insufficient data"}


def is_abstention(likely_intent: str) -> bool:
    return likely_intent.strip().lower() in ABSTENTION_TOKENS


def match_intent(predicted: str, expected_family: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in INTENT_FAMILIES.get(expected_family, set()))


def match_threat(predicted: str, expected_level: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in THREAT_FAMILIES.get(expected_level, set()))


def match_action(predicted: str, expected_family: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in ACTION_FAMILIES.get(expected_family, set()))


def is_hallucination(predicted_intent: str, predicted_threat: str) -> bool:
    pi, pt = predicted_intent.lower().strip(), predicted_threat.lower().strip()
    intent_ok = any(tok in pi or pi in tok for tok in ALL_VALID_INTENT_TOKENS)
    threat_ok = any(tok in pt or pt in tok for tok in ALL_VALID_THREAT_TOKENS)
    return not (intent_ok and threat_ok)


# Evaluation scenarios. expected_intent/expected_action use INTENT_FAMILIES/
# ACTION_FAMILIES keys. expected_action values are cross-referenced from RULES
# (build_sft_dataset.py) for each case's (formation_a, formation_b) pair — not
# independently chosen — so they agree with the canonical rule table by construction.
#
# ORIGINAL_TEST_CASES: the first 6 cases this project ever had. Left byte-for-byte
# unchanged so results measured against them (every eval JSON in evaluation/ prior
# to the RULES-coverage expansion below) stay comparable. "strict" is a legacy
# field from the original eval cell -- nothing in this codebase reads it anymore;
# kept only because removing it would touch lines that must stay identical.
ORIGINAL_TEST_CASES = [
    {"name": "Converging Attack", "formation_a": "dispersed", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator", "strict": False},
    {"name": "Stable Patrol", "formation_a": "column", "formation_b": "column",
     "expected_intent": "patrol", "expected_threat": "low",
     "expected_action": "monitor", "strict": True},
    {"name": "Encirclement Behavior", "formation_a": "v_shape", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator", "strict": False},
    {"name": "Defensive Shield", "formation_a": "shield", "formation_b": "shield",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor", "strict": True},
    {"name": "Area Search", "formation_a": "diamond", "formation_b": "dispersed",
     "expected_intent": "area_search", "expected_threat": "medium",
     "expected_action": "monitor", "strict": False},
    {"name": "Breaking Contact", "formation_a": "converging", "formation_b": "dispersed",
     "expected_intent": "withdraw", "expected_threat": "low",
     "expected_action": "monitor", "strict": False},
]

# RULES_COVERAGE_CASES: one case per (formation_a, formation_b) key in RULES
# (build_sft_dataset.py) -- ALL 49 of them. BASE_FORMATIONS has 7 members and
# 7*7 = 49 = len(RULES) exactly, so RULES is a complete function over every
# ordered formation pair: there is no 50th "remaining" pair to add, 49/49 (100%)
# is the maximum possible coverage under this rule table. expected_intent/
# expected_threat/expected_action are read DIRECTLY off RULES (generated by a
# one-off script, not hand-typed) so they cannot silently drift from it; see
# tests/test_test_cases_rules_consistency.py for the standing guard that catches
# a future manual edit introducing a mismatch instead of "silently picking one."
# 6 of these 49 pairs coincide with an ORIGINAL_TEST_CASES pair (that scenario is
# tested twice, once hand-written with narrative framing, once systematically --
# not a bug). Includes all 7 steady-state (a==b) pairs and all 13 threat=="low"
# (de-escalation) pairs, both of which ORIGINAL_TEST_CASES under-represented.
#
# IN-DISTRIBUTION CAVEAT (AUDIT.md sec J): llm_finetuning/check_training_coverage.py
# confirms all three SFT files (sft_train_v2/final/final_abstain.jsonl) already cover
# 49/49 of these RULES pairs. TEST_CASES (= ORIGINAL_TEST_CASES + this list) is
# therefore a rule-table RECALL battery for every adapter trained on those files
# (v2/v3a/v3a-nomask/v3b), NOT a held-out generalization test -- a 100% score here
# is not evidence of generalization and should not be cited as such. The held-out
# generalization test is llm_finetuning/holdout_shapes.py, not this file.
RULES_COVERAGE_CASES = [
    {"name": "v_shape->v_shape", "formation_a": "v_shape", "formation_b": "v_shape",
     "expected_intent": "surveillance", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "encirclement->encirclement", "formation_a": "encirclement", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "column->column", "formation_a": "column", "formation_b": "column",
     "expected_intent": "patrol", "expected_threat": "low",
     "expected_action": "monitor"},  # dup of ORIGINAL_TEST_CASES "Stable Patrol"
    {"name": "diamond->diamond", "formation_a": "diamond", "formation_b": "diamond",
     "expected_intent": "patrol", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "dispersed->dispersed", "formation_a": "dispersed", "formation_b": "dispersed",
     "expected_intent": "surveillance", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "converging->converging", "formation_a": "converging", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "shield->shield", "formation_a": "shield", "formation_b": "shield",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor"},  # dup of ORIGINAL_TEST_CASES "Defensive Shield"
    {"name": "v_shape->converging", "formation_a": "v_shape", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "encirclement->converging", "formation_a": "encirclement", "formation_b": "converging",
     "expected_intent": "encircle", "expected_threat": "critical",
     "expected_action": "deploy_countermeasure"},
    {"name": "column->converging", "formation_a": "column", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "diamond->converging", "formation_a": "diamond", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "dispersed->converging", "formation_a": "dispersed", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator"},  # dup of ORIGINAL_TEST_CASES "Converging Attack"
    {"name": "shield->converging", "formation_a": "shield", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "v_shape->encirclement", "formation_a": "v_shape", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator"},  # dup of ORIGINAL_TEST_CASES "Encirclement Behavior"
    {"name": "column->encirclement", "formation_a": "column", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "diamond->encirclement", "formation_a": "diamond", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "dispersed->encirclement", "formation_a": "dispersed", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "converging->encirclement", "formation_a": "converging", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "critical",
     "expected_action": "deploy_countermeasure"},
    {"name": "shield->encirclement", "formation_a": "shield", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "encirclement->v_shape", "formation_a": "encirclement", "formation_b": "v_shape",
     "expected_intent": "regroup", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "encirclement->column", "formation_a": "encirclement", "formation_b": "column",
     "expected_intent": "withdraw", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "encirclement->diamond", "formation_a": "encirclement", "formation_b": "diamond",
     "expected_intent": "consolidate", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "encirclement->dispersed", "formation_a": "encirclement", "formation_b": "dispersed",
     "expected_intent": "withdraw", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "encirclement->shield", "formation_a": "encirclement", "formation_b": "shield",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "converging->v_shape", "formation_a": "converging", "formation_b": "v_shape",
     "expected_intent": "regroup", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "converging->column", "formation_a": "converging", "formation_b": "column",
     "expected_intent": "patrol", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "converging->diamond", "formation_a": "converging", "formation_b": "diamond",
     "expected_intent": "consolidate", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "converging->dispersed", "formation_a": "converging", "formation_b": "dispersed",
     "expected_intent": "withdraw", "expected_threat": "low",
     "expected_action": "monitor"},  # dup of ORIGINAL_TEST_CASES "Breaking Contact"
    {"name": "converging->shield", "formation_a": "converging", "formation_b": "shield",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "v_shape->column", "formation_a": "v_shape", "formation_b": "column",
     "expected_intent": "transit", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "v_shape->diamond", "formation_a": "v_shape", "formation_b": "diamond",
     "expected_intent": "defensive_transition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "v_shape->dispersed", "formation_a": "v_shape", "formation_b": "dispersed",
     "expected_intent": "area_search", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "v_shape->shield", "formation_a": "v_shape", "formation_b": "shield",
     "expected_intent": "defensive_transition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "column->v_shape", "formation_a": "column", "formation_b": "v_shape",
     "expected_intent": "attack_preparation", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "column->diamond", "formation_a": "column", "formation_b": "diamond",
     "expected_intent": "consolidate", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "column->dispersed", "formation_a": "column", "formation_b": "dispersed",
     "expected_intent": "area_search", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "column->shield", "formation_a": "column", "formation_b": "shield",
     "expected_intent": "defensive_transition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "diamond->v_shape", "formation_a": "diamond", "formation_b": "v_shape",
     "expected_intent": "reposition", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "diamond->column", "formation_a": "diamond", "formation_b": "column",
     "expected_intent": "transit", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "diamond->dispersed", "formation_a": "diamond", "formation_b": "dispersed",
     "expected_intent": "area_search", "expected_threat": "medium",
     "expected_action": "monitor"},  # dup of ORIGINAL_TEST_CASES "Area Search"
    {"name": "diamond->shield", "formation_a": "diamond", "formation_b": "shield",
     "expected_intent": "defensive_transition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "dispersed->v_shape", "formation_a": "dispersed", "formation_b": "v_shape",
     "expected_intent": "rally", "expected_threat": "high",
     "expected_action": "alert_operator"},
    {"name": "dispersed->column", "formation_a": "dispersed", "formation_b": "column",
     "expected_intent": "transit", "expected_threat": "low",
     "expected_action": "monitor"},
    {"name": "dispersed->diamond", "formation_a": "dispersed", "formation_b": "diamond",
     "expected_intent": "consolidate", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "dispersed->shield", "formation_a": "dispersed", "formation_b": "shield",
     "expected_intent": "defensive_transition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "shield->v_shape", "formation_a": "shield", "formation_b": "v_shape",
     "expected_intent": "reposition", "expected_threat": "medium",
     "expected_action": "increase_surveillance"},
    {"name": "shield->column", "formation_a": "shield", "formation_b": "column",
     "expected_intent": "reposition", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "shield->diamond", "formation_a": "shield", "formation_b": "diamond",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor"},
    {"name": "shield->dispersed", "formation_a": "shield", "formation_b": "dispersed",
     "expected_intent": "surveillance", "expected_threat": "low",
     "expected_action": "monitor"},
]

TEST_CASES = ORIGINAL_TEST_CASES + RULES_COVERAGE_CASES


JUDGE_PROMPT = """\
You are an expert evaluator of UAV swarm tactical analysis systems.
Given the ML tactical context and the LLM assessment below, score each dimension 1-5.

Scoring rubric:
- factual_accuracy    : Does the assessment accurately reflect the ML context? No hallucinations?
- threat_calibration  : Is the threat level reasonable given velocity, formation, stability?
- action_relevance    : Is the recommended_action appropriate?
- reasoning_quality   : Is the reasoning specific and non-generic?
- internal_consistency: Are summary, intent, threat_level, and action mutually consistent?

Return ONLY valid JSON:
{{
  "factual_accuracy": <1-5>,
  "threat_calibration": <1-5>,
  "action_relevance": <1-5>,
  "reasoning_quality": <1-5>,
  "internal_consistency": <1-5>,
  "overall_score": <1.0-5.0>,
  "critique": "<one sentence>"
}}

--- ML TACTICAL CONTEXT ---
{tactical_context}

--- LLM ASSESSMENT ---
{llm_assessment}
"""
