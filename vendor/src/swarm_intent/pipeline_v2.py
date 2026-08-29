"""
AUDIT.md sec AE step 3: the three-layer pipeline step 2's coverage measurement
justifies. Every observation is bucketed (src/swarm_intent/coverage.py) before
any LLM is involved in a DECISION at all:

  Layer 1 (bucket A, ~1.8% of real observations, step 2) -- deterministic.
    threat_level / likely_intent / recommended_action come straight from
    RULES[(from, to)] -- the actual dict, not rules_in_prompt, not any model.
    An LLM IS still called, but ONLY to write situation_summary /
    threat_reasoning / key_indicators / follow_up_watch around a decision it
    is told is already final. Its output is validated field-by-field and the
    three decision fields are overwritten with the RULES values regardless of
    what it returned -- every deviation is logged (see "llm_deviation" in the
    returned detail dict), never silently allowed through.

  Layer 2 (bucket B, ~38.2%) -- guard. No model call. Abstains
    (threat_level/likely_intent = "unknown", schema-legal per sec AA step 3)
    with a machine-generated reason built directly from coverage.py's
    guard_reasons (which of oov_name / dominant_history_contradiction /
    dispersed_converging_ambiguity / low_confidence fired, possibly several).

  Layer 3 (bucket C, ~60.0% -- the majority case) -- the LLM's judgment is
    actually load-bearing here: no RULES key can exist for a multi-hop chain,
    an oscillation, or a terminal-unknown read. Routes to the v3b-fix adapter,
    then applies sec AE step 1's SCOPED prior correction (medium/low near-tie
    only, provably never touches high/critical -- see
    src/swarm_intent/llm/prior_correction.py) using v3b-fix's OWN training
    file's class frequency (data/sft_train_final_abstain_fix.jsonl), the same
    per-adapter-frequency convention every prior correction in this project
    has used.

Two input representations, mirroring composite.py's existing two-factory
convention (make_composite_run_case / make_composite_battery_run_case):

  assess_ctx / make_pipeline_v2_run_case / make_pipeline_v2_battery_run_case
    -- the templated tactical-context TEXT representation (TEST_CASES via
    synth_context(), degradation/holdout batteries' precomputed ctx) that
    every other system in this project's eval is compared on. Buckets via
    coverage.classify_ctx.

  assess_observation -- REAL swarm_intent.stgt.inference.sliding_window_
    inference() output, for actual deployment (not exercised by this
    project's eval batteries, which carry no real STGT inference output --
    included for completeness/genuine production use). Buckets via
    coverage.classify_observation.

Batched variants (make_pipeline_v2_batched_run_case /
make_pipeline_v2_batched_battery_run_case) exist because sec AE step 4's
comparison eval (55-case + degradation battery, n_runs=20 on the volatile
strata) is large enough that unbatched generation is impractical -- reuses
the SAME batching technique sec U validated as equivalent to unbatched
(~2.78x speedup), applied per-bucket (Layer 1's narrator calls batched
together, Layer 3's v3b-fix calls batched together; Layer 2 needs no model
call at all).

RULES (the canonical decision table) lives in llm_finetuning/build_sft_dataset.py,
not under src/ -- imported here via a disclosed, deliberate sys.path exception
(this is the one place in src/swarm_intent that reaches into llm_finetuning/,
specifically to avoid duplicating a hand-curated, domain-expert-edited 49-entry
table into a second copy that could silently drift from the source of truth).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from random import Random
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "llm_finetuning"))

from .inference import build_llm_prompt  # noqa: E402
from .coverage import (  # noqa: E402
    classify_ctx, classify_observation, _collapse_consecutive,
    MAX_WINDOWS_PER_SINGLE_TRANSITION, BUCKET_A, BUCKET_B, BUCKET_C,
)
from .stgt_bridge import UNKNOWN_FORMATION  # noqa: E402
from .config import TRANSITION_CLASS  # noqa: E402
from .llm.client import _parse_llm_response  # noqa: E402
from .llm.prior_correction import (  # noqa: E402
    CANDIDATES, THREAT_KEY_MARKER, find_key_marker_position, score_candidates,
    softmax_over_candidates, scoped_correct,
)

from build_sft_dataset import RULES, synth_context  # noqa: E402  (see module docstring)

LAYER_1_DETERMINISTIC = "layer1_deterministic"
LAYER_2_GUARD = "layer2_guard"
LAYER_3_LLM = "layer3_llm"

# --- Confidence-flag layer (docs/CONFIDENCE_FLAG.md) --------------------------------------
# Additive-only post-processing on Layer 3's existing answer. This is NOT abstention and does
# NOT reroute anything -- Layer 1/2/3 routing (_dispatch/_resolve_batched/classify_ctx/
# classify_observation) below is completely untouched by this section. Reuses the exact
# structural detection rule AUDIT.md sec AN's (reverted) _structural_chain_reason() used --
# the OBSERVED (not ground-truth) known-formation-history length >= 3 structurally means no
# single (from, to) RULES entry can represent the case -- but instead of forcing a Layer-2
# guard response (which sec AN's mandatory false-positive audit found misroutes 51/498,
# 10.2%, of genuinely answerable cases), this only ATTACHES an operator-facing caution field
# to the answer Layer 3 already produced. See docs/CONFIDENCE_FLAG.md for the full writeup,
# including why a soft flag was chosen over sec AN's hard guard.
CONFIDENCE_FLAG_MULTIHOP = "LOW_CONFIDENCE_MULTIHOP"
CONFIDENCE_FLAG_OSCILLATION = "LOW_CONFIDENCE_OSCILLATION"

# Same pattern already used by llm_finetuning/build_abstain_rows.py, degradation.py, and
# corpus_abstention_by_mechanism.py to parse a rendered ctx's own "Formation history: ..."
# line -- reused here, not reimplemented.
_HISTORY_LINE_RE = re.compile(r"Formation history:\s*(.+)")


# Tokens that never count as a real, distinct formation in a rendered history line.
# UNKNOWN_FORMATION ("unknown") is what stgt_bridge.py's real _validate_formation() maps any
# non-BASE_FORMATIONS class (including TRANSITION_CLASS) to before rendering -- so it alone
# would be sufficient for REAL bridge_predictions() ctx text (classify_observation's path,
# and evaluation/seed999_eval_set.json's ctx, confirmed by inspection: "transitioning" never
# appears literally there). But this project's SYNTHETIC ctx renderers (build_sft_dataset.
# synth_context() -- used by Act 3's live demo and TEST_CASES/RULES training pairs --  plus
# degradation.py/holdout_shapes.py/build_abstention_corpus.py) write the literal word
# "transitioning" directly between formation names for an ordinary single transition (e.g.
# "column -> transitioning -> encirclement" for a plain, answerable 2-formation case).
# Filtering only UNKNOWN_FORMATION and not TRANSITION_CLASS would spuriously read every such
# case as a 3-formation chain and false-flag roughly half of all live Act 3 demo cases --
# caught before shipping. Filtering both matches _validate_formation's real-pipeline
# convention and is a no-op on ctx that already went through it.
_NON_FORMATION_TOKENS = {UNKNOWN_FORMATION, TRANSITION_CLASS}


def _known_history_from_ctx(ctx: str) -> list:
    """The same collapse classify_observation() performs on summary['formation_history']
    (filter non-formation tokens, then collapse consecutive duplicates -- sec AN's
    _structural_chain_reason did this from the summary dict directly), but parsed from the
    rendered ctx TEXT's own 'Formation history: ...' line instead. This is what makes it work
    on BOTH assess_ctx's templated-text path (classify_ctx's bucket_info carries no summary
    dict at all) and assess_observation's real bridge_predictions() path alike, since both
    ultimately render this same line (stgt_bridge.py / build_sft_dataset.py)."""
    m = _HISTORY_LINE_RE.search(ctx)
    if not m:
        return []
    raw = [f.strip() for f in m.group(1).split("->")]
    return _collapse_consecutive([f for f in raw if f not in _NON_FORMATION_TOKENS])


# Same line every ctx renderer in this project produces ("Observation window: 0.0s - Xs
# (N windows)", stgt_bridge.py / build_sft_dataset.py) -- reused, not a new convention.
_OBSERVATION_WINDOW_RE = re.compile(r"Observation window:.*\((\d+) windows?\)")


def _n_windows_from_ctx(ctx: str) -> int:
    m = _OBSERVATION_WINDOW_RE.search(ctx)
    return int(m.group(1)) if m else 0


def compute_confidence_flag(ctx: str) -> Optional[str]:
    """None, or one of CONFIDENCE_FLAG_MULTIHOP/CONFIDENCE_FLAG_OSCILLATION. Fires when the
    OBSERVED (not ground-truth) formation history structurally spans >=3 distinct formations
    -- the same signal AUDIT.md sec AN measured as a hard guard, reused verbatim here as a
    soft signal instead -- AND the observation ran for MORE sliding windows than any single
    2-formation transition could ever physically produce (coverage.
    MAX_WINDOWS_PER_SINGLE_TRANSITION, sec AK/AL's own analytically-derived bound from the
    generator's own timing ranges -- reused here in the complementary direction to
    coverage.py's own "observation_too_long_for_reduced_state_count" guard, which uses the
    same constant to flag the opposite failure: a state plausibly MISSING, not phantom).

    Refinement rationale (docs/CONFIDENCE_FLAG.md dated erratum): measured on a disjoint dev
    split (evaluation/confidence_flag_dev_split.json, seed=13579, never the locked eval
    file), the original length-only rule's 61 dev-split false positives all occurred on
    observations of <=8 windows (max 8), while every true multi-hop/oscillation case had
    >=14 windows -- a clean, wide, unforced gap, not a razor's-edge threshold. Adding this
    gate removed 100% of dev-split false positives at zero recall cost (the existing 12
    dev-split misses are a separate, independent failure mode -- the STGT model sometimes
    never classifies one true formation leg at all, regardless of observation length -- and
    are untouched by this condition either way).

    KNOWN LIMITATION, not fixed by this gate: every SYNTHETIC ctx renderer in this project
    (build_sft_dataset.synth_context() -- used by Act 3's live demo -- plus degradation.py/
    holdout_shapes.py/build_abstention_corpus.py) hardcodes a fixed, decorative "(9
    overlapping windows)" string regardless of actual content -- this gate is only
    meaningful on ctx text that went through the REAL stgt_bridge.bridge_predictions()
    rendering (classify_observation's path; evaluation/seed999_eval_set.json's ctx).
    Harmless for Act 3 today (its synth_context() path can only ever produce a 2-formation
    chain in the first place, so the base length>=3 check already never fires there), but a
    future caller wiring this onto a synthetic-ctx battery would see this gate silently
    veto every case -- disclosed here and in docs/CONFIDENCE_FLAG.md, not fixed (fixing the
    synthetic renderers' placeholder text is out of this task's scope: additive-only,
    no changes to shared corpus-generation code)."""
    known_history = _known_history_from_ctx(ctx)
    if len(known_history) < 3:
        return None
    if _n_windows_from_ctx(ctx) <= MAX_WINDOWS_PER_SINGLE_TRANSITION:
        return None
    return CONFIDENCE_FLAG_OSCILLATION if known_history[0] == known_history[-1] else CONFIDENCE_FLAG_MULTIHOP


def attach_confidence_flag(assessment: dict, ctx: str, layer: str, enabled: bool = True) -> dict:
    """Additive post-processing step, called explicitly by a caller AFTER assess_ctx()/
    assess_observation() return -- assess_ctx/assess_observation/_dispatch/_resolve_batched
    themselves are not modified by this feature at all, so every existing caller that does not
    call this function is completely unaffected. Attaches confidence_flag to Layer 3's answer;
    no other field is ever touched, and the model's threat_level/likely_intent/
    recommended_action are never altered or rerouted.

    When enabled=False, or layer is not LAYER_3_LLM, returns `assessment` completely
    unmodified -- the same object, no new key added -- so disabled/non-Layer-3 output is
    byte-identical to output from before this feature existed. When enabled=True and
    layer==LAYER_3_LLM, returns a NEW dict (a shallow copy; the input `assessment` is never
    mutated in place) with confidence_flag set to None or a CONFIDENCE_FLAG_* value."""
    if not enabled or layer != LAYER_3_LLM:
        return assessment
    flagged = dict(assessment)
    flagged["confidence_flag"] = compute_confidence_flag(ctx)
    return flagged

V3B_FIX_TRAIN_FILE = "data/sft_train_final_abstain_fix.jsonl"

GUARD_REASON_TEXT = {
    "oov_name": "an out-of-vocabulary formation name appeared in the observation",
    "dominant_history_contradiction": "the dominant formation could not be determined "
                                      "unambiguously (tied window counts / contradicts the history line)",
    "dispersed_converging_ambiguity": "dispersed and converging classifier probabilities "
                                      "were within the ambiguity margin on at least one window",
    "low_confidence": "every available window fell below the 0.6 confidence threshold",
}

LAYER1_NARRATOR_SYSTEM_PROMPT = (
    "You are a tactical AI analyst assisting a counter-UAV operator. The threat_level, "
    "likely_intent, and recommended_action for this observation have ALREADY been "
    "determined by the canonical rule engine and are given to you below -- you must NOT "
    "question, second-guess, or change them under any circumstance. Your only job is to "
    "write the supporting narrative fields (situation_summary, threat_reasoning, "
    "key_indicators, follow_up_watch) that explain and support the given decision. Echo "
    "threat_level / likely_intent / recommended_action back EXACTLY as given."
)


def _preds_from_key_windows(key_windows):
    return [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw.get("formation"),
            "centroid_velocity": kw.get("velocity"), "approach_rate": kw.get("approach"),
            "formation_stability": kw.get("stability"), "formation_confidence": kw.get("confidence"),
            "role_differentiation": kw.get("role_differentiation", False),
            "transition_from": kw.get("from"), "transition_to": kw.get("to")} for kw in key_windows]


def _train_class_freq(path) -> dict:
    from collections import Counter
    counts = Counter()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            target = json.loads(row["messages"][1]["content"])
            t = target.get("threat_level")
            if t in CANDIDATES:
                counts[t] += 1
    total = sum(counts.values())
    return {c: counts[c] / total for c in CANDIDATES}


def default_class_freq(repo_root: Optional[Path] = None) -> dict:
    """v3b-fix's OWN training file's class frequency -- the same per-adapter-
    frequency convention logit_inspection.py's SYSTEMS dict uses for v3a/v3b/v3d."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    return _train_class_freq(root / V3B_FIX_TRAIN_FILE)


def _finalize_layer1(llm_out, threat: str, intent: str, action: str):
    """Validate the narrator LLM's output against the already-decided fields and
    overwrite regardless of what it returned. Returns (assessment, deviation) --
    deviation is a dict of {field: what the LLM said instead}, empty if it
    complied. Always overwritten -- the LLM is never allowed to alter the
    decision, deviation is logged, not merely corrected silently."""
    if not isinstance(llm_out, dict):
        llm_out = {}
    deviation = {}
    for field, val in (("threat_level", threat), ("likely_intent", intent),
                       ("recommended_action", action)):
        got = llm_out.get(field)
        if got != val:
            deviation[field] = got
    assessment = dict(llm_out)
    assessment["threat_level"] = threat
    assessment["likely_intent"] = intent
    assessment["recommended_action"] = action
    assessment.setdefault("confidence_in_assessment", "high")
    assessment.setdefault("situation_summary", "Rule-table lookup for the observed transition.")
    assessment.setdefault("threat_reasoning", f"Rule-table lookup: {threat} threat.")
    assessment.setdefault("key_indicators", ["rule-table lookup"])
    assessment.setdefault("follow_up_watch", "Monitor for the next formation change.")
    return assessment, deviation


def _layer1_prompt(ctx, key_windows, threat, intent, action):
    preds = _preds_from_key_windows(key_windows)
    prompt = build_llm_prompt(preds, ctx, {})
    decision_note = (f'\n\nDECISION (already determined, echo back exactly, do not alter): '
                     f'threat_level="{threat}", likely_intent="{intent}", '
                     f'recommended_action="{action}".')
    return prompt + decision_note


def _layer1_narrate(rules_narrator_client, ctx, key_windows, threat, intent, action):
    prompt = _layer1_prompt(ctx, key_windows, threat, intent, action)
    llm_out = rules_narrator_client.complete(prompt, system_prompt=LAYER1_NARRATOR_SYSTEM_PROMPT)
    return _finalize_layer1(llm_out, threat, intent, action)


def _layer2_abstain(guard_reasons: list) -> dict:
    reason_text = "; ".join(GUARD_REASON_TEXT.get(r, r) for r in guard_reasons) or "unspecified guard condition"
    return {
        "situation_summary": "Unable to confidently resolve this observation to a single rule-table entry.",
        "threat_level": "unknown",
        "threat_reasoning": f"Guarded abstention: {reason_text}.",
        "likely_intent": "unknown",
        "recommended_action": "monitor",
        "confidence_in_assessment": "low",
        "key_indicators": list(guard_reasons),
        "follow_up_watch": "Re-acquire a higher-confidence, unambiguous observation window.",
    }


def _rescore_and_correct(finetuned_client, ctx, key_windows, completion_text: str, class_freq: dict) -> dict:
    """Teacher-force-rescore the ALREADY-GENERATED completion text (from either
    the unbatched .generate() path or a batched .generate_batch() pass -- this
    function re-tokenizes `completion_text` fresh rather than requiring the
    original generation's raw token ids, so it works identically after either
    path) and apply the scoped correction. Returns scoped_correct()'s dict
    unchanged, or {"applied": False, "reason": "..."} if the threat_level
    marker isn't found in the completion at all."""
    import torch
    tok, model = finetuned_client.tok, finetuned_client.model
    preds = _preds_from_key_windows(key_windows)
    prompt = build_llm_prompt(preds, ctx, {})
    messages = ([{"role": "system", "content": finetuned_client.system_prompt}]
               if finetuned_client.system_prompt else [])
    messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
    gen_ids = torch.tensor(tok(completion_text, add_special_tokens=False)["input_ids"], device=model.device)
    pos = find_key_marker_position(tok, gen_ids, THREAT_KEY_MARKER)
    if pos is None:
        return {"applied": False, "corrected_argmax": None, "corrected_p": {},
                "reason": "threat_level marker not found in completion"}
    prefix_ids = torch.cat([enc["input_ids"][0], gen_ids[:pos]]).unsqueeze(0)
    logprobs = score_candidates(model, tok, prefix_ids)
    raw_p = softmax_over_candidates(logprobs)
    return scoped_correct(raw_p, class_freq)


def _layer3_llm(finetuned_client, ctx, key_windows, class_freq):
    preds = _preds_from_key_windows(key_windows)
    prompt = build_llm_prompt(preds, ctx, {})
    text = finetuned_client.generate(prompt)
    assessment = _parse_llm_response(text)
    correction = _rescore_and_correct(finetuned_client, ctx, key_windows, text, class_freq)
    if correction["applied"] and isinstance(assessment, dict) and "threat_level" in assessment:
        assessment["threat_level"] = correction["corrected_argmax"]
    return assessment, correction


def _dispatch(bucket_info, ctx, key_windows, rules_narrator_client, finetuned_client, class_freq):
    bucket = bucket_info["bucket"]
    if bucket == BUCKET_A:
        form_a, form_b = bucket_info["rules_key"]
        threat, intent, action = RULES[(form_a, form_b)]
        assessment, deviation = _layer1_narrate(rules_narrator_client, ctx, key_windows, threat, intent, action)
        return assessment, LAYER_1_DETERMINISTIC, {"rules_key": [form_a, form_b], "llm_deviation": deviation}
    if bucket == BUCKET_B:
        assessment = _layer2_abstain(bucket_info["guard_reasons"])
        return assessment, LAYER_2_GUARD, {"guard_reasons": bucket_info["guard_reasons"]}
    assessment, correction = _layer3_llm(finetuned_client, ctx, key_windows, class_freq)
    return assessment, LAYER_3_LLM, {"subtype": bucket_info["subtype"], "correction": correction}


def assess_ctx(rules_narrator_client, finetuned_client, ctx: str, key_windows: list, class_freq: dict):
    """ctx/key_windows: the templated tactical-context representation (synth_context()
    / degradation.py's precomputed cases). Returns (assessment, layer, detail)."""
    bucket_info = classify_ctx(ctx, key_windows)
    return _dispatch(bucket_info, ctx, key_windows, rules_narrator_client, finetuned_client, class_freq)


def assess_observation(rules_narrator_client, finetuned_client, predictions: list, class_freq: dict,
                       calibrator=None, robust: bool = False, robust_threshold: float = None):
    """predictions: REAL swarm_intent.stgt.inference.sliding_window_inference() output.
    robust (AUDIT.md sec AG): use stgt_bridge's majority-based robust reduction
    instead of the original unanimity reduction when bucketing -- False (default)
    is the sec AE/AF-measured behaviour; True is sec AG's fix. Returns
    (assessment, layer, detail)."""
    kwargs = {"robust": robust}
    if robust_threshold is not None:
        kwargs["robust_threshold"] = robust_threshold
    bucket_info = classify_observation(predictions, calibrator, **kwargs)
    return _dispatch(bucket_info, bucket_info["context_text"], bucket_info["key_windows"],
                     rules_narrator_client, finetuned_client, class_freq)


# --- run_case factories (evaluate_llm-compatible: run_case(case) -> (assessment, ctx)) ---

def make_pipeline_v2_run_case(rules_narrator_client, finetuned_client, class_freq: dict,
                              layer_log: dict, seed: int = 0):
    """For TEST_CASES -- ctx built via synth_context(), same shared-rng protocol as
    every other headline-eval run_case factory in this project."""
    rng = Random(seed)

    def run_case(case):
        ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
        assessment, layer, detail = assess_ctx(rules_narrator_client, finetuned_client,
                                               ctx, key_windows, class_freq)
        layer_log[case["name"]].append({"layer": layer, "detail": detail})
        return assessment, ctx

    return run_case


def make_pipeline_v2_battery_run_case(rules_narrator_client, finetuned_client, class_freq: dict,
                                      layer_log: dict):
    """For degradation.py / holdout_shapes.py battery cases -- ctx/key_windows already
    precomputed on the case dict, same convention as degradation.make_llm_battery_run_case
    / composite.make_composite_battery_run_case."""
    def run_case(case):
        assessment, layer, detail = assess_ctx(rules_narrator_client, finetuned_client,
                                               case["ctx"], case["key_windows"], class_freq)
        layer_log[case["name"]].append({"layer": layer, "detail": detail})
        return assessment, case["ctx"]

    return run_case


# --- batched variants (sec AE step 4: n_runs=20 x 2 batteries x 5 systems is too large
# to run unbatched -- reuses sec U's validated-equivalent batching technique, applied
# per-bucket since different buckets need different model calls, or none at all) ---

def _resolve_batched(items: list, rules_narrator_client, finetuned_client, class_freq: dict, batch_size: int):
    """Mutates each item in `items` (each a dict with ctx/key_windows/bucket_info) in
    place, adding "assessment"/"layer"/"detail". Bucket B needs no model call at all;
    bucket A batches the narrator client's calls together (one system_prompt for the
    whole batch, since every Layer-1 call uses the same LAYER1_NARRATOR_SYSTEM_PROMPT);
    bucket C batches the v3b-fix generations together, then rescoring/correction runs
    sequentially afterward (cheap: teacher-forced, non-autoregressive, ~4 forward
    passes/item -- generate_batch's plain-text return doesn't expose the raw
    generation token ids the correction needs, so this is a second pass over the
    already-generated text, not a second generation)."""
    a_idx, c_idx = [], []
    for i, item in enumerate(items):
        bucket = item["bucket_info"]["bucket"]
        if bucket == BUCKET_A:
            a_idx.append(i)
        elif bucket == BUCKET_C:
            c_idx.append(i)
        else:
            item["assessment"] = _layer2_abstain(item["bucket_info"]["guard_reasons"])
            item["layer"] = LAYER_2_GUARD
            item["detail"] = {"guard_reasons": item["bucket_info"]["guard_reasons"]}

    if a_idx:
        prompts, decisions = [], []
        for i in a_idx:
            item = items[i]
            form_a, form_b = item["bucket_info"]["rules_key"]
            threat, intent, action = RULES[(form_a, form_b)]
            prompts.append(_layer1_prompt(item["ctx"], item["key_windows"], threat, intent, action))
            decisions.append((form_a, form_b, threat, intent, action))
        raw_outs = rules_narrator_client.complete_batch(prompts, batch_size=batch_size,
                                                         system_prompt=LAYER1_NARRATOR_SYSTEM_PROMPT)
        for i, llm_out, (form_a, form_b, threat, intent, action) in zip(a_idx, raw_outs, decisions):
            assessment, deviation = _finalize_layer1(llm_out, threat, intent, action)
            items[i]["assessment"] = assessment
            items[i]["layer"] = LAYER_1_DETERMINISTIC
            items[i]["detail"] = {"rules_key": [form_a, form_b], "llm_deviation": deviation}

    if c_idx:
        prompts = []
        for i in c_idx:
            item = items[i]
            prompts.append(build_llm_prompt(_preds_from_key_windows(item["key_windows"]), item["ctx"], {}))
        texts = finetuned_client.generate_batch(prompts, batch_size=batch_size)
        for i, text in zip(c_idx, texts):
            item = items[i]
            assessment = _parse_llm_response(text)
            correction = _rescore_and_correct(finetuned_client, item["ctx"], item["key_windows"],
                                              text, class_freq)
            if correction["applied"] and isinstance(assessment, dict) and "threat_level" in assessment:
                assessment["threat_level"] = correction["corrected_argmax"]
            item["assessment"] = assessment
            item["layer"] = LAYER_3_LLM
            item["detail"] = {"subtype": item["bucket_info"]["subtype"], "correction": correction}


def make_pipeline_v2_batched_run_case(rules_narrator_client, finetuned_client, class_freq: dict,
                                      layer_log: dict, test_cases: list, n_runs: int,
                                      batch_size: int = 8, seed: int = 0):
    """Precomputes bucket classification for every (case, run) pair up front, in the
    exact case-major/run-minor order evaluate_llm calls run_case (same shared Random(seed)
    as the unbatched path, so ctx is byte-identical), then resolves every item in ONE
    batched pass per bucket. Drop-in replacement for make_pipeline_v2_run_case."""
    rng = Random(seed)
    items = []
    for case in test_cases:
        for _ in range(n_runs):
            ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
            items.append({"case": case, "ctx": ctx, "key_windows": key_windows,
                          "bucket_info": classify_ctx(ctx, key_windows)})

    _resolve_batched(items, rules_narrator_client, finetuned_client, class_freq, batch_size)

    call_idx = [0]

    def run_case(case):
        item = items[call_idx[0]]
        call_idx[0] += 1
        layer_log[case["name"]].append({"layer": item["layer"], "detail": item["detail"]})
        return item["assessment"], item["ctx"]

    return run_case


def make_pipeline_v2_batched_battery_run_case(rules_narrator_client, finetuned_client, class_freq: dict,
                                              layer_log: dict, battery_cases: list, n_runs: int,
                                              batch_size: int = 8):
    """Same as make_pipeline_v2_batched_run_case but for precomputed-ctx battery cases
    (ctx/key_windows are fixed per case -- only the LLM call itself is re-sampled
    n_runs times, same case-major/run-minor order)."""
    items = []
    for case in battery_cases:
        bucket_info = classify_ctx(case["ctx"], case["key_windows"])
        for _ in range(n_runs):
            items.append({"case": case, "ctx": case["ctx"], "key_windows": case["key_windows"],
                          "bucket_info": bucket_info})

    _resolve_batched(items, rules_narrator_client, finetuned_client, class_freq, batch_size)

    call_idx = [0]

    def run_case(case):
        item = items[call_idx[0]]
        call_idx[0] += 1
        layer_log[case["name"]].append({"layer": item["layer"], "detail": item["detail"]})
        return item["assessment"], item["ctx"]

    return run_case
