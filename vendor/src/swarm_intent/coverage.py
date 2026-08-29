"""
Bucket classification (AUDIT.md sec AE): is a swarm observation something a
plain (from, to) -> RULES dict lookup can resolve on its own (A, RESOLVABLE),
something a dict COULD answer but shouldn't be trusted to without a hedge (B,
GUARDABLE), or something no 2-tuple lookup can ever answer regardless of
confidence (C, UNRESOLVABLE)? This is the coverage question behind "why not
just ship the dict" -- pipeline_v2.py routes A to RULES directly, B to a
machine-generated abstention, and C to the fine-tuned LLM layer.

Two input shapes, two entry points, same bucket vocabulary:

  classify_observation(predictions) -- REAL swarm_intent.stgt.inference.
    sliding_window_inference() output (rich per-window dicts, including
    class_probabilities). Built on stgt_bridge.bridge_predictions(), which
    already does the OOV-sentinel / oscillation-preserving / dispersed-
    converging-ambiguity work this module needs -- reused, not reimplemented.
    This is the path used to MEASURE real-world coverage (measure_coverage.py).

  classify_ctx(ctx, key_windows) -- the templated tactical-context TEXT
    representation used by TEST_CASES (build_sft_dataset.synth_context) and
    the degradation/holdout batteries, which have no real per-window
    class_probabilities (synth_context never simulates one) -- the dispersed/
    converging ambiguity guard structurally can never fire on this path,
    disclosed, not a bug. Reuses the same (from, to) extraction regex
    llm_finetuning/baselines.py's composite router already keys its own
    routing decision on, so pipeline_v2's bucket boundary is directly
    comparable to what the existing composite already does. This is the path
    used to EVALUATE pipeline_v2 against the existing 55-case/degradation
    batteries, which carry no real STGT inference output at all.
"""
from __future__ import annotations

import re
from itertools import groupby

from .config import BASE_FORMATIONS
from .stgt_bridge import bridge_predictions, UNKNOWN_FORMATION, DEFAULT_ROBUST_THRESHOLD

BUCKET_A = "A"  # resolvable
BUCKET_B = "B"  # guardable
BUCKET_C = "C"  # unresolvable

# AUDIT.md sec AK/AL: how many strided windows a SINGLE formation-hold-or-transition
# segment can ever produce, at window_size=50/stride=10 (the only configuration used
# anywhere in this project -- every sliding_window_inference call site passes these same
# two values). Derived from src/swarm_intent/data.py's LEAD_IN_RANGE=(30,50)/
# BLEND_DURATION_RANGE=(10,25)/MIN_DWELL_RANGE=(40,60): a single TRANSITION segment's raw
# length is capped at their maxima, 49+24+59=132 timesteps; a single STEADY-STATE segment
# is capped lower, at 100. sliding_window_inference((132-50)//10)+1 = 9 windows is the
# true maximum either way -- tests/test_coverage.py::TestMaxWindowsConstantNotDrifted
# imports those three ranges directly and asserts this constant is still their correct
# derivation, so a future change to data.py's ranges fails loudly here instead of silently
# invalidating this guard (coverage.py itself does not import from data.py, to keep this
# module's own import surface free of data.py's sklearn dependency).
#
# sec AK found 9 real cases (seq 105/291/292/314/385/520/574/611/982, n_windows in
# {15,16,17,26}, all well past 9) where the classifier's own per-window argmax never once
# named a real 3rd formation the true generator chain actually had -- not a low-confidence
# or ambiguous read on any single window, just silently absent from every window's top
# classes. No existing guard is designed to catch "a state was dropped with no local
# evidence of it" -- this one catches the STRUCTURAL shape instead: more windows were
# observed than a genuine <=2-formation sequence could physically produce, so a missed
# state is likely, regardless of what any individual window's own probabilities show.
#
# Known, disclosed limitation (not fixed by this guard): this bound is only exact for
# THIS project's synthetic generation regime. A real deployment stream with a genuinely
# long-held single steady formation (no synthetic segment-length cap) would trip this
# guard too, forcing a Layer-2 hedge on a case that is actually resolvable. Accepted here
# because every case in this project's eval/training data is drawn from the same capped
# regime (confirmed zero false positives on the full 1000-trajectory locked set, sec AL) --
# revisit if/when this pipeline consumes an uncapped real-world stream.
MAX_WINDOWS_PER_SINGLE_TRANSITION = 9


def _collapse_consecutive(seq):
    return [k for k, _ in groupby(seq)]


def classify_observation(predictions: list[dict], calibrator=None,
                         max_key_windows: int = 10, robust: bool = False,
                         robust_threshold: float = DEFAULT_ROBUST_THRESHOLD) -> dict:
    """predictions: swarm_intent.stgt.inference.sliding_window_inference() output
    (or hand-built dicts of the same shape). Returns a dict with keys: bucket
    (A/B/C), subtype (C only, else None), rules_key ((from,to) tuple or None),
    guard_reasons (list[str], B only), context_text, summary, key_windows --
    the last three are bridge_predictions()'s own return values, passed through
    so a caller doesn't have to call it twice.

    robust: False (default) reproduces the original behaviour exactly.
    True (AUDIT.md sec AG) attempts majority-based robust reduction in
    bridge_predictions first; when it recovers a pair, the oov_name and
    dominant_history_contradiction guards below are suppressed for THAT case
    (the robust recovery's own threshold check already is the "is this
    trustworthy" test; re-applying the raw-unknown-count/tie guards on top
    would just re-throw away exactly what robust reduction was built to
    recover). dispersed_converging_ambiguity and low_confidence remain
    UNCONDITIONAL either way -- per this module's design and stgt_bridge.py's
    docstring point 4/step 2d, that guard is a real upstream defect, not
    reduction-logic brittleness, and is deliberately not touched by this flag."""
    context_text, summary, key_windows = bridge_predictions(
        predictions, calibrator, max_key_windows, robust=robust, robust_threshold=robust_threshold)
    robust_recovered = bool((summary.get("robust_reduction") or {}).get("recovered"))

    def _c(subtype):
        return {"bucket": BUCKET_C, "subtype": subtype, "rules_key": None, "guard_reasons": [],
                "context_text": context_text, "summary": summary, "key_windows": key_windows,
                "robust_recovery": None}

    if summary.get("abstain"):
        return _c("all_unknown")

    history = summary["formation_history"]  # consecutive-collapsed, includes UNKNOWN_FORMATION groups
    if history[-1] == UNKNOWN_FORMATION:
        # the swarm's CURRENT state is unclassifiable (STGT's own "transitioning"
        # class, or any non-BASE_FORMATIONS string) -- covers the degradation
        # battery's terminal_transitioning axis and any real trailing OOV read.
        return _c("terminal_unknown")

    known_history = _collapse_consecutive([f for f in history if f != UNKNOWN_FORMATION])

    if len(known_history) >= 3:
        return _c("oscillation" if known_history[0] == known_history[-1] else "multi_hop")

    key = ((known_history[0], known_history[0]) if len(known_history) == 1
           else (known_history[0], known_history[-1]))
    if key[0] not in BASE_FORMATIONS or key[1] not in BASE_FORMATIONS:
        # defensive/unreachable: _validate_formation (stgt_bridge.py) already maps
        # anything outside BASE_FORMATIONS to UNKNOWN_FORMATION before this point.
        return _c("no_rules_key")

    guard_reasons = []
    # 2026-08-09 fix (docs/CEILING.md, V5 Phase 0 guard audit): "oov_name" claims to
    # flag a genuinely out-of-vocabulary formation name -- a real data-integrity
    # concern -- but n_unknown_windows also counts the classifier's own valid
    # "transitioning" class (a normal, expected read for a blend/transient window,
    # not a vocabulary problem). Audited: 69% spurious when this was the sole
    # blocker. n_genuinely_oov_windows (stgt_bridge.py) excludes "transitioning",
    # so this now only fires on what the guard's name actually claims.
    if summary["n_genuinely_oov_windows"] > 0 and not robust_recovered:
        guard_reasons.append("oov_name")
    if len(known_history) == 2 and not robust_recovered:
        # 2026-08-09 fix (docs/CEILING.md, V5 Phase 0 guard audit): this guard
        # claims to test whether summary["dominant_formation"] actually
        # CONTRADICTS the derived (from,to) key -- a genuine reason to distrust
        # which formation is "dominant". The previous check (a raw window-count
        # TIE between key[0] and key[1]) tested something unrelated: `key` is
        # derived from TEMPORAL ORDER (the first/last distinct formation in
        # formation_history), never from window counts, so a count tie says
        # nothing about whether `key` itself is trustworthy -- a clean, obviously
        # -correct 2/2 split ties exactly as easily as a genuinely uncertain one.
        # Audited: 100% spurious when this was the sole blocker (n=4). Fixed to
        # test the actual claim: does dominant_formation differ from BOTH members
        # of key? By this point known_history already has <=2 distinct real
        # formations (the len(known_history)>=3 check above already routed 3+ to
        # bucket C), and summary["dominant_formation"] is the mode over the SAME
        # valid-formations set known_history was built from -- so it is PROVABLY
        # always one of key[0]/key[1] here. This condition is therefore correctly
        # unreachable on real input: kept as a defensive check (matches the
        # "no_rules_key" pattern above), not deleted, in case a future change to
        # the upstream invariants makes it reachable again.
        if summary["dominant_formation"] not in (key[0], key[1]):
            guard_reasons.append("dominant_history_contradiction")
    if summary["n_ambiguous_dispersed_converging_windows"] > 0:
        guard_reasons.append("dispersed_converging_ambiguity")
    if predictions and all(p["formation_confidence"] < 0.6 for p in predictions):
        guard_reasons.append("low_confidence")
    if len(predictions) > MAX_WINDOWS_PER_SINGLE_TRANSITION:
        # AUDIT.md sec AK/AL -- see MAX_WINDOWS_PER_SINGLE_TRANSITION's own docstring.
        # UNCONDITIONAL (not suppressed by robust_recovered): robust reduction's majority
        # vote answers "is this noisy per-window disagreement trustworthy," a different
        # question from "did the observation run long enough that a state was plausibly
        # missed entirely" -- a confident robust recovery does not address this at all.
        guard_reasons.append("observation_too_long_for_reduced_state_count")

    bucket = BUCKET_B if guard_reasons else BUCKET_A
    return {"bucket": bucket, "subtype": None, "rules_key": key, "guard_reasons": guard_reasons,
            "context_text": context_text, "summary": summary, "key_windows": key_windows,
            "robust_recovery": summary.get("robust_reduction")}


# --- ctx-text representation (TEST_CASES / degradation / holdout batteries) ---

# Matches synth_context()'s and degradation.py's _render_lines()'s exact phrasing
# -- the SAME regex llm_finetuning/baselines.py's _extract_pair uses for the
# composite router, duplicated here (not imported) so this module has no
# dependency on llm_finetuning/ (see pipeline_v2.py for the one place that does).
_TRANSITION_RE = re.compile(r"Transition detected at t=[\d.]+s:\s*(\w+)\s*->\s*(\w+)")
_DOMINANT_RE = re.compile(r"Dominant formation:\s*(\w+)")
_NO_TRANSITION_RE = re.compile(r"No formation transitions detected\.")


def _extract_pair_from_ctx(ctx: str):
    m = _TRANSITION_RE.search(ctx)
    if m:
        return m.group(1), m.group(2)
    if _NO_TRANSITION_RE.search(ctx):
        dm = _DOMINANT_RE.search(ctx)
        if dm:
            return dm.group(1), dm.group(1)
    return None


def classify_ctx(ctx: str, key_windows: list[dict]) -> dict:
    """ctx: a tactical-context string in synth_context()/_render_lines() format.
    key_windows: the case's precomputed key_windows list (each with a
    "confidence" field). Returns the same {bucket, subtype, rules_key,
    guard_reasons} shape as classify_observation (no context_text/summary/
    key_windows passthrough -- caller already has ctx/key_windows)."""
    pair = _extract_pair_from_ctx(ctx)
    if pair is None:
        # "Multiple formation changes detected..." (multi_hop severity>=3,
        # terminal_transitioning) -- no clean 2-tuple to extract at all.
        return {"bucket": BUCKET_C, "subtype": "no_extractable_pair", "rules_key": None,
                "guard_reasons": []}

    guard_reasons = []
    if pair[0] not in BASE_FORMATIONS or pair[1] not in BASE_FORMATIONS:
        guard_reasons.append("oov_name")

    dm = _DOMINANT_RE.search(ctx)
    if dm and dm.group(1) not in pair:
        guard_reasons.append("dominant_history_contradiction")

    if key_windows and all(kw.get("confidence", 1.0) < 0.6 for kw in key_windows):
        guard_reasons.append("low_confidence")

    # dispersed/converging ambiguity has NO signal in this text representation
    # (synth_context() never simulates per-window class_probabilities) -- can
    # never fire here. See module docstring.

    bucket = BUCKET_B if guard_reasons else BUCKET_A
    return {"bucket": bucket, "subtype": None, "rules_key": pair, "guard_reasons": guard_reasons}
