"""
Integration shim: converts the vendored, read-only teammate's STGT
(``stgt/inference.py::sliding_window_inference``) prediction list into OUR
tactical context (``context_spec`` vocabulary + ``calibration`` thresholds) --
NOT upstream's ``build_tactical_context``/``build_llm_prompt`` (see
``stgt/README.md`` for why those weren't vendored).

Four correctness issues this module exists to fix (AUDIT.md sec V, step 4):

1. ``predict_v2``'s ``transition_from``/``transition_to`` fields are the
   classifier's top-2 probability RANKS for a single window (only populated
   when that window's OWN predicted class is "transitioning"), not a temporal
   "what did the swarm change from/to over time" signal. This module NEVER
   reads those two fields -- transitions are derived purely from consecutive
   ``formation_type`` values across the window sequence, exactly like this
   repo's own ``inference.py::build_tactical_context`` already does for its
   ``transitions_detected`` list (that part was already correct; the bug this
   guards against is a hypothetical future caller keying ``RULES``
   (``llm_finetuning/build_sft_dataset.py``) on the per-window fields instead).

2. Upstream's ``build_tactical_context`` (not vendored) dedupes formation
   history with ``dict.fromkeys``, which collapses ANY repeated value anywhere
   in the sequence, not just consecutive runs -- an oscillation like
   ``dispersed -> encirclement -> dispersed`` collapses to
   ``dispersed -> encirclement``, silently erasing the return trip. This
   module collapses only consecutive repeats (``itertools.groupby``), which
   preserves oscillation while still not spamming one entry per window.

3. ``RULES`` (``build_sft_dataset.py``) is keyed EXCLUSIVELY on
   ``BASE_FORMATIONS x BASE_FORMATIONS`` (49 entries) -- it has no entry for
   the classifier's own "transitioning" class, nor for any other string. A
   window whose ``formation_type`` isn't a member of ``BASE_FORMATIONS`` is
   therefore un-lookupable and must not silently flow into the transition /
   dominant-formation computation as if it were a real geometry. Such windows
   are replaced with the sentinel ``"unknown"`` (matching
   ``OUTPUT_SCHEMA``'s own ``likely_intent: ... / unknown`` abstention word),
   excluded from transition/dominant-formation logic, and if EVERY window in
   the batch is unknown the whole context abstains (``summary["abstain"] =
   True``) rather than fabricating a confident-looking narrative from nothing.

4. ``dispersed`` and ``converging`` share IDENTICAL base geometry in
   ``data_gen.py`` (same ``rng.uniform`` branch) -- when their classifier
   probabilities are close, the predicted label is close to a coin flip. This
   matters because ``converging -> encirclement`` is one of the more severe
   ``RULES`` entries; a window flagged ambiguous here is a place a downstream
   assessment should hedge, not escalate confidently on a near-50/50 read.

Also caps ``key_windows`` at a fixed maximum (long streams over many strides
would otherwise grow the prompt unboundedly) and never emits a character
absent from every training prompt (e.g. upstream's own ``build_tactical_context``
emits a literal "warning" glyph the fine-tuned models have never seen).

ROBUST REDUCTION (AUDIT.md sec AG, ``robust=True``, default ``False``): sec AF's
real-STGT-output eval found the UNANIMITY reduction above throws away 240/249
(96.4%) of real sequences whose generator ground truth IS a clean (a,b) pair --
sec AG step 1 diagnosed why: 25.7% end in a trailing "transitioning" run that
breaks unanimity even though every earlier window was clean (a windowing
artefact, sec AF step 4), 22.5% are all-transitioning with no unanimous window
at all, and 46.2% (the largest single cause) are guarded by the dispersed/
converging ambiguity check above -- a real upstream defect, not brittleness,
deliberately NOT touched here. ``robust=True`` fixes the first two causes only:
  a. strip leading/trailing "transitioning" runs before reducing -- a window
     ending mid-transition still has a determinable start formation.
  b. reduce by MAJORITY, not unanimity: the modal NON-transitioning formation
     in the first half and the second half of what's left after (a); if they
     differ and each holds a plurality >= ``robust_threshold`` (of that half's
     FULL length, so a half dominated by noise still fails), that is the
     (a,b) pair.
  c. if every window is transitioning even after (a) (nothing left to split),
     aggregate ``class_probabilities`` across all windows and take the top-2
     non-transitioning classes; if well separated (margin >=
     ``ALL_UNKNOWN_FALLBACK_MARGIN``), treat as a resolvable steady state,
     flagged low-confidence.
  d. the dispersed/converging ambiguity guard is UNCONDITIONAL -- it still
     applies to a robustly-recovered pair exactly as it does to a unanimous
     one, per this module's docstring point 4. Sec AG step 2's own dev-set
     tuning found this guard alone still blocks 92/96 (95.8%) of otherwise-
     robustly-recovered cases -- reported honestly, not hidden, in AUDIT.md.
``robust_threshold`` defaults to 0.7, chosen by sweeping {0.45..1.00} on a
SEPARATE dev split (seed=1, never the seed=0 eval set) and picking the lowest
threshold at the sweep's precision ceiling (49.0% -- no threshold reached the
stated 90% floor; see ``llm_finetuning/tune_robust_reduction_threshold.py``
and AUDIT.md sec AG for the full, unflattering result).
"""
from __future__ import annotations

from collections import Counter
from itertools import groupby

import numpy as np

from . import context_spec as spec
from .calibration import AbsoluteCalibrator, Calibrator
from .config import BASE_FORMATIONS, TRANSITION_CLASS

UNKNOWN_FORMATION = "unknown"
DEFAULT_MAX_KEY_WINDOWS = 10
DISPERSED_CONVERGING_AMBIGUITY_MARGIN = 0.15
DISALLOWED_CHARACTERS = ("⚠",)  # "⚠" -- never in any training prompt

# sec AG step 2: tuned on a dev split (seed=1), never the seed=0 eval set --
# see llm_finetuning/tune_robust_reduction_threshold.py.
DEFAULT_ROBUST_THRESHOLD = 0.7
ALL_UNKNOWN_FALLBACK_MARGIN = 0.05

# 2026-08-09 fix (docs/CEILING.md, V5 Phase 0 guard audit step 4): the leading/
# trailing trim below used to strip EVERY "transitioning"-classified edge window
# unconditionally, trusting each one's label at face value -- audited at 62.5%
# spurious (the window's TRUE label was a real, settled formation the classifier
# simply got wrong, not genuine blend geometry). Reuses the SAME 0.6 confidence
# bar the `low_confidence` guard already uses elsewhere in this codebase (not a
# freshly-tuned value -- no new parameter, no new mining-split tuning question):
# only a "transitioning" read the model itself is reasonably confident about is
# trusted enough to strip. A genuine out-of-vocabulary name (never "transitioning"
# itself) is stripped regardless of confidence -- it was never trustworthy for
# voting purposes either way.
TRIM_CONFIDENCE_THRESHOLD = 0.6


def _validate_formation(name) -> str:
    """Anything not in BASE_FORMATIONS (this INCLUDES the classifier's own
    "transitioning" class -- RULES has no entry for it) becomes the explicit
    abstention sentinel rather than flowing through as if it were real."""
    return name if name in BASE_FORMATIONS else UNKNOWN_FORMATION


def _is_ambiguous_dispersed_converging(class_probabilities: dict) -> bool:
    """2026-08-09 fix (HISTORY.md strategy 7 / AUDIT.md sec AG): the original version of
    this check compared ONLY the raw dispersed/converging probabilities (abs(d-c) < margin)
    with no requirement that either class be competitive for the window's top prediction.
    With 8 softmax classes, whenever one class dominates, the other ~7 split a small
    residual probability mass and land within 0.15 of each other by chance almost always --
    measured directly (scripts/phase0_guard_audit.py) at a 75.8% fire rate on real
    predictions, 66.1% of which had NEITHER class in the window's top-2. Now requires
    dispersed and converging to be the window's top-2 predicted classes (i.e. genuinely
    competing for the top spot), in addition to the original probability-closeness check."""
    if not class_probabilities:
        return False
    d, c = class_probabilities.get("dispersed"), class_probabilities.get("converging")
    if d is None or c is None:
        return False
    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top2_names = {name for name, _ in ranked[:2]}
    if top2_names != {"dispersed", "converging"}:
        return False
    return abs(d - c) < DISPERSED_CONVERGING_AMBIGUITY_MARGIN


def _robust_all_unknown_fallback(predictions, n_stripped_leading, n_stripped_trailing):
    """Step 2c: every window is transitioning/OOV -- aggregate class_probabilities
    across ALL windows and take the top-2 non-transitioning (BASE_FORMATIONS)
    classes by mean probability. Resolvable only if well separated; always
    flagged low_confidence when it does resolve. Returns the same
    (dominant, formation_history, transitions, info) shape as _robust_reduce,
    or None."""
    sums = {f: 0.0 for f in BASE_FORMATIONS}
    counts = 0
    for p in predictions:
        cp = p.get("class_probabilities", {})
        if not cp:
            continue
        counts += 1
        for f in BASE_FORMATIONS:
            sums[f] += cp.get(f, 0.0)
    if counts == 0:
        return None
    means = {f: sums[f] / counts for f in BASE_FORMATIONS}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    top1, top1_p = ranked[0]
    top2, top2_p = ranked[1]
    margin = top1_p - top2_p
    if margin < ALL_UNKNOWN_FALLBACK_MARGIN:
        return None
    info = {"stripped_leading": n_stripped_leading, "stripped_trailing": n_stripped_trailing,
           "mode": "all_unknown_probability_fallback", "top1": top1, "top1_p": round(top1_p, 4),
           "top2": top2, "top2_p": round(top2_p, 4), "margin": round(margin, 4),
           "recovered": True, "low_confidence": True}
    return (top1, [top1], [], info)


def _trustworthy_to_strip(predictions, i):
    """2026-08-09 fix: is window i's UNKNOWN_FORMATION read trustworthy enough
    to strip as genuine edge noise? A "transitioning" read is only trusted when
    the model itself is reasonably confident about it (TRIM_CONFIDENCE_THRESHOLD,
    reusing the existing low_confidence bar); a genuine out-of-vocabulary name
    (never "transitioning") is stripped regardless of confidence -- it was never
    trustworthy for voting purposes either way."""
    if predictions[i]["formation_type"] == TRANSITION_CLASS:
        return predictions[i]["formation_confidence"] >= TRIM_CONFIDENCE_THRESHOLD
    return True


def _robust_reduce(formation_seq, predictions, threshold):
    """Steps 2a+2b (AUDIT.md sec AG): strip leading/trailing UNKNOWN_FORMATION
    runs, then reduce what's left by MAJORITY vote (modal non-unknown formation
    in each half, weighted against that half's FULL length so a noisy half
    correctly fails the threshold) instead of requiring unanimity everywhere.
    Falls back to _robust_all_unknown_fallback (2c) if nothing survives
    stripping. Returns (dominant, formation_history, transitions, info) or
    None if not robustly resolvable -- callers must fall back to the original
    unanimity-based reduction on None, not treat it as "abstain".

    2026-08-09 fix: stripping used to trust every UNKNOWN_FORMATION edge window
    unconditionally (62.5% spurious, audited -- the window's true label was
    often a real, settled formation the classifier simply misclassified as
    "transitioning", not genuine blend geometry). Now only strips a window
    `_trustworthy_to_strip` accepts; an untrusted low-confidence "transitioning"
    window is left in `stripped` as UNKNOWN_FORMATION, where `modal()` below
    already correctly excludes it from voting and counts it against that half's
    plurality fraction -- the safe fallback, never a wrong confident answer."""
    n = len(formation_seq)
    start = 0
    while start < n and formation_seq[start] == UNKNOWN_FORMATION and _trustworthy_to_strip(predictions, start):
        start += 1
    end = n
    while end > start and formation_seq[end - 1] == UNKNOWN_FORMATION and _trustworthy_to_strip(predictions, end - 1):
        end -= 1
    stripped = formation_seq[start:end]
    n_lead, n_trail = start, n - end

    if not stripped:
        return _robust_all_unknown_fallback(predictions, n_lead, n_trail)

    if len(stripped) == 1:
        f = stripped[0]
        return (f, [f], [], {"stripped_leading": n_lead, "stripped_trailing": n_trail,
                             "mode": "single_window", "recovered": True, "low_confidence": True})

    mid = len(stripped) // 2
    first_half, second_half = stripped[:mid], stripped[mid:]

    def modal(seq):
        # UNKNOWN_FORMATION must never itself win the vote -- a half dominated
        # by transitioning/OOV windows should fail the threshold, not be
        # reported as if "unknown" were a real formation. Denominator is the
        # full half length (unknowns count against the plurality), not just
        # the non-unknown subset, or the threshold couldn't reject noise at all.
        counts = Counter(f for f in seq if f != UNKNOWN_FORMATION)
        if not counts:
            return None, 0.0
        f, count = counts.most_common(1)[0]
        return f, count / len(seq)

    f1, frac1 = modal(first_half)
    f2, frac2 = modal(second_half)
    if f1 is None or f2 is None or frac1 < threshold or frac2 < threshold:
        return None

    info = {"stripped_leading": n_lead, "stripped_trailing": n_trail, "mode": "majority_halves",
           "frac_first_half": round(frac1, 4), "frac_second_half": round(frac2, 4),
           "recovered": True, "low_confidence": False}
    if f1 == f2:
        return (f1, [f1], [], info)
    dominant = f1 if frac1 >= frac2 else f2
    orig_mid_idx = min(start + mid, n - 1)
    transitions = [{"at_time_s": predictions[orig_mid_idx]["time_start_s"], "from": f1, "to": f2}]
    return (dominant, [f1, f2], transitions, info)


def _select_key_window_indices(predictions, formation_seq, ambiguous_flags, max_key_windows):
    """Priority order: first, last, unknown-formation windows, low-confidence
    windows, ambiguous windows -- deduped, capped at max_key_windows, THEN
    sorted chronologically. Priority-first (not index-first) truncation is
    what guarantees first/last survive the cap even when many earlier windows
    are also salient."""
    n = len(predictions)
    ordered = []

    def add(i):
        if i not in ordered:
            ordered.append(i)

    add(0)
    add(n - 1)
    for i, f in enumerate(formation_seq):
        if f == UNKNOWN_FORMATION:
            add(i)
    for i, p in enumerate(predictions):
        if p["formation_confidence"] < 0.65:
            add(i)
    for i, amb in enumerate(ambiguous_flags):
        if amb:
            add(i)
    return sorted(ordered[:max_key_windows])


def bridge_predictions(predictions: list[dict], calibrator: "Calibrator | None" = None,
                       max_key_windows: int = DEFAULT_MAX_KEY_WINDOWS,
                       robust: bool = False, robust_threshold: float = DEFAULT_ROBUST_THRESHOLD):
    """predictions: output of ``swarm_intent.stgt.inference.sliding_window_inference``
    (or a hand-assembled list of ``predict_v2`` dicts). Returns
    ``(tactical_context: str, summary: dict, key_windows: list[dict])`` in the same
    shape as ``inference.py``'s ``build_tactical_context``/``build_llm_prompt``, with
    the four fixes in this module's docstring applied.

    robust: False (default) reproduces the original unanimity-based reduction
    byte-for-byte -- every existing caller/test is unaffected. True attempts
    the majority-based robust reduction (module docstring, AUDIT.md sec AG)
    FIRST; if it fails to resolve anything, falls back to the SAME unanimity
    logic below unchanged, so a robust=True case is never worse-informed than
    a robust=False one, only possibly better.
    """
    calibrator = calibrator or AbsoluteCalibrator()
    n = len(predictions)
    if n == 0:
        return ("No predictions available.",
                {"abstain": True, "abstain_reason": "no predictions", "n_windows": 0}, [])

    formation_seq = [_validate_formation(p["formation_type"]) for p in predictions]
    n_unknown = sum(1 for f in formation_seq if f == UNKNOWN_FORMATION)
    # 2026-08-09 fix (docs/CEILING.md, "oov_name guard"): n_unknown counts BOTH a
    # genuinely out-of-vocabulary formation name (real data-integrity concern) AND
    # the classifier's own valid "transitioning" class (a normal, expected read for
    # a blend/transient window, not a vocabulary problem at all) under the SAME
    # sentinel. The oov_name guard's name claims to test the former; audited, it
    # was firing on the latter (V5 Phase 0 full guard audit: 69% spurious when the
    # sole blocker). This narrower count lets the guard test only what it claims.
    n_genuinely_oov = sum(1 for p in predictions
                          if p["formation_type"] not in BASE_FORMATIONS
                          and p["formation_type"] != TRANSITION_CLASS)

    robust_info = None
    robust_result = _robust_reduce(formation_seq, predictions, robust_threshold) if robust else None
    if robust_result is not None:
        dominant, formation_history, transitions, robust_info = robust_result
    else:
        # (1) transitions: consecutive-pair, temporal, NEVER predict_v2's transition_from/to.
        transitions = []
        for i in range(1, n):
            a, b = formation_seq[i - 1], formation_seq[i]
            if UNKNOWN_FORMATION in (a, b):
                continue
            if a != b:
                transitions.append({"at_time_s": predictions[i]["time_start_s"], "from": a, "to": b})

        # (2) formation_history: collapse only CONSECUTIVE repeats -- preserves oscillation.
        formation_history = [k for k, _ in groupby(formation_seq)]

    valid_formations = [f for f in formation_seq if f != UNKNOWN_FORMATION]
    if not valid_formations and robust_result is None:
        summary = {
            "abstain": True,
            "abstain_reason": f"all {n} window(s) classified outside BASE_FORMATIONS "
                              f"(transitioning / unrecognized) -- no reliable formation signal",
            "formation_history": formation_history, "n_windows": n, "n_unknown_windows": n_unknown,
            "n_genuinely_oov_windows": n_genuinely_oov,
        }
        context = ("No reliable formation classification in this observation window "
                  "(every window's classification fell outside the known formation "
                  "vocabulary).")
        return context, summary, []

    if robust_result is None:
        dominant = max(set(valid_formations), key=valid_formations.count)

    velocities = [p["centroid_velocity"] for p in predictions]
    mid = n // 2
    delta_v = (np.mean(velocities[mid:]) - np.mean(velocities[:mid])) if mid > 0 else 0.0
    vel_trend = calibrator.velocity_trend(float(delta_v))

    stabilities = [p["formation_stability"] for p in predictions]
    mean_stability = float(np.mean(stabilities))
    early = np.mean(stabilities[:max(1, mid)])
    late = np.mean(stabilities[mid:]) if mid > 0 else stabilities[-1]
    stab_trend = calibrator.stability_trend(float(early), float(late))

    approach = [p["approach_rate"] for p in predictions]
    mean_approach = float(np.mean(approach))
    approach_summary = calibrator.spread_dynamics(mean_approach)

    role_true_count = sum(1 for p in predictions if p.get("role_differentiation"))
    role_str = calibrator.role_differentiation(role_true_count, n)
    role_flag = role_str == spec.ROLE_DIFFERENTIATION_PRESENT

    confidences = [p["formation_confidence"] for p in predictions]
    mean_conf, low_conf = float(np.mean(confidences)), sum(1 for c in confidences if c < 0.6)

    # (4) per-window dispersed/converging ambiguity.
    ambiguous_flags = [_is_ambiguous_dispersed_converging(p.get("class_probabilities", {}))
                       for p in predictions]
    n_ambiguous = sum(ambiguous_flags)

    lines = [
        f"Observation window: {predictions[0]['time_start_s']}s - {predictions[-1]['time_end_s']}s "
        f"({n} windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {' -> '.join(formation_history)}",
    ]
    if n_unknown:
        lines.append(f"{n_unknown} window(s) had an unrecognized/non-base formation "
                     f"classification and were excluded from transition analysis.")
    lines += ([f"Transition at t={t['at_time_s']}s: {t['from']} -> {t['to']}" for t in transitions]
             or [spec.NO_TRANSITIONS_DETECTED])
    lines += [
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_trend} (mean={mean_stability:.2f})",
        f"Spread dynamics: {approach_summary} (mean approach_rate={mean_approach:.3f})",
        f"Role differentiation: {role_str}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=mean_conf, low_conf=low_conf),
    ]
    if n_ambiguous:
        lines.append(f"{n_ambiguous} window(s) show near-equal dispersed/converging classifier "
                     f"probability (within {DISPERSED_CONVERGING_AMBIGUITY_MARGIN}) -- these two "
                     f"formations share near-identical base geometry, treat formation identity "
                     f"for those windows with caution.")

    summary = {
        "abstain": False,
        "dominant_formation": dominant, "formation_history": formation_history,
        "transitions_detected": transitions, "velocity_trend": vel_trend,
        "delta_velocity": round(float(delta_v), 3), "stability_trend": stab_trend,
        "mean_stability": round(mean_stability, 3), "spread_dynamics": approach_summary,
        "mean_approach_rate": round(mean_approach, 3), "role_differentiation": role_flag,
        "mean_confidence": round(mean_conf, 3), "low_conf_windows": low_conf, "n_windows": n,
        "n_unknown_windows": n_unknown, "n_genuinely_oov_windows": n_genuinely_oov,
        "n_ambiguous_dispersed_converging_windows": n_ambiguous,
        "robust_reduction": robust_info,  # None unless robust=True AND it actually recovered a pair
    }

    # (3) key_windows, capped.
    key_idx = _select_key_window_indices(predictions, formation_seq, ambiguous_flags, max_key_windows)
    key_windows = [{
        "t": f"{predictions[i]['time_start_s']}-{predictions[i]['time_end_s']}s",
        "formation": formation_seq[i],
        "confidence": predictions[i]["formation_confidence"],
        "velocity": predictions[i]["centroid_velocity"],
        "approach": predictions[i]["approach_rate"],
        "stability": predictions[i]["formation_stability"],
        "ambiguous_dispersed_converging": ambiguous_flags[i],
    } for i in key_idx]

    context_text = "\n".join(lines)
    for ch in DISALLOWED_CHARACTERS:
        assert ch not in context_text, f"disallowed character {ch!r} in generated context"
    return context_text, summary, key_windows
