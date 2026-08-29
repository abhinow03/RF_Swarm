"""
AUDIT.md sec AD found that the global log-p(c) prior correction (corrected_logP(c)
= raw_logP(c) - log(class_freq(c)), applied over all four threat_level candidates)
that recovers accuracy on the low-threat medium/low near-tie collapse ACTIVELY HURTS
accuracy on high/critical cases (35.7% -> 14.3%) -- an extreme class-frequency ratio
(medium 44.9% vs critical 4.1%, ~11x) makes the log-boost for the rare class
overshoot and drag correctly-classified `high` predictions into `critical`.

sec AE step 1 fixes this by SCOPING the correction: only apply it when the raw
argmax is "medium" AND "low" is the specific runner-up (the actual near-tie shape
the low-threat collapse showed) -- never when medium's runner-up is high/critical.
Within scope, the corrected choice is also restricted to {low, medium} only (never
lets the correction move outside the pair it was scoped for, even if the full
4-way corrected distribution would rank critical/high above medium -- see
scoped_correct's docstring). Out of scope (any high/critical argmax, or a medium
argmax whose runner-up is high/critical), the prediction passes through byte-
identical to raw. See tests/test_prior_correction.py for the guard this exists to
never violate.

Sequence-scoring helpers (score_candidates, find_key_marker_position) are the same
teacher-forced-candidate technique used throughout this project's logit-inspection
sessions (llm_finetuning/logit_inspection.py, measure_base_rules_prior.py) --
consolidated here as the one production-importable copy so src/swarm_intent (this
package) doesn't need to import from llm_finetuning/ to run scoped correction at
inference time (pipeline_v2.py's Layer 3). Lazy-imports torch, same convention as
client.py's LocalHFClient, so importing this module doesn't require a GPU/torch
install.
"""
from __future__ import annotations

import numpy as np

CANDIDATES = ("low", "medium", "high", "critical")
THREAT_KEY_MARKER = '"threat_level": "'


def find_key_marker_position(tok, gen_ids, marker: str = THREAT_KEY_MARKER):
    """Minimal prefix length (in generated-token count) whose decoded text
    contains `marker` -- the NEXT token is where the value begins. Returns None
    if the marker never appears in the generation."""
    for n in range(1, len(gen_ids) + 1):
        if marker in tok.decode(gen_ids[:n], skip_special_tokens=True):
            return n
    return None


def score_candidates(model, tok, prefix_ids) -> dict:
    """Teacher-forced sequence log-probability of each CANDIDATES string at
    `prefix_ids` (a (1, L) tensor already on model.device) -- handles multi-token
    candidates correctly (raw next-token comparison would be wrong if e.g.
    "critical" tokenizes differently than "low")."""
    import torch
    logprobs = {}
    for cand in CANDIDATES:
        cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
        full_ids = torch.cat([prefix_ids, torch.tensor([cand_ids], device=prefix_ids.device)], dim=1)
        with torch.no_grad():
            out = model(full_ids)
        logits = out.logits[0]
        start = prefix_ids.shape[1] - 1
        logprob_sum = 0.0
        for i, tid in enumerate(cand_ids):
            logprob_sum += torch.log_softmax(logits[start + i], dim=-1)[tid].item()
        logprobs[cand] = logprob_sum
    return logprobs


def softmax_over_candidates(logprobs: dict) -> dict:
    vals = np.array([logprobs[c] for c in CANDIDATES])
    vals = vals - vals.max()
    p = np.exp(vals)
    p = p / p.sum()
    return {c: float(p[i]) for i, c in enumerate(CANDIDATES)}


def scoped_correct(raw_p: dict, class_freq: dict) -> dict:
    """The SCOPED log-p(c) correction (sec AE step 1). `raw_p`: {candidate: P},
    a softmax over CANDIDATES (e.g. softmax_over_candidates's output). `class_freq`:
    {candidate: frequency}, the reference distribution to correct toward.

    Only fires when raw argmax is "medium" and "low" is the runner-up -- the exact
    near-tie shape the low-threat collapse showed (secs Y/CC/AA). Never fires on a
    high/critical argmax, and never fires on a medium argmax whose runner-up is
    high/critical (that shape is the sec AD failure mode this function exists to
    stop reproducing). When in scope, the corrected choice is restricted to
    {low, medium} ONLY -- computed from the 2-class renormalized log-p(c)
    correction over just that pair, not the full 4-class correction -- so even a
    rare class's oversized log-boost (critical's ~11x ratio vs medium) has no path
    to become the output; it was never in scope for this near-tie in the first
    place.

    Returns {"applied": bool, "corrected_argmax": str, "corrected_p": dict,
    "reason": str}. "applied" is True only when correction actually changed the
    label (medium -> low); an in-scope case that stays medium after correction
    reports applied=False with reason="in scope, no flip".
    """
    sorted_classes = sorted(raw_p, key=raw_p.get, reverse=True)
    raw_argmax, runner_up = sorted_classes[0], sorted_classes[1]

    if not (raw_argmax == "medium" and runner_up == "low"):
        return {"applied": False, "corrected_argmax": raw_argmax, "corrected_p": dict(raw_p),
                "reason": f"out of scope: argmax={raw_argmax} runner_up={runner_up}"}

    corrected_logp = {c: np.log(max(raw_p[c], 1e-12)) - np.log(max(class_freq[c], 1e-6))
                      for c in ("low", "medium")}
    corrected_argmax = max(corrected_logp, key=corrected_logp.get)
    applied = corrected_argmax != raw_argmax
    reason = "in scope, flipped medium -> low" if applied else "in scope, no flip"
    return {"applied": applied, "corrected_argmax": corrected_argmax, "corrected_p": dict(raw_p),
            "reason": reason}
