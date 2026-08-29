"""
CONSOLIDATED long-trajectory generator for Phase 0 eval/audit scripts (docs/V5_LOG.md step
26). Single canonical implementation of ``sample_chain``/``build_long_sequence_labeled``/
``ground_truth_pair``, replacing what had become 4 independently-maintained, verbatim-duplicate
copies (``scripts/phase0_ceiling.py``, ``scripts/phase0_guard_audit.py``,
``scripts/phase0_decompose_failures.py``, ``llm_finetuning/measure_coverage.py``) that had
ALREADY silently diverged once (the 2026-08-09 chain-2 dwell-time fix landed in only one of
the four before this consolidation) -- exactly the failure mode this module exists to make
structurally impossible going forward.

NOT the training-data path. STGT's actual training data comes from
``swarm_intent.data.generate_dataset()``, which calls ``generate_transition_sequence`` directly
with its own fixed ``n_timesteps=50``, 3-regime blend timing (see that module) -- entirely
separate code, never shared with this one. This module exists purely to build long, multi-hop,
REALISTIC EVALUATION trajectories for measuring an already-trained checkpoint's ceiling; it has
no bearing on what STGT is ever trained on, and no "canonical" version of it feeds a retrain,
because nothing in this file's lineage ever has.

``llm_finetuning/analyze_bucket_c_windowing.py`` has its OWN separate, inline copy
(``build_long_sequence_instrumented``) that is DELIBERATELY NOT consolidated here: that script's
entire purpose (see its own docstring) is to reproduce one specific historical seed=0
population bit-for-bit for a closed, already-reported diagnostic (``AUDIT.md`` sec AF step 4).
Pointing it at this module's current (fixed) formula would silently break the bit-for-bit
reproducibility that script exists to guarantee -- an intentional, documented exception, not an
oversight.

--- Observability fix derivation (docs/V5_LOG.md steps 24-26) ---

A window's true label is the majority vote (``Counter.most_common``) over its 50 timesteps
(``window_size=50``), so an endpoint formation needs >=26 timesteps of a window to ever win it.

DESTINATION (B) side (step 24/25): B-labeled timesteps come only from the post-blend "dwell"
region, ``D = seg_len - blend_end``. ``sliding_window_inference``'s ``stride=10`` grid can leave
up to 9 trailing timesteps outside EVERY window (worst case: ``(seg_len-window_size) % stride
== 9``), so the real requirement is ``D >= 26 + 9 = 35``. ``MIN_DWELL_RANGE=(40,60)`` gives
real margin above that minimum.

SOURCE (A) side (step 26): A-labeled timesteps occupy ``[0, blend_start]`` and window 0 always
starts at exactly ``t=0`` (the sliding-window grid always includes ``start=0``) -- there is NO
equivalent trailing-slack loss on this side. The requirement is simply
``lead_in + 1 >= 26``, i.e. ``lead_in >= 25``, for formation A to hold an outright majority
(not just a plurality) of window 0 regardless of how ``blend_duration``/dwell split the rest of
that window. ``LEAD_IN_RANGE=(30,50)`` mirrors ``MIN_DWELL_RANGE``'s margin shape (threshold+5
to threshold+25) above this minimum. Before this fix, ``LEAD_IN_RANGE=(15,35)`` carried over
the pre-observability-fix formula's REALIZED scale rather than being derived -- roughly half
that range (15-25) sits below the 25-26 minimum, which is exactly why the destination-only fix
(commit before this one) left a new, self-inflicted ~15%-16% source-non-observability gap
(docs/V5_LOG.md step 25's 20-case trace; verified at population scale in step 26: 16.3% source
OBS_NONE under the old range vs 0.0% after this fix).

``blend_duration`` is UNCHANGED transition physics/timing scale, chosen to match the original
(pre-fix) formula's REALIZED duration range (~10-25 timesteps, mean ~18.8) -- only WHERE the
blend is placed changed, never ``generate_transition_sequence``'s own cosine-ramp mechanism.
"""
from __future__ import annotations

import numpy as np

from .config import BASE_FORMATIONS, TRANSITION_CLASS
from .data import (  # noqa: F401 -- LEAD_IN_RANGE/BLEND_DURATION_RANGE/MIN_DWELL_RANGE
    # re-exported here since every existing consumer of this module imports them from here;
    # canonical definition moved to data.py 2026-08-10 (docs/V5_LOG.md step 31) now that
    # generate_dataset()'s corrected_blend_timing/windowed_examples flags use them too --
    # single source of truth, closing the exact silent-drift risk step 26's audit found once.
    generate_transition_sequence, LEAD_IN_RANGE, BLEND_DURATION_RANGE, MIN_DWELL_RANGE,
)


def sample_chain(rng: np.random.Generator) -> list[str]:
    num_formations = int(rng.integers(1, 5))  # {1,2,3,4}, uniform
    chain = [rng.choice(BASE_FORMATIONS)]
    for _ in range(num_formations - 1):
        pool = [f for f in BASE_FORMATIONS if f != chain[-1]]
        chain.append(rng.choice(pool))
    return chain


def ground_truth_pair(true_chain: list):
    if len(true_chain) == 1:
        return (true_chain[0], true_chain[0])
    if len(true_chain) == 2:
        return (true_chain[0], true_chain[1])
    return None


def build_long_sequence_labeled(chain: list[str], rng: np.random.Generator, spread: float, noise_std: float):
    """Returns (long_seq, true_labels): a rigid-translated concatenation of one
    generate_transition_sequence() call per hop (or one steady segment if len(chain)==1),
    plus a parallel per-timestep TRUE label array (formation name, or TRANSITION_CLASS
    during each hop's cosine blend window)."""
    segments, seg_labels = [], []
    if len(chain) == 1:
        seg_len = int(rng.integers(50, 101))
        seg = generate_transition_sequence(chain[0], chain[0], n_timesteps=seg_len,
                                           spread=spread, noise_std=noise_std, rng=rng)
        segments.append(seg)
        seg_labels.append([chain[0]] * seg_len)
    else:
        for i in range(len(chain) - 1):
            lead_in = int(rng.integers(*LEAD_IN_RANGE))
            blend_duration = int(rng.integers(*BLEND_DURATION_RANGE))
            dwell = int(rng.integers(*MIN_DWELL_RANGE))
            blend_start = lead_in
            blend_end = lead_in + blend_duration
            seg_len = blend_end + dwell
            seg = generate_transition_sequence(chain[i], chain[i + 1], n_timesteps=seg_len,
                                               spread=spread, noise_std=noise_std,
                                               blend_start=blend_start, blend_end=blend_end, rng=rng)
            segments.append(seg)
            labels = []
            for t in range(seg_len):
                if t <= blend_start:
                    labels.append(chain[i])
                elif t >= blend_end:
                    labels.append(chain[i + 1])
                else:
                    labels.append(TRANSITION_CLASS)
            seg_labels.append(labels)

    stitched = [segments[0]]
    for seg in segments[1:]:
        prev_last_centroid = stitched[-1][-1].mean(axis=0)
        this_first_centroid = seg[0].mean(axis=0)
        delta = prev_last_centroid - this_first_centroid
        stitched.append(seg + delta[None, None, :])
    long_seq = np.concatenate(stitched, axis=0)
    true_labels = [lab for seg_lab in seg_labels for lab in seg_lab]
    assert len(true_labels) == long_seq.shape[0]
    return long_seq, true_labels
