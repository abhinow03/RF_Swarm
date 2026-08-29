"""
Synthetic swarm-trajectory generation, dataset assembly, and splitting.

Consolidates the data-generation code that was duplicated across the three
notebooks. Two correctness fixes versus the originals:

1. A single seeded ``np.random.Generator`` is threaded through ALL sampling
   (velocity, direction, dispersed offsets, noise). The originals seeded only
   the outer loop, so datasets were not actually reproducible.
2. Train/val/test normalisation uses TRAIN statistics only, applied to all
   splits (the original computed per-split stats — harmless on i.i.d. synthetic
   data, but wrong in principle and a bug if data ever stops being i.i.d.).

NOTE: ``generate_transition_sequence`` is reconstructed from the documented
behaviour in the notebook (cosine-ramp blend between two formations). Diff it
against the original notebook on first run if exact parity matters.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .config import Config, BASE_FORMATIONS, TRANSITION_CLASS
from .formations import get_formation_offsets

# --- corrected blend-timing / windowing / labeling constants (2026-08-10 decision, see
# docs/V5_LOG.md steps 31-33, HISTORY.md's 2026-08-10 "port design" decision) ---
#
# Canonical home for LEAD_IN_RANGE/BLEND_DURATION_RANGE/MIN_DWELL_RANGE: this module, not
# eval_trajectories.py, now that both the training path (generate_dataset, via the
# corrected_blend_timing/windowed_examples flags below) and the evaluation path want them.
# eval_trajectories.py imports these FROM here rather than duplicating them, closing off
# the exact silent-drift risk the V5 program's step-26 audit found once already (a Monte
# Carlo diagnostic script whose own hardcoded copy of these ranges went stale).
WINDOW_SIZE = 50            # matches Config.max_seq_len -- PositionalEncoding's buffer is
                             # sized to this; every existing checkpoint assumes it exactly.
STRIDE = 10                 # matches sliding_window_inference's default stride.
LEAD_IN_RANGE = (30, 50)          # timesteps of settled formation_a before the blend starts
BLEND_DURATION_RANGE = (10, 25)   # timesteps the blend itself spans
MIN_DWELL_RANGE = (40, 60)        # timesteps of settled formation_b after the blend ends

# Window-labeling thresholds (docs/V5_LOG.md step 31 has the full derivation):
#
# PURE_LABEL_THRESHOLD = 0.70 (35/50): a window is labeled a PURE endpoint formation only
# if that formation's per-timestep true content is a COMFORTABLE majority, not a bare one.
# Reuses the exact same 35/50 figure already derived for MIN_DWELL_RANGE/LEAD_IN_RANGE
# (docs/V5_LOG.md steps 24-26: a window needs >=26/50 timesteps of one formation for an
# outright majority; +9 stride-slack margin gives real headroom above that minimum, 35/50 =
# 0.70). Reusing it here is not laziness: it means "a window eval's OWN observability logic
# would trust as reliably showing formation A" and "a window training confidently labels
# pure-A" are THE SAME window, by construction -- train and eval agree on what "confidently
# observed" means, not just on blend timing.
#
# TRANS_LABEL_MIN_BLEND_FRAC = 0.20 (10/50): "transitioning" as a label CANNOT use the same
# 0.70 bar -- BLEND_DURATION_RANGE's own max (25 timesteps) is exactly half of WINDOW_SIZE,
# so blend content can never reach even a bare majority (>=26/50) of any window, let alone
# 70%. Requiring 0.70 for "transitioning" would make the label unreachable under the
# corrected timing and silently delete the class from ported training data. Instead: a
# window is "transitioning" if blend content is the PLURALITY of its three content types
# (beats both pure_a and pure_b individually) AND is at least as large as the shortest
# possible full blend region can produce (BLEND_DURATION_RANGE's own minimum, 10/50 = 0.20)
# -- below that floor, the window is grazing a blend edge without containing a meaningful
# chunk of it, and belongs in the EXCLUDE band, not the transitioning class.
PURE_LABEL_THRESHOLD = 0.70
TRANS_LABEL_MIN_BLEND_FRAC = 0.20

# 2026-08-10, docs/V5_LOG.md steps 39/41: an ACCEL_SPEED_CAP was added here (step 39) to
# bound generate_transition_sequence's acceleration term over the port's 80-132-step hops,
# hypothesized as the cause of a small-scale training collapse (step 37). Step 39's own
# controlled test showed the cap made NO measurable difference to the collapse (identical
# result, capped vs uncapped). Step 41 then showed ROBUST (percentile-trimmed) position
# normalization ALONE -- with acceleration left completely uncapped -- resolves the collapse
# (test_acc 13.2%->87.7%, chain-2 pair/threat 0%/0%->51.8%/60.5%). The cap was never a real
# contributor; REVERTED here rather than carried alongside a fix that does the actual work,
# per the standing project convention of not shipping an unneeded change. See
# split_and_normalize's robust= parameter for the fix that actually matters.


def _label_window(frac_a: float, frac_blend: float, frac_b: float):
    """Assigns ONE training label to a window from its realized per-timestep content
    fractions (must sum to ~1.0), or returns None if the window should be EXCLUDED from
    training rather than mislabeled. See the threshold derivations above.

    Order matters: pure checks first (a window that is 70% A and 20% blend and 10% B is
    unambiguously pure-A even though blend > TRANS_LABEL_MIN_BLEND_FRAC) -- transitioning
    is deliberately the fallback for windows where NEITHER endpoint dominates, not a
    competing first-class check.
    """
    if frac_a >= PURE_LABEL_THRESHOLD:
        return "a"
    if frac_b >= PURE_LABEL_THRESHOLD:
        return "b"
    if frac_blend >= TRANS_LABEL_MIN_BLEND_FRAC and frac_blend > frac_a and frac_blend > frac_b:
        return "transitioning"
    return None


def generate_swarm_sequence(
    formation_type: str,
    n_timesteps: int = 50,
    dt: float = 0.5,
    spread: float = 1.0,
    noise_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, list, dict]:
    """Generate one (n_timesteps, 6, 3) trajectory for a single formation."""
    if rng is None:
        rng = np.random.default_rng()

    centroid = np.array([0.0, 0.0, 100.0])
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(3.0, 8.0)
    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle),
                         rng.uniform(-0.5, 0.5)])
    # Acceleration ported from upstream commit 9158b081 -- colinear with the
    # initial heading, magnitude/sign randomized, so velocity is no longer
    # constant for the whole trajectory.
    accel_mag = rng.uniform(-1.0, 1.0)
    acceleration = np.array([accel_mag * np.cos(angle), accel_mag * np.sin(angle),
                             rng.uniform(-0.1, 0.1)])

    base_offsets = get_formation_offsets(formation_type, spread, rng=rng)
    sequence = np.zeros((n_timesteps, 6, 3))
    labels = []

    for t in range(n_timesteps):
        velocity = velocity + acceleration * dt
        centroid = centroid + velocity * dt
        if formation_type == "converging":
            shrink = 1.0 - (0.9 * t / (n_timesteps - 1))
            offsets = base_offsets * shrink
        else:
            offsets = base_offsets
        drone_positions = centroid + offsets
        drone_positions = drone_positions + rng.normal(0.0, noise_std, size=(6, 3))
        sequence[t] = drone_positions
        labels.append(formation_type)

    meta = {"formation_type": formation_type, "speed": speed,
            "direction_angle": angle, "spread": spread,
            "noise_std": noise_std, "n_timesteps": n_timesteps, "dt": dt}
    return sequence, labels, meta


def generate_transition_sequence(
    formation_a: str,
    formation_b: str,
    n_timesteps: int = 50,
    dt: float = 0.5,
    spread: float = 1.0,
    noise_std: float = 0.5,
    blend_start: int = 20,
    blend_end: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a sequence that morphs from formation_a to formation_b.

    Offsets are interpolated per-drone with a cosine ramp over
    [blend_start, blend_end] so the transition is physically smooth.
    """
    if rng is None:
        rng = np.random.default_rng()

    centroid = np.array([0.0, 0.0, 100.0])
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(3.0, 8.0)
    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle),
                         rng.uniform(-0.5, 0.5)])
    # Acceleration ported from upstream commit 9158b081 -- see
    # generate_swarm_sequence's comment for the rationale.
    accel_mag = rng.uniform(-1.0, 1.0)
    acceleration = np.array([accel_mag * np.cos(angle), accel_mag * np.sin(angle),
                             rng.uniform(-0.1, 0.1)])

    off_a = get_formation_offsets(formation_a, spread, rng=rng)
    off_b = get_formation_offsets(formation_b, spread, rng=rng)
    sequence = np.zeros((n_timesteps, 6, 3))

    for t in range(n_timesteps):
        velocity = velocity + acceleration * dt
        centroid = centroid + velocity * dt
        if t <= blend_start:
            alpha = 0.0
        elif t >= blend_end:
            alpha = 1.0
        else:
            frac = (t - blend_start) / (blend_end - blend_start)
            alpha = 0.5 * (1 - np.cos(np.pi * frac))  # cosine ramp 0->1
        offsets = (1 - alpha) * off_a + alpha * off_b
        drone_positions = centroid + offsets
        drone_positions = drone_positions + rng.normal(0.0, noise_std, size=(6, 3))
        sequence[t] = drone_positions
    return sequence


def generate_dataset(
    cfg: Config,
    n_per_formation: int = 1000,
    n_timesteps: int = 50,
    dt: float = 0.5,
    include_transitions: bool = False,
    n_transition: int = 0,
    corrected_blend_timing: bool = False,
    windowed_examples: bool = False,
    content_majority_labeling: bool = False,
    return_diagnostics: bool = False,
):
    """Build the full dataset across all formations (+ optional transitions).

    Three independently toggleable flags port eval_trajectories.py's corrected blend-timing
    distribution into training (2026-08-10 decision, docs/V5_LOG.md steps 31-33; all default
    False, reproducing the exact pre-2026-08-10 behaviour):

    corrected_blend_timing   -- sample blend placement from BLEND_DURATION_RANGE (and, when
                                 windowed_examples is also True, LEAD_IN_RANGE/MIN_DWELL_RANGE)
                                 instead of the old 3-regime fractional scheme. When False but
                                 windowed_examples is True, a neutral fixed seg_len (106, the
                                 corrected range's own mean) is used with the OLD regime logic
                                 scaled to it, so that ablation isolates windowing alone.
    windowed_examples        -- generate a long hop and slide WINDOW_SIZE/STRIDE across it
                                 (like sliding_window_inference), yielding zero or more 50-step
                                 training examples per hop, instead of one direct n_timesteps
                                 example per call. n_transition then means "hops sampled", not
                                 "examples produced" -- see return_diagnostics.
    content_majority_labeling -- label each example/window from its REALIZED per-timestep
                                 content via _label_window's purity/plurality rule (may EXCLUDE
                                 a window rather than mislabel it). When False, a window's
                                 label is either the old regime's implied label (when
                                 corrected_blend_timing is False) or a naive best-effort
                                 plurality label with NO purity floor and NO exclusion (when
                                 corrected_blend_timing is True and this is False) -- documented
                                 as the crude fallback it is, used only to isolate the other two
                                 flags' individual effects during ablation.

    return_diagnostics: if True, also returns a dict with per-example (frac_a, frac_blend,
    frac_b, assigned_label) for every transitioning-pool example actually kept, plus counts of
    how many were excluded -- used by scripts/phase0_generator_port_diagnostics.py for the
    step-4 label-sanity report. Never used by scripts/generate_data.py / train_model.py.

    Returns
    -------
    X : (N, n_timesteps or WINDOW_SIZE, 6, 3)
    y : (N,)  integer labels
    names : list[str]  index -> formation name
    diagnostics : dict, only if return_diagnostics=True
    """
    rng = np.random.default_rng(cfg.seed)
    names = list(BASE_FORMATIONS)
    label_map = {name: i for i, name in enumerate(names)}

    seqs, labels = [], []
    for formation in BASE_FORMATIONS:
        for _ in range(n_per_formation):
            spread = rng.uniform(0.7, 1.5)
            noise = rng.uniform(0.3, 0.8)
            seq, _, _ = generate_swarm_sequence(
                formation, n_timesteps, dt, spread, noise, rng=rng
            )
            seqs.append(seq)
            labels.append(label_map[formation])

    diag = {"n_hops_sampled": 0, "n_examples_kept": 0, "n_excluded": 0,
            "examples": [],           # each: (frac_a, frac_blend, frac_b, label_str)
            "excluded_examples": []}  # each: (frac_a, frac_blend, frac_b, formation_a, formation_b)

    if include_transitions and n_transition > 0:
        names.append(TRANSITION_CLASS)
        trans_label = label_map.setdefault(TRANSITION_CLASS, len(label_map))
        pairs = [(a, b) for a in BASE_FORMATIONS for b in BASE_FORMATIONS if a != b]

        def _old_regime_blend(seg_len, rng):
            """The pre-2026-08-10 3-regime fractional scheme (see git history for its own
            derivation), parameterized by segment length so it can be reused both at
            n_timesteps=50 (unwindowed) and at a neutral long seg_len (windowed-only
            ablation)."""
            margin = max(1, int(0.10 * seg_len))
            regime = int(rng.integers(0, 3))
            if regime == 0:
                blend_start = int(seg_len * rng.uniform(0.74, 0.90))
                blend_end = int(np.clip(blend_start + seg_len * rng.uniform(0.05, 0.10),
                                        blend_start + margin, seg_len - 1))
            elif regime == 1:
                blend_start = int(seg_len * rng.uniform(0.12, 0.30))
                blend_end = int(np.clip(blend_start + seg_len * rng.uniform(0.45, 0.62),
                                        blend_start + margin, seg_len - margin))
            else:
                blend_end = int(seg_len * rng.uniform(0.10, 0.26))
                blend_start = int(np.clip(blend_end - seg_len * rng.uniform(0.05, 0.10),
                                          1, blend_end - margin))
            blend_start = max(1, min(blend_start, seg_len - 2))
            blend_end = max(blend_start + 1, min(blend_end, seg_len - 1))
            return blend_start, blend_end, regime

        def _content_fracs(window_start, window_end, blend_start, blend_end):
            """Fraction of [window_start, window_end) timesteps that are pure_a
            (t<=blend_start), blend (blend_start<t<blend_end), pure_b (t>=blend_end) --
            same convention eval_trajectories.py's build_long_sequence_labeled uses."""
            n = window_end - window_start
            ts = np.arange(window_start, window_end)
            n_a = int(np.sum(ts <= blend_start))
            n_b = int(np.sum(ts >= blend_end))
            n_blend = n - n_a - n_b
            return n_a / n, n_blend / n, n_b / n

        def _assign(frac_a, frac_blend, frac_b, regime_label, use_content_rule):
            if use_content_rule:
                r = _label_window(frac_a, frac_blend, frac_b)
                if r is None:
                    return None
                return {"a": "pure_a", "b": "pure_b", "transitioning": "transitioning"}[r]
            if regime_label is not None:
                return regime_label  # old regime's implied label, unchanged behaviour
            # naive fallback (corrected_blend_timing=True, content_majority_labeling=False):
            # bare plurality, no purity floor, no exclusion -- deliberately crude, isolates
            # the OTHER two flags' effects rather than being a real labeling proposal.
            best = max(("a", frac_a), ("transitioning", frac_blend), ("b", frac_b), key=lambda kv: kv[1])[0]
            return {"a": "pure_a", "b": "pure_b", "transitioning": "transitioning"}[best]

        for _ in range(n_transition):
            f_a, f_b = pairs[rng.integers(len(pairs))]
            spread = rng.uniform(0.7, 1.5)
            noise = rng.uniform(0.3, 0.8)
            diag["n_hops_sampled"] += 1

            if not windowed_examples:
                if corrected_blend_timing:
                    blend_duration = int(rng.integers(*BLEND_DURATION_RANGE))
                    blend_duration = min(blend_duration, n_timesteps - 2)
                    blend_start = int(rng.integers(0, n_timesteps - blend_duration))
                    blend_end = blend_start + blend_duration
                    regime_label = None
                else:
                    blend_start, blend_end, regime = _old_regime_blend(n_timesteps, rng)
                    regime_label = {0: "pure_a", 1: "transitioning", 2: "pure_b"}[regime]

                frac_a, frac_blend, frac_b = _content_fracs(0, n_timesteps, blend_start, blend_end)
                lbl = _assign(frac_a, frac_blend, frac_b, regime_label, content_majority_labeling)
                if lbl is None:
                    diag["n_excluded"] += 1
                    diag["excluded_examples"].append((frac_a, frac_blend, frac_b, f_a, f_b))
                    continue
                seq_label = {"pure_a": label_map[f_a], "pure_b": label_map[f_b],
                            "transitioning": trans_label}[lbl]
                seq = generate_transition_sequence(f_a, f_b, n_timesteps, dt, spread, noise,
                                                   blend_start=blend_start, blend_end=blend_end, rng=rng)
                seqs.append(seq)
                labels.append(seq_label)
                diag["n_examples_kept"] += 1
                diag["examples"].append((frac_a, frac_blend, frac_b, lbl, f_a, f_b))

            else:
                if corrected_blend_timing:
                    lead_in = int(rng.integers(*LEAD_IN_RANGE))
                    blend_duration = int(rng.integers(*BLEND_DURATION_RANGE))
                    dwell = int(rng.integers(*MIN_DWELL_RANGE))
                    blend_start, blend_end = lead_in, lead_in + blend_duration
                    seg_len = blend_end + dwell
                    old_regime = None
                else:
                    seg_len = int(np.mean([sum(LEAD_IN_RANGE) / 2 + sum(BLEND_DURATION_RANGE) / 2
                                          + sum(MIN_DWELL_RANGE) / 2]))  # 106, held constant so
                                                                        # this ablation isolates
                                                                        # windowing, not length
                    blend_start, blend_end, old_regime = _old_regime_blend(seg_len, rng)

                seq = generate_transition_sequence(f_a, f_b, seg_len, dt, spread, noise,
                                                   blend_start=blend_start, blend_end=blend_end, rng=rng)
                for start in range(0, seg_len - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    frac_a, frac_blend, frac_b = _content_fracs(start, end, blend_start, blend_end)
                    regime_label = ({0: "pure_a", 1: "transitioning", 2: "pure_b"}[old_regime]
                                    if old_regime is not None and not content_majority_labeling
                                    else None)
                    lbl = _assign(frac_a, frac_blend, frac_b, regime_label, content_majority_labeling)
                    if lbl is None:
                        diag["n_excluded"] += 1
                        diag["excluded_examples"].append((frac_a, frac_blend, frac_b, f_a, f_b))
                        continue
                    seq_label = {"pure_a": label_map[f_a], "pure_b": label_map[f_b],
                                "transitioning": trans_label}[lbl]
                    seqs.append(seq[start:end])
                    labels.append(seq_label)
                    diag["n_examples_kept"] += 1
                    diag["examples"].append((frac_a, frac_blend, frac_b, lbl, f_a, f_b))

    if return_diagnostics:
        return np.array(seqs), np.array(labels), names, diag
    return np.array(seqs), np.array(labels), names


NORM_TRIM_PERCENTILE = 5  # 2026-08-10, docs/V5_LOG.md step 41 -- see split_and_normalize's
                          # robust= docstring for the derivation


def split_and_normalize(X: np.ndarray, y: np.ndarray, cfg: Config, robust: bool = False) -> dict:
    """Stratified 70/15/15 split + train-only normalisation.

    robust=False (default, unchanged behaviour): train_mean/train_std are the RAW mean/std of
    the whole X_train array -- a single global scalar pair. Fine when X_train's absolute-
    position scale is homogeneous across examples (every example this project generated before
    2026-08-10's windowed_examples port).

    robust=True (2026-08-10, docs/V5_LOG.md step 41): train_mean/train_std are computed from
    the [NORM_TRIM_PERCENTILE, 100-NORM_TRIM_PERCENTILE] percentile range of X_train's values
    instead of the raw range -- a standard trimmed-statistics recipe. Motivation: windowed_
    examples' windows are drawn from arbitrary positions within long (80-132 step) hops, so
    absolute centroid drift varies enormously across examples in the SAME dataset (measured,
    docs/V5_LOG.md step 40: p50 drift 201.8 vs p95 829.9 vs max 1536.4, in the same train set)
    -- a small high-drift tail can dominate a raw global mean/std, uniformly distorting the
    normalized scale for every OTHER (well-behaved) example too. 5% is not arbitrary: it's
    chosen to exclude that tail (drift only becomes extreme above roughly the 90th-95th
    percentile in the step-40 measurement) while trimming only a small, symmetric fraction of
    the distribution, standard practice for trimmed mean/std. Requires no change anywhere else
    in the pipeline: train_mean/train_std remain exactly two saved scalars, applied identically
    at eval time (sliding_window_inference's `(window - train_mean) / train_std`) regardless of
    how they were computed.

    Returns a dict with X_train/X_val/X_test, y_*, and train_mean/train_std.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=cfg.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=cfg.seed
    )
    # Normalisation stats from TRAIN ONLY, applied everywhere.
    if robust:
        lo, hi = np.percentile(X_train, [NORM_TRIM_PERCENTILE, 100 - NORM_TRIM_PERCENTILE])
        trimmed = X_train[(X_train >= lo) & (X_train <= hi)]
        train_mean = trimmed.mean()
        train_std = trimmed.std() + 1e-8
    else:
        train_mean = X_train.mean()
        train_std = X_train.std() + 1e-8
    norm = lambda a: (a - train_mean) / train_std
    return {
        "X_train": norm(X_train), "X_val": norm(X_val), "X_test": norm(X_test),
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "train_mean": float(train_mean), "train_std": float(train_std),
    }


def save_splits(splits: dict, cfg: Config) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
    for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        np.save(os.path.join(cfg.data_dir, f"{k}.npy"), splits[k])
    np.save(os.path.join(cfg.data_dir, "norm_stats.npy"),
            np.array([splits["train_mean"], splits["train_std"]]))
