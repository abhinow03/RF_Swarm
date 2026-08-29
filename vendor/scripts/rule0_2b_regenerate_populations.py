"""
Rule-0 follow-up (2026-08-13), resolving finding 2b: no locked seed=999 eval file exists,
only "same seed" across independently-rewritten generator code. This script regenerates the
REAL seed=999, n=1000 trajectory population under each DISTINCT version of the eval-trajectory
generator that has ever produced a ceiling number cited in V5_LOG.md/CEILING.md/AUDIT.md/
PREREGISTRATION.md, and persists full raw trajectories (not summary stats) so step 3's
trajectory-by-trajectory diff can run against real data.

Three distinct generator-code states exist (git-archaeology, not guesswork -- see each
version's docstring below for the exact commit citation):

  OLD   -- scripts/phase0_ceiling.py's own inline copy, unchanged from b680c80 (first commit
           that added it) through 628dc56 (last commit before consolidation). This is the code
           that produced EVERY *_v2 through *_v6 / *_domfix / *_guardfix / *_oovfix / *_trimfix
           ceiling figure in evaluation/ -- all of those runs predate commit 3591051
           (2026-08-09 18:24:31) which replaced this inline copy.
  MID   -- src/swarm_intent/eval_trajectories.py exactly as landed in 3591051 (consolidation):
           dwell-time-fixed formula, but LEAD_IN_RANGE still (15,35) (pre-symmetrization value,
           per that commit's own message: "LEAD_IN_RANGE is carried over unchanged... the
           pre-consolidation value"). No known ceiling figure in the docs cites this exact
           1-minute-lived intermediate state directly (9061392 landed 51 seconds later), but
           it is included for completeness since step 1's task explicitly asks for "every
           script version."
  CURRENT -- src/swarm_intent/eval_trajectories.py at HEAD (since 9061392, symmetrized
           LEAD_IN_RANGE=(30,50)). This is the code phase0_ceiling.py/phase0_threat_ceiling.py
           import from TODAY, and the code underlying every "current state" figure in
           CEILING.md from 2026-08-10 onward (chain-1 87.6%/85.8%, chain-2 77.2%/66.7%, the
           83.0%/77.3% pooled bridge figure).

generate_transition_sequence() itself (swarm_intent.data) is reused unmodified across all
three -- its signature has not changed since before OLD was written (verified: git log shows
no commits to data.py's generate_transition_sequence in this window), so all three versions
call the SAME underlying physics function; only the SAMPLING of blend_start/blend_end/seg_len
around it differs.

Usage:
    python scripts/rule0_2b_regenerate_populations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.data import generate_transition_sequence  # noqa: E402

SEED = 999
N = 1000
OUT_DIR = REPO / "eval_data"


# ---------------------------------------------------------------------------
# OLD: verbatim from `git show 628dc56:scripts/phase0_ceiling.py` (identical since b680c80).
# Produced every phase0_ceiling*.json / phase0_threat_ceiling*.json version tagged
# v2/v3/v4/v4_robust/v5/v5_domfix/v5_guardfix/v5_oovfix/v5_trimfix/v6.
# ---------------------------------------------------------------------------
def sample_chain_OLD(rng: np.random.Generator) -> list[str]:
    num_formations = int(rng.integers(1, 5))
    chain = [rng.choice(BASE_FORMATIONS)]
    for _ in range(num_formations - 1):
        pool = [f for f in BASE_FORMATIONS if f != chain[-1]]
        chain.append(rng.choice(pool))
    return chain


def build_long_sequence_labeled_OLD(chain, rng, spread, noise_std):
    segments, seg_labels = [], []
    if len(chain) == 1:
        seg_len = int(rng.integers(50, 101))
        seg = generate_transition_sequence(chain[0], chain[0], n_timesteps=seg_len,
                                           spread=spread, noise_std=noise_std, rng=rng)
        segments.append(seg)
        seg_labels.append([chain[0]] * seg_len)
    else:
        for i in range(len(chain) - 1):
            seg_len = int(rng.integers(50, 101))
            blend_start = int(seg_len * rng.uniform(0.3, 0.5))
            blend_end = int(seg_len * rng.uniform(0.55, 0.75))
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
    return long_seq, true_labels


# ---------------------------------------------------------------------------
# MID: verbatim from `git show 3591051:src/swarm_intent/eval_trajectories.py`.
# Dwell-time-fixed sampling, LEAD_IN_RANGE=(15,35) (pre-symmetrization).
# ---------------------------------------------------------------------------
MID_LEAD_IN_RANGE = (15, 35)
MID_BLEND_DURATION_RANGE = (10, 25)
MID_MIN_DWELL_RANGE = (40, 60)


def sample_chain_MID(rng: np.random.Generator) -> list[str]:
    num_formations = int(rng.integers(1, 5))
    chain = [rng.choice(BASE_FORMATIONS)]
    for _ in range(num_formations - 1):
        pool = [f for f in BASE_FORMATIONS if f != chain[-1]]
        chain.append(rng.choice(pool))
    return chain


def build_long_sequence_labeled_MID(chain, rng, spread, noise_std):
    segments, seg_labels = [], []
    if len(chain) == 1:
        seg_len = int(rng.integers(50, 101))
        seg = generate_transition_sequence(chain[0], chain[0], n_timesteps=seg_len,
                                           spread=spread, noise_std=noise_std, rng=rng)
        segments.append(seg)
        seg_labels.append([chain[0]] * seg_len)
    else:
        for i in range(len(chain) - 1):
            lead_in = int(rng.integers(*MID_LEAD_IN_RANGE))
            blend_duration = int(rng.integers(*MID_BLEND_DURATION_RANGE))
            dwell = int(rng.integers(*MID_MIN_DWELL_RANGE))
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
    return long_seq, true_labels


# ---------------------------------------------------------------------------
# CURRENT: import live from src/swarm_intent/eval_trajectories.py (HEAD).
# ---------------------------------------------------------------------------
from swarm_intent.eval_trajectories import (  # noqa: E402
    sample_chain as sample_chain_CURRENT,
    build_long_sequence_labeled as build_long_sequence_labeled_CURRENT,
)

VERSIONS = {
    "OLD_pre_dwell_fix_b680c80_628dc56": (sample_chain_OLD, build_long_sequence_labeled_OLD),
    "MID_post_consolidation_3591051": (sample_chain_MID, build_long_sequence_labeled_MID),
    "CURRENT_post_symmetrize_9061392_HEAD": (sample_chain_CURRENT, build_long_sequence_labeled_CURRENT),
}


def generate_population(sample_chain_fn, build_fn):
    rng = np.random.default_rng(SEED)
    records = []
    for i in range(N):
        chain = sample_chain_fn(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_fn(chain, rng, spread, noise_std)
        records.append({
            "i": i,
            "chain": chain,
            "spread": round(spread, 8),
            "noise_std": round(noise_std, 8),
            "n_timesteps": long_seq.shape[0],
            "true_labels": true_labels,
            "positions": np.round(long_seq, 6).tolist(),
        })
    return records


def main():
    OUT_DIR.mkdir(exist_ok=True)
    manifest = {}
    for version_name, (sc_fn, bl_fn) in VERSIONS.items():
        print(f"=== generating {version_name} (seed={SEED}, n={N}) ===")
        records = generate_population(sc_fn, bl_fn)
        out_path = OUT_DIR / f"locked_seed999_{version_name}.json"
        out_path.write_text(json.dumps({
            "version": version_name, "seed": SEED, "n": N, "records": records,
        }))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  saved {out_path} ({size_mb:.1f} MB)")
        manifest[version_name] = str(out_path.relative_to(REPO))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nsaved {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
