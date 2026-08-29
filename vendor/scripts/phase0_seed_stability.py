"""
Step 2 of the 2026-08-10 pre-scaling checks (docs/V5_LOG.md step 36): does step 34's ~51%
exclusion rate (single seed=7) generalize, or was that one seed lucky/unlucky? Re-runs the
combined-port generation at 5 seeds total and reports the keep/exclude fraction per seed plus
mean/std, so the "~2x n_transition" compensation estimate is grounded in a distribution, not
one sample.

Seeds: 7 (step 34/35's, reused for continuity) + 4 fresh ones (8, 9, 10, 11 -- simple,
disjoint from every seed already in use elsewhere in this program: 0, 1, 42, 999, 2024).

Usage:
    python scripts/phase0_seed_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset  # noqa: E402

SEEDS = [7, 8, 9, 10, 11]
N_TRANSITION = 300


def main():
    rows = []
    for seed in SEEDS:
        cfg = Config(seed=seed)
        X, y, names, diag = generate_dataset(
            cfg, n_per_formation=0, n_timesteps=50, include_transitions=True,
            n_transition=N_TRANSITION, corrected_blend_timing=True, windowed_examples=True,
            content_majority_labeling=True, return_diagnostics=True,
        )
        total = diag["n_examples_kept"] + diag["n_excluded"]
        keep_rate = diag["n_examples_kept"] / total
        rows.append((seed, diag["n_hops_sampled"], total, diag["n_examples_kept"],
                    diag["n_excluded"], keep_rate))

    print(f"{'seed':>6} {'hops':>6} {'total_windows':>14} {'kept':>6} {'excluded':>9} {'keep_rate':>10}")
    for seed, hops, total, kept, excl, rate in rows:
        print(f"{seed:>6} {hops:>6} {total:>14} {kept:>6} {excl:>9} {rate:>9.1%}")

    rates = np.array([r[5] for r in rows])
    print(f"\nkeep_rate across {len(SEEDS)} seeds: mean={rates.mean():.1%} std={rates.std():.1%} "
         f"min={rates.min():.1%} max={rates.max():.1%}")

    exclusion_rates = 1 - rates
    print(f"exclusion_rate across {len(SEEDS)} seeds: mean={exclusion_rates.mean():.1%} "
         f"std={exclusion_rates.std():.1%} min={exclusion_rates.min():.1%} max={exclusion_rates.max():.1%}")

    # Compensation factor: to get N final kept examples, need to request N / keep_rate hops
    # (using windows-per-hop from this same population, since windowed_examples means
    # n_transition = hops, not final examples).
    windows_per_hop = np.array([r[2] / r[1] for r in rows])
    compensation = 1.0 / (rates * 1.0)  # multiply target KEPT-window count by this to get
                                        # target windows; combined with windows_per_hop to
                                        # get target hops
    print(f"\nwindows_per_hop across seeds: mean={windows_per_hop.mean():.2f} std={windows_per_hop.std():.2f}")
    worst_case_keep_rate = rates.min()
    print(f"\nRECOMMENDED compensation factor (using worst-case seed's keep_rate, not the "
         f"mean, for a conservative safety margin): 1/{worst_case_keep_rate:.3f} = "
         f"{1/worst_case_keep_rate:.2f}x")
    print(f"(previous single-seed estimate from step 34 was ~2x; revise to "
         f"{1/worst_case_keep_rate:.2f}x if this differs meaningfully)")


if __name__ == "__main__":
    main()
