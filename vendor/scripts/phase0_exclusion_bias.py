"""
Step 1 of the 2026-08-10 pre-scaling checks (docs/V5_LOG.md step 35): characterizes the
combined port's 51.2% window-exclusion rate (step 34) -- is it uniform across
(own-content-fraction, blend-fraction) space, and is any (a,b) formation pair or single
formation systematically starved relative to its share of KEPT windows?

Same seed (7) and n_transition (300) as step 34 -- a direct breakdown of that exact run, not
a new sample. Uses diag["excluded_examples"] and diag["examples"]'s formation-pair fields
(both added this turn to generate_dataset's return_diagnostics output).

Usage:
    python scripts/phase0_exclusion_bias.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swarm_intent.config import Config, BASE_FORMATIONS  # noqa: E402
from swarm_intent.data import generate_dataset  # noqa: E402


def main():
    cfg = Config(seed=7)
    X, y, names, diag = generate_dataset(
        cfg, n_per_formation=0, n_timesteps=50, include_transitions=True, n_transition=300,
        corrected_blend_timing=True, windowed_examples=True, content_majority_labeling=True,
        return_diagnostics=True,
    )
    kept = diag["examples"]
    excluded = diag["excluded_examples"]
    total = len(kept) + len(excluded)
    print(f"kept={len(kept)}  excluded={len(excluded)}  exclusion_rate={len(excluded)/total:.1%}")

    print("\n" + "=" * 100)
    print("WHERE excluded windows sit in (own-content-fraction, blend-fraction) space")
    print("=" * 100)
    own_fracs = np.array([max(f[0], f[2]) for f in excluded])
    blend_fracs = np.array([f[1] for f in excluded])
    print(f"max(frac_a,frac_b): min={own_fracs.min():.3f} p25={np.percentile(own_fracs,25):.3f} "
         f"median={np.median(own_fracs):.3f} p75={np.percentile(own_fracs,75):.3f} max={own_fracs.max():.3f}")
    print(f"frac_blend:         min={blend_fracs.min():.3f} p25={np.percentile(blend_fracs,25):.3f} "
         f"median={np.median(blend_fracs):.3f} p75={np.percentile(blend_fracs,75):.3f} max={blend_fracs.max():.3f}")

    own_bins = np.arange(0.0, 0.71, 0.1)  # excluded windows are all <0.70 by construction
    blend_bins = np.arange(0.0, 0.51, 0.1)
    hist, _, _ = np.histogram2d(own_fracs, blend_fracs, bins=[own_bins, blend_bins])
    print("\n2D histogram, max(frac_a,frac_b) row x frac_blend column (n=%d):" % len(excluded))
    print("own\\blend " + "".join(f"[{b:.1f},{b+0.1:.1f})".rjust(11) for b in blend_bins[:-1]))
    for i in range(len(own_bins) - 1):
        row = f"[{own_bins[i]:.1f},{own_bins[i+1]:.1f})".ljust(10)
        row += "".join(f"{int(hist[i,j]):>11d}" for j in range(len(blend_bins) - 1))
        print(row)
    print("(concentration near own_frac in [0.5,0.7) confirms exclusions are near-miss pure")
    print(" windows -- close to confident but not quite -- not uniformly spread across the space)")

    print("\n" + "=" * 100)
    print("BY FORMATION PAIR (a,b): excluded share vs kept share, and single-formation totals")
    print("=" * 100)
    kept_pairs = Counter((f[4], f[5]) for f in kept)
    excl_pairs = Counter((f[3], f[4]) for f in excluded)
    all_pairs = sorted(set(kept_pairs) | set(excl_pairs))
    print(f"{'pair':<28} {'kept':>6} {'excl':>6} {'excl_rate':>10}")
    rates = []
    for pair in all_pairs:
        k, e = kept_pairs.get(pair, 0), excl_pairs.get(pair, 0)
        rate = e / (k + e) if (k + e) else float("nan")
        rates.append((pair, k, e, rate))
        print(f"{str(pair):<28} {k:>6} {e:>6} {rate:>9.1%}")
    rates_only = np.array([r[3] for r in rates if (r[1] + r[2]) > 0])
    print(f"\nper-pair exclusion rate: min={rates_only.min():.1%} mean={rates_only.mean():.1%} "
         f"max={rates_only.max():.1%} std={rates_only.std():.1%}")
    worst = max(rates, key=lambda r: r[3])
    best = min(rates, key=lambda r: r[3])
    print(f"most-excluded pair: {worst[0]} ({worst[3]:.1%})  least-excluded pair: {best[0]} ({best[3]:.1%})")

    print("\nPer single formation (appearing as EITHER a or b), kept vs excluded counts:")
    kept_f = Counter()
    excl_f = Counter()
    for f in kept:
        kept_f[f[4]] += 1
        kept_f[f[5]] += 1
    for f in excluded:
        excl_f[f[3]] += 1
        excl_f[f[4]] += 1
    print(f"{'formation':<15} {'kept':>6} {'excl':>6} {'excl_rate':>10}")
    frates = []
    for formation in BASE_FORMATIONS:
        k, e = kept_f.get(formation, 0), excl_f.get(formation, 0)
        rate = e / (k + e) if (k + e) else float("nan")
        frates.append(rate)
        print(f"{formation:<15} {k:>6} {e:>6} {rate:>9.1%}")
    frates_arr = np.array(frates)
    print(f"\nper-formation exclusion rate spread: min={frates_arr.min():.1%} max={frates_arr.max():.1%} "
         f"std={frates_arr.std():.1%}")


if __name__ == "__main__":
    main()
