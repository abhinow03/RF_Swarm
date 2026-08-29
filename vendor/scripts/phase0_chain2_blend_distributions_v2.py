"""
Step 7 of the 2026-08-09 chain-2 generator-fix experiment: after the observability fix
(src/swarm_intent/eval_trajectories.py's build_long_sequence_labeled, LEAD_IN_RANGE/
BLEND_DURATION_RANGE/MIN_DWELL_RANGE), does eval's blend timing still miss every training
regime the way scripts/phase0_chain2_blend_distributions.py found pre-fix (0.0% overlap)?

Same methodology as that script (Monte Carlo, mirrors the exact sampling formulas, no GPU),
just re-expressing the decoupled lead_in/blend_duration/dwell sampling as (start_frac,
duration_frac) of the DERIVED seg_len so it's directly comparable to the training regimes'
realized boxes on the same axes.

2026-08-09 fix (found while auditing step 26's "eval v3 (both fixes)" claim in V5_LOG.md/
HISTORY.md): this script previously hardcoded its own copy of LEAD_IN_RANGE=(15,35) -- the
step-25 (dest-only-fix) value -- and was never updated when step 26 symmetrized the range to
(30,50) in the new consolidated module. The documented "eval v3 (both fixes)" row (start_frac
[0.265,0.495], duration_frac [0.085,0.255], 0.0% overlap) was real and independently
reproduced when re-derived with (30,50), but this committed script could not itself reproduce
it -- running it as-is silently regenerated the superseded "eval v2" numbers instead. Now
imports the three ranges directly from eval_trajectories.py (the single canonical source
since step 26) instead of a hand-duplicated copy, so this can't silently drift again.

Usage:
    python scripts/phase0_chain2_blend_distributions_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase0_chain2_blend_distributions import train_regime_fractions, summarize
from swarm_intent.eval_trajectories import LEAD_IN_RANGE, BLEND_DURATION_RANGE, MIN_DWELL_RANGE  # noqa: E402


def eval_hop_fractions_v2(n_draws=20000, seed=2):
    """Mirrors src/swarm_intent/eval_trajectories.py's build_long_sequence_labeled chain>=2
    sampling exactly (lead_in/blend_duration/dwell sampled directly, seg_len derived), using
    that module's LIVE range constants -- always reflects whatever is currently canonical."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        lead_in = int(rng.integers(*LEAD_IN_RANGE))
        blend_duration = int(rng.integers(*BLEND_DURATION_RANGE))
        dwell = int(rng.integers(*MIN_DWELL_RANGE))
        blend_start = lead_in
        blend_end = lead_in + blend_duration
        seg_len = blend_end + dwell
        out.append((blend_start / seg_len, blend_end / seg_len, blend_duration / seg_len, seg_len))
    return out


def main():
    train = train_regime_fractions()
    eval_v2 = eval_hop_fractions_v2()

    print("=" * 100)
    print(f"EVAL (current canonical, LEAD_IN_RANGE={LEAD_IN_RANGE}): per-hop blend timing "
         f"as fraction of DERIVED seg_len")
    print("=" * 100)
    summarize("eval v2 per-hop blend", eval_v2)
    seg_lens = [f[3] for f in eval_v2]
    print(f"  seg_len (derived): min={min(seg_lens)} mean={np.mean(seg_lens):.1f} max={max(seg_lens)}")

    regime_boxes = {}
    for regime in (0, 1, 2):
        starts = [f[0] for f in train[regime]]
        durs = [f[2] for f in train[regime]]
        regime_boxes[regime] = (min(starts), max(starts), min(durs), max(durs))
        print(f"  train regime {regime} realized box: start_frac=[{min(starts):.3f},{max(starts):.3f}], "
             f"duration_frac=[{min(durs):.3f},{max(durs):.3f}]")

    eval_starts = np.array([f[0] for f in eval_v2])
    eval_durs = np.array([f[2] for f in eval_v2])
    n_in_any_regime = 0
    for s, d in zip(eval_starts, eval_durs):
        in_any = any(regime_boxes[r][0] <= s <= regime_boxes[r][1] and
                    regime_boxes[r][2] <= d <= regime_boxes[r][3] for r in (0, 1, 2))
        n_in_any_regime += in_any
    pct = n_in_any_regime / len(eval_v2)
    print(f"\nFraction of NEW eval per-hop blends whose (start_frac, duration_frac) falls inside "
         f"ANY training regime's realized box: {n_in_any_regime}/{len(eval_v2)} ({pct:.1%})")


if __name__ == "__main__":
    main()
