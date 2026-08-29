"""
Quantifies what "porting the dwell-time/symmetrization fix into generate_dataset()" would
actually change, as its own scoped decision (not folded into a numbered V5 strategy) --
see HISTORY.md's corresponding entry for the full reasoning.

TRAIN: src/swarm_intent/data.py generate_dataset()'s CURRENT 3-regime blend timing, the
       actual distribution STGT strategy 1-6 checkpoints were all trained on (n_timesteps
       fixed at 50 -- this IS the training example, not something later windowed).
       Mirrored by scripts/phase0_chain2_blend_distributions.py's train_regime_fractions(),
       verified line-for-line identical to data.py's current regime 0/1/2 formulas.

EVAL:  src/swarm_intent/eval_trajectories.py's CURRENT (both-fixes) per-hop blend timing --
       LEAD_IN_RANGE/BLEND_DURATION_RANGE/MIN_DWELL_RANGE imported live, not duplicated, so
       this can never silently drift from what phase0_ceiling.py etc. actually use.

Both expressed as (start_frac, duration_frac) of their own segment length, the established
convention in this codebase for comparing across differing absolute segment lengths
(train's fixed 50-timestep window vs eval's derived 80-135-timestep hop).

No GPU, no dataset regeneration, no retraining -- pure Monte Carlo over the two real formulas.

Usage:
    python scripts/phase0_train_eval_blend_divergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase0_chain2_blend_distributions import train_regime_fractions  # noqa: E402
from swarm_intent.eval_trajectories import LEAD_IN_RANGE, BLEND_DURATION_RANGE, MIN_DWELL_RANGE  # noqa: E402


def eval_hop_fractions(n_draws=20000, seed=2):
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
    eval_ = eval_hop_fractions()

    print("=" * 100)
    print(f"EVAL (current, live from eval_trajectories.py): LEAD_IN_RANGE={LEAD_IN_RANGE}, "
         f"BLEND_DURATION_RANGE={BLEND_DURATION_RANGE}, MIN_DWELL_RANGE={MIN_DWELL_RANGE}")
    print("=" * 100)
    eval_starts = np.array([f[0] for f in eval_])
    eval_durs = np.array([f[2] for f in eval_])
    seg_lens = [f[3] for f in eval_]
    print(f"  start_frac: [{eval_starts.min():.3f}, {eval_starts.max():.3f}]  "
         f"duration_frac: [{eval_durs.min():.3f}, {eval_durs.max():.3f}]")
    print(f"  seg_len (absolute timesteps): min={min(seg_lens)} mean={np.mean(seg_lens):.1f} max={max(seg_lens)}")
    print(f"  (train's fixed example length is 50 timesteps -- eval's hop is "
         f"{min(seg_lens)/50:.1f}x-{max(seg_lens)/50:.1f}x longer)")

    print()
    print("=" * 100)
    print("TRAIN (current, generate_dataset() regime 0/1/2, n_timesteps=50 fixed): per-regime realized box")
    print("=" * 100)
    regime_boxes = {}
    regime_names = {0: "regime 0 (-> labeled pure formation_a)",
                    1: "regime 1 (-> labeled 'transitioning')",
                    2: "regime 2 (-> labeled pure formation_b)"}
    for regime in (0, 1, 2):
        starts = np.array([f[0] for f in train[regime]])
        durs = np.array([f[2] for f in train[regime]])
        regime_boxes[regime] = (starts.min(), starts.max(), durs.min(), durs.max())
        n_train_total = sum(len(train[rr]) for rr in (0, 1, 2))
        print(f"  {regime_names[regime]}: start_frac=[{starts.min():.3f},{starts.max():.3f}] "
             f"duration_frac=[{durs.min():.3f},{durs.max():.3f}]  (~{len(train[regime])/n_train_total:.1%} of transitioning examples)")

    print()
    print("=" * 100)
    print("DIVERGENCE: fraction of EVAL windows whose (start_frac, duration_frac) falls inside")
    print("ANY current TRAIN regime's realized box (0% = the regime is entirely unrepresented)")
    print("=" * 100)
    n_in_any = 0
    n_in_regime = {0: 0, 1: 0, 2: 0}
    for s, d in zip(eval_starts, eval_durs):
        hit_any = False
        for r in (0, 1, 2):
            if regime_boxes[r][0] <= s <= regime_boxes[r][1] and regime_boxes[r][2] <= d <= regime_boxes[r][3]:
                n_in_regime[r] += 1
                hit_any = True
        if hit_any:
            n_in_any += 1
    n = len(eval_)
    print(f"  in regime 0's box: {n_in_regime[0]}/{n} ({n_in_regime[0]/n:.1%})")
    print(f"  in regime 1's box: {n_in_regime[1]}/{n} ({n_in_regime[1]/n:.1%})")
    print(f"  in regime 2's box: {n_in_regime[2]}/{n} ({n_in_regime[2]/n:.1%})")
    print(f"  in ANY regime's box: {n_in_any}/{n} ({n_in_any/n:.1%})")
    print()
    print(f"  => if generate_dataset() were to sample blend timing the way eval_trajectories.py")
    print(f"     currently does, {(n-n_in_any)/n:.1%} of the resulting examples would exhibit a")
    print(f"     (start_frac, duration_frac) shape that NONE of the current 3 regimes ever produce.")


if __name__ == "__main__":
    main()
