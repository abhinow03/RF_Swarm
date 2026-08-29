"""
Step 4 of the chain-2 observability diagnostic (docs/UPSTREAM_ISSUES.md issue #3
follow-up): does STGT's training data ever teach it a blend shaped like the ones
chain-2 EVALUATION trajectories actually contain?

Pure Monte Carlo over the two formulas (no model, no GPU, no dataset regeneration --
just re-implements the exact arithmetic already in the two source functions and
samples each many times):

  TRAIN:  src/swarm_intent/data.py generate_dataset()'s 3-regime transitioning-example
          blend timing, n_timesteps FIXED at 50 (train_model.py's actual call).
  EVAL:   llm_finetuning/measure_coverage.py / scripts/phase0_decompose_failures.py
          build_long_sequence()'s per-hop blend timing, seg_len ~ Uniform{50..100}.

Both are expressed as fractions of their own segment length (n_timesteps for train,
seg_len for eval) so they're directly comparable regardless of the differing absolute
segment lengths.

Usage:
    python scripts/phase0_chain2_blend_distributions.py
"""
from __future__ import annotations

import numpy as np


def train_regime_fractions(n_draws=20000, n_timesteps=50, seed=1):
    """Mirrors src/swarm_intent/data.py generate_dataset()'s regime 0/1/2 exactly."""
    rng = np.random.default_rng(seed)
    margin = max(1, int(0.10 * n_timesteps))
    out = {0: [], 1: [], 2: []}
    for _ in range(n_draws):
        regime = int(rng.integers(0, 3))
        if regime == 0:
            blend_start = int(n_timesteps * rng.uniform(0.74, 0.90))
            blend_end = int(np.clip(blend_start + n_timesteps * rng.uniform(0.05, 0.10),
                                    blend_start + margin, n_timesteps - 1))
        elif regime == 1:
            blend_start = int(n_timesteps * rng.uniform(0.12, 0.30))
            blend_end = int(np.clip(blend_start + n_timesteps * rng.uniform(0.45, 0.62),
                                    blend_start + margin, n_timesteps - margin))
        else:
            blend_end = int(n_timesteps * rng.uniform(0.10, 0.26))
            blend_start = int(np.clip(blend_end - n_timesteps * rng.uniform(0.05, 0.10),
                                      1, blend_end - margin))
        blend_start = max(1, min(blend_start, n_timesteps - 2))
        blend_end = max(blend_start + 1, min(blend_end, n_timesteps - 1))
        out[regime].append((blend_start / n_timesteps, blend_end / n_timesteps,
                           (blend_end - blend_start) / n_timesteps))
    return out


def eval_hop_fractions(n_draws=20000, seed=2):
    """Mirrors llm_finetuning/measure_coverage.py / phase0_decompose_failures.py
    build_long_sequence()'s per-hop blend timing exactly (chain length >= 2 case)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        seg_len = int(rng.integers(50, 101))
        blend_start = int(seg_len * rng.uniform(0.3, 0.5))
        blend_end = int(seg_len * rng.uniform(0.55, 0.75))
        out.append((blend_start / seg_len, blend_end / seg_len, (blend_end - blend_start) / seg_len, seg_len))
    return out


def summarize(label, fracs, idx_start=0, idx_end=1, idx_dur=2):
    starts = [f[idx_start] for f in fracs]
    ends = [f[idx_end] for f in fracs]
    durs = [f[idx_dur] for f in fracs]
    print(f"{label} (n={len(fracs)}):")
    print(f"  blend START  fraction: min={min(starts):.3f} p25={np.percentile(starts,25):.3f} "
         f"mean={np.mean(starts):.3f} p75={np.percentile(starts,75):.3f} max={max(starts):.3f}")
    print(f"  blend END    fraction: min={min(ends):.3f} p25={np.percentile(ends,25):.3f} "
         f"mean={np.mean(ends):.3f} p75={np.percentile(ends,75):.3f} max={max(ends):.3f}")
    print(f"  blend DURATION fraction: min={min(durs):.3f} p25={np.percentile(durs,25):.3f} "
         f"mean={np.mean(durs):.3f} p75={np.percentile(durs,75):.3f} max={max(durs):.3f}")


def main():
    train = train_regime_fractions()
    eval_ = eval_hop_fractions()

    print("=" * 100)
    print("TRAIN: generate_dataset() transitioning-example blend timing (n_timesteps=50 fixed)")
    print("=" * 100)
    for regime in (0, 1, 2):
        label_map = {0: "regime 0 (blend late/short -> labeled pure formation_a)",
                    1: "regime 1 (blend dominates -> labeled 'transitioning')",
                    2: "regime 2 (blend early/short -> labeled pure formation_b)"}
        summarize(label_map[regime], train[regime])
        print()

    print("=" * 100)
    print("EVAL: build_long_sequence() per-hop blend timing (seg_len ~ Uniform{50..100})")
    print("=" * 100)
    summarize("eval per-hop blend", eval_)
    seg_lens = [f[3] for f in eval_]
    print(f"  seg_len: min={min(seg_lens)} mean={np.mean(seg_lens):.1f} max={max(seg_lens)}")

    print("\n" + "=" * 100)
    print("OVERLAP CHECK: does eval's (start_frac, duration_frac) region fall inside ANY")
    print("train regime's (start_frac, duration_frac) region?")
    print("=" * 100)
    # bounding boxes actually sampled per regime (not the raw uniform() call bounds --
    # post-clip, as realized)
    regime_boxes = {}
    for regime in (0, 1, 2):
        starts = [f[0] for f in train[regime]]
        durs = [f[2] for f in train[regime]]
        regime_boxes[regime] = (min(starts), max(starts), min(durs), max(durs))
        print(f"  train regime {regime} realized box: "
             f"start_frac=[{min(starts):.3f},{max(starts):.3f}], "
             f"duration_frac=[{min(durs):.3f},{max(durs):.3f}]")

    eval_starts = np.array([f[0] for f in eval_])
    eval_durs = np.array([f[2] for f in eval_])
    n_in_any_regime = 0
    for s, d in zip(eval_starts, eval_durs):
        in_any = any(regime_boxes[r][0] <= s <= regime_boxes[r][1] and
                    regime_boxes[r][2] <= d <= regime_boxes[r][3] for r in (0, 1, 2))
        n_in_any_regime += in_any
    pct = n_in_any_regime / len(eval_)
    print(f"\nFraction of eval per-hop blends whose (start_frac, duration_frac) falls inside "
         f"ANY training regime's realized box: {n_in_any_regime}/{len(eval_)} ({pct:.1%})")
    print("(a LOW number here means the eval blend-timing shape is essentially unrepresented")
    print(" in what generate_dataset() ever labels and trains STGT on)")


if __name__ == "__main__":
    main()
