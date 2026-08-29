"""
Step 4 of the 2026-08-10 generator-port design session (docs/V5_LOG.md steps 31-34):
numerically sanity-checks the three toggleable generate_dataset() flags
(corrected_blend_timing, windowed_examples, content_majority_labeling) by generating four
SMALL diagnostic datasets -- baseline, each flag alone, and all three combined -- and
verifying labels are sane, not just inspecting them.

For every kept example: does the assigned label agree with a PLURALITY of its own
timestep-level content (argmax of frac_a/frac_blend/frac_b)? For content_majority_labeling
runs this is guaranteed true by construction (_label_window IS a plurality-consistent rule);
reported anyway, unconditionally, as an actual check rather than an assumption -- and
specifically expected to find real disagreement in the BASELINE run, which is the whole
reason this porting work exists (the old regime label is chosen at generation time, before
noise/edge-clipping can shift realized content, and is never reconciled against it).

No GPU, no dataset regeneration at training scale -- n_transition=300 per run.

Usage:
    python scripts/phase0_generator_port_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset  # noqa: E402

RUNS = [
    ("baseline (all off, current behaviour)", dict(corrected_blend_timing=False, windowed_examples=False, content_majority_labeling=False)),
    ("blend-timing only", dict(corrected_blend_timing=True, windowed_examples=False, content_majority_labeling=False)),
    ("windowing only", dict(corrected_blend_timing=False, windowed_examples=True, content_majority_labeling=False)),
    ("labeling only", dict(corrected_blend_timing=False, windowed_examples=False, content_majority_labeling=True)),
    ("all three combined (the actual proposed port)", dict(corrected_blend_timing=True, windowed_examples=True, content_majority_labeling=True)),
]

N_TRANSITION = 300


def analyze(name, examples, n_hops, n_excluded):
    print("=" * 100)
    print(f"{name}  (n_hops_sampled={n_hops}, n_kept={len(examples)}, n_excluded={n_excluded}, "
         f"exclusion_rate={n_excluded/(n_excluded+len(examples)):.1%})" if (n_excluded + len(examples)) else name)
    print("=" * 100)
    by_label = {}
    for frac_a, frac_blend, frac_b, lbl, _f_a, _f_b in examples:
        by_label.setdefault(lbl, []).append((frac_a, frac_blend, frac_b))

    n_disagree = 0
    for lbl, rows in sorted(by_label.items()):
        own_frac = {"pure_a": 0, "pure_b": 2, "transitioning": 1}[lbl]
        matching = np.array([r[own_frac] for r in rows])
        print(f"  {lbl}: n={len(rows)}  own-content-fraction: min={matching.min():.3f} "
             f"p25={np.percentile(matching,25):.3f} mean={matching.mean():.3f} "
             f"p75={np.percentile(matching,75):.3f} max={matching.max():.3f}")
        for frac_a, frac_blend, frac_b in rows:
            plurality = max((("pure_a", frac_a), ("transitioning", frac_blend), ("pure_b", frac_b)),
                           key=lambda kv: kv[1])[0]
            if plurality != lbl:
                n_disagree += 1
    total = len(examples)
    print(f"  windows where assigned label DISAGREES with plurality of own content: "
         f"{n_disagree}/{total} ({n_disagree/total:.1%})" if total else "  n=0, nothing to check")
    print()
    return n_disagree, total


def main():
    cfg = Config(seed=7)  # fresh seed, not 0/1/42/999/2024 (already-used seeds elsewhere)
    summary = []
    for name, flags in RUNS:
        cfg_i = Config(seed=7)
        X, y, names, diag = generate_dataset(
            cfg_i, n_per_formation=0, n_timesteps=50, include_transitions=True,
            n_transition=N_TRANSITION, return_diagnostics=True, **flags,
        )
        n_disagree, total = analyze(name, diag["examples"], diag["n_hops_sampled"], diag["n_excluded"])
        summary.append((name, X.shape, diag["n_hops_sampled"], diag["n_examples_kept"],
                       diag["n_excluded"], n_disagree, total))

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'run':<45} {'X.shape':<18} {'hops':>6} {'kept':>6} {'excl':>6} {'disagree':>10}")
    for name, shape, hops, kept, excl, dis, tot in summary:
        dis_str = f"{dis}/{tot}" if tot else "n/a"
        print(f"{name:<45} {str(shape):<18} {hops:>6} {kept:>6} {excl:>6} {dis_str:>10}")


if __name__ == "__main__":
    main()
