"""
V5 program, Phase 0, step 4 of the discipline-catch turn (docs/V5_LOG.md):
491/1000 trajectories in the standing ceiling battery are chain-length-3+ --
no RULES key exists for them, and they are silently excluded from every
pair-level/threat-level ceiling number reported so far. REPORT ONLY, per
instruction -- does NOT touch RULES (extending it is HALT GATE 2, needs
Dr. Patil's sign-off, not done here).

Answers three questions:
  (a) the exact chain-length distribution the generator produces
  (b) is that distribution a generator PARAMETER we set, or an EMERGENT
      property of the sampling -- read directly from sample_chain()'s source
  (c) how many DISTINCT chain-3+ patterns appear in the standing battery, and
      whether they collapse to a small, reusable set (relevant to how big a
      RULES-extension effort HALT GATE 2 would actually be, without doing it)

Replays the EXACT same seed=999 rng consumption as scripts/phase0_ceiling.py
(sample_chain, then spread/noise_std draws, then the full build_long_sequence_
labeled call) so the chain population is IDENTICAL to the standing ceiling
battery -- but never runs STGT inference (no GPU, no model load), since
nothing in this report depends on model output at all, only on the generator.

Usage:
    python scripts/phase0_rules_coverage_report.py --n 1000
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from phase0_ceiling import sample_chain, build_long_sequence_labeled  # noqa: E402

SEED = 999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("=== (b) is the chain-length distribution a generator PARAMETER or emergent? ===")
    print("Source of sample_chain() (scripts/phase0_ceiling.py):")
    print(inspect.getsource(sample_chain))
    print("`num_formations = rng.integers(1, 5)` is a HARD-CODED PARAMETER (the bounds 1..5, "
         "numpy's default endpoint=False -> uniform over {1,2,3,4}) -- an explicit generator "
         "design choice, not something that emerges from any other property of the sampling. "
         "Every subsequent formation is drawn uniformly from BASE_FORMATIONS minus only the "
         "immediately-preceding one (no consecutive repeats forced, non-consecutive repeats "
         "ARE allowed -- e.g. A->B->A is a legal length-3 chain).")

    rng = np.random.default_rng(args.seed)
    chains = []
    for i in range(args.n):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        build_long_sequence_labeled(chain, rng, spread, noise_std)  # consume rng identically; discard sequence
        chains.append(chain)

    print(f"\n=== (a) exact chain-length distribution, n={args.n}, seed={args.seed} "
         f"(identical population to the standing ceiling battery) ===")
    length_counts = Counter(len(c) for c in chains)
    print("| chain_length | n | % | theoretical (uniform 1-4) |")
    print("|---|---|---|---|")
    for length in (1, 2, 3, 4):
        k = length_counts.get(length, 0)
        print(f"| {length} | {k} | {k/args.n:.1%} | 25.0% |")
    n_3plus = length_counts.get(3, 0) + length_counts.get(4, 0)
    print(f"| 3+ (combined) | {n_3plus} | {n_3plus/args.n:.1%} | 50.0% |")

    print(f"\n=== (c) distinct chain-3+ patterns: do they collapse to a small set? ===")
    chains_3plus = [tuple(c) for c in chains if len(c) >= 3]
    n_3plus_actual = len(chains_3plus)
    pattern_counts = Counter(chains_3plus)
    n_distinct = len(pattern_counts)
    print(f"n chain-3+ trajectories: {n_3plus_actual}")
    print(f"n DISTINCT chain-3+ patterns observed: {n_distinct}")
    print(f"n patterns seen more than once: {sum(1 for c in pattern_counts.values() if c > 1)}")
    print(f"max times any single pattern repeats: {max(pattern_counts.values()) if pattern_counts else 0}")
    print(f"fraction of chain-3+ trajectories with a UNIQUE (never-repeated) pattern: "
         f"{sum(1 for c in pattern_counts.values() if c == 1) / n_3plus_actual:.1%}"
         if n_3plus_actual else "n/a")

    # theoretical size of the pattern space, for context
    n_base = 7  # len(BASE_FORMATIONS)
    n_len3 = n_base * (n_base - 1) * (n_base - 1)
    n_len4 = n_base * (n_base - 1) * (n_base - 1) * (n_base - 1)
    print(f"\ntheoretical distinct-pattern space: {n_len3} (length 3) + {n_len4} (length 4) "
         f"= {n_len3 + n_len4} possible chains (7 base formations, no consecutive repeats)")
    print(f"observed {n_distinct} distinct patterns from {n_3plus_actual} draws -- "
         f"{'a small, reusable set' if n_distinct < n_3plus_actual * 0.5 else 'NOT a small set -- overwhelmingly unique, no meaningful collapse'}")

    print(f"\nmost common chain-3+ patterns (top 10):")
    for pattern, count in pattern_counts.most_common(10):
        print(f"  {' -> '.join(pattern)}: {count}")

    out = {"seed": args.seed, "n": args.n, "length_distribution": dict(length_counts),
          "n_chain_3plus": n_3plus_actual, "n_distinct_patterns": n_distinct,
          "pattern_counts": {" -> ".join(k): v for k, v in pattern_counts.items()},
          "theoretical_pattern_space": {"length_3": n_len3, "length_4": n_len4}}
    out_path = REPO / "evaluation" / "phase0_rules_coverage_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    print("\nNo RULES change made -- report only, per instruction. Extending RULES is HALT GATE "
         "2 and needs Dr. Patil's sign-off.")


if __name__ == "__main__":
    main()
