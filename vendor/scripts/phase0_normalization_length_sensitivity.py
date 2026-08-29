"""
Step 40 of the 2026-08-10 acceleration-cap follow-up (docs/V5_LOG.md step 40): after step 39
found capping generate_transition_sequence's acceleration does NOT resolve the small-scale
training collapse (docs/V5_LOG.md step 37/39), checks whether split_and_normalize's global
position-normalization scalar is a second, independent, length-related contributor.

Reports train_std (the single global scalar) and final-timestep centroid drift distribution
for both the baseline (old format) and corrected (combined port, capped acceleration) small-
scale datasets, same seed=7 population as steps 37/39.

Usage:
    python scripts/phase0_normalization_length_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset, split_and_normalize  # noqa: E402

ORIGIN = np.array([0.0, 0.0, 100.0])


def report(label, X, splits):
    centroids = X.mean(axis=2)  # (N, T, 3)
    drift = np.linalg.norm(centroids[:, -1, :] - ORIGIN, axis=-1)
    print(f"{label}: X_train raw stats mean={X.mean():.2f} std={X.std():.2f} "
         f"min={X.min():.2f} max={X.max():.2f}")
    print(f"{label}: train_mean={splits['train_mean']:.4f} train_std={splits['train_std']:.4f}")
    print(f"{label}: final-timestep centroid drift from origin: min={drift.min():.1f} "
         f"p50={np.percentile(drift,50):.1f} p95={np.percentile(drift,95):.1f} max={drift.max():.1f}")
    print()
    return splits["train_std"]


def main():
    cfg = Config(seed=7)
    Xb, yb, nb = generate_dataset(cfg, n_per_formation=300, n_timesteps=50,
                                  include_transitions=True, n_transition=900)
    sb = split_and_normalize(Xb, yb, cfg)
    std_b = report("BASELINE", Xb, sb)

    cfg2 = Config(seed=7)
    Xc, yc, nc = generate_dataset(
        cfg2, n_per_formation=300, n_timesteps=50, include_transitions=True, n_transition=303,
        corrected_blend_timing=True, windowed_examples=True, content_majority_labeling=True,
    )
    sc = split_and_normalize(Xc, yc, cfg2)
    std_c = report("CORRECTED (capped acceleration)", Xc, sc)

    print(f"train_std ratio corrected/baseline: {std_c/std_b:.2f}x")


if __name__ == "__main__":
    main()
