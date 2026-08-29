"""
Geometric formation templates.

Migrated verbatim from the notebooks, with one fix: the ``dispersed`` and
``converging`` branches now accept an optional seeded ``rng`` so the dataset
is reproducible (the original used ``np.random.default_rng(seed=None)``,
which silently broke the seed=42 promise in the generator).
"""
from __future__ import annotations

import numpy as np

# Tactical meaning per formation (used by the LLM context + visualisations).
FORMATION_INFO = {
    "v_shape":      "Leader-follower coordinated movement",
    "encirclement": "High threat — potential attack / surrounding",
    "column":       "Penetration / trail pattern",
    "diamond":      "Resilient mesh, anti-jamming",
    "dispersed":    "Surveillance / area scanning",
    "converging":   "Closing in — high threat",
    "shield":       "Electromagnetic protection bubble",
    "transitioning": "Changing formation mid-sequence",
}


def get_formation_offsets(formation_type: str, spread: float = 1.0,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a (6, 3) array of per-drone offsets from the swarm centroid.

    Parameters
    ----------
    formation_type : str
    spread : float
        Scaling factor — larger = more dispersed.
    rng : np.random.Generator, optional
        Used only by the random formations (dispersed / converging). Pass a
        seeded generator for reproducible datasets.
    """
    if rng is None:
        rng = np.random.default_rng()

    if formation_type == "v_shape":
        offsets = np.array([
            [0, 0, 0], [-5, -5, 0], [5, -5, 0],
            [-10, -10, 0], [10, -10, 0], [0, -8, 0],
        ], dtype=float)

    elif formation_type == "encirclement":
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        radius = 10.0
        offsets = np.array(
            [[radius * np.cos(a), radius * np.sin(a), 0] for a in angles],
            dtype=float,
        )

    elif formation_type == "column":
        offsets = np.array([
            [0, 0, 0], [0, -5, 0], [0, -10, 0],
            [0, -15, 0], [0, -20, 0], [0, -25, 0],
        ], dtype=float)

    elif formation_type == "diamond":
        offsets = np.array([
            [0, 0, 10], [0, 10, 0], [-10, 0, 0],
            [10, 0, 0], [0, -10, 0], [0, 0, -10],
        ], dtype=float)

    elif formation_type == "dispersed":
        offsets = rng.uniform(
            low=[-30, -30, -15], high=[30, 30, 15], size=(6, 3)
        ).astype(float)

    elif formation_type == "converging":
        # Distinct ring geometry from `dispersed`'s scatter -- ported from
        # upstream commit 9158b081 ("added acceleration and split dispersed
        # and converging logic", https://github.com/pizz-beep/capstone).
        # Returns the (spread-out) STARTING offsets; the shrink over time is
        # handled in generate_swarm_sequence / generate_transition_sequence.
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        radius = 25.0
        offsets = np.array([
            [radius * np.cos(a), radius * np.sin(a), rng.uniform(-5, 5)]
            for a in angles
        ], dtype=float)

    elif formation_type == "shield":
        offsets = np.array([
            [-10, 10, 5], [0, 10, 5], [10, 10, 5],
            [-10, 10, -5], [0, 10, -5], [10, 10, -5],
        ], dtype=float)

    else:
        raise ValueError(f"Unknown formation type: {formation_type}")

    return offsets * spread
