"""
Versioned calibrators: the ONLY place that maps raw STGT regression scalars onto
the frozen narrative vocabulary in ``context_spec.py``.

Why this exists: STGT's regression outputs (``centroid_velocity``, ``approach_rate``,
``formation_stability``) are being retrained and may change scale/units (see
AUDIT.md sec F/F2 for the current normalized-space magnitudes — centroid_velocity
~0.04-0.07, approach_rate ~-0.001, stability 0.79-0.95 — and the CODE_REVIEW.md
units caveat this stems from). Before this refactor, the thresholds that decide
"accelerating" vs "steady" etc. were hardcoded literals inside
``build_tactical_context()``; a units change would have meant redesigning that
function. Now a units change means fitting a new ``Calibrator`` and swapping it in —
``build_tactical_context()`` itself doesn't change.

Two implementations:
  - AbsoluteCalibrator ("calib-v0-absolute"): today's hardcoded cutoffs, unchanged.
    Default, so existing behaviour is byte-identical (see the regression test).
  - PercentileCalibrator ("calib-v1-percentile"): cut-points fitted from the spread
    of a real distribution of STGT outputs, so they scale automatically with
    whatever units the new STGT model produces — a units change becomes a refit,
    not a redesign.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import context_spec as spec


class Calibrator(ABC):
    """Maps raw STGT scalars to the frozen vocabulary in context_spec.py."""

    VERSION: str = "unversioned"

    @abstractmethod
    def velocity_trend(self, delta_v: float) -> str:
        """delta_v: mean(late-half centroid_velocity) - mean(early-half). ->
        one of context_spec.VELOCITY_TREND_VALUES."""

    @abstractmethod
    def stability_trend(self, early: float, late: float) -> str:
        """early/late: mean formation_stability over the first/second half of the
        window sequence. -> one of context_spec.STABILITY_TREND_VALUES."""

    @abstractmethod
    def spread_dynamics(self, mean_approach: float) -> str:
        """mean_approach: mean approach_rate over the window sequence. ->
        one of context_spec.SPREAD_DYNAMICS_VALUES."""

    def role_differentiation(self, role_true_count: int, n: int) -> str:
        """Majority vote over role_differentiation booleans. This is already a
        unit-invariant fraction (not a scale-dependent scalar threshold like the
        other three), so both calibrators share this exact same rule — there is
        nothing for a units change to affect here."""
        return (spec.ROLE_DIFFERENTIATION_PRESENT if role_true_count > (n // 2)
                else spec.ROLE_DIFFERENTIATION_NOT_PROMINENT)


class AbsoluteCalibrator(Calibrator):
    """Today's hardcoded cutoffs, extracted verbatim from build_tactical_context()
    (pre-refactor): +-0.5 velocity, +-0.1 approach, +-0.1 relative stability delta.
    Default calibrator — using it reproduces current behaviour exactly."""

    VERSION = "calib-v0-absolute"

    def __init__(self, velocity_threshold: float = 0.5, approach_threshold: float = 0.1,
                 stability_delta_threshold: float = 0.1):
        self.velocity_threshold = velocity_threshold
        self.approach_threshold = approach_threshold
        self.stability_delta_threshold = stability_delta_threshold

    def velocity_trend(self, delta_v: float) -> str:
        if delta_v > self.velocity_threshold:
            return spec.VELOCITY_ACCELERATING
        if delta_v < -self.velocity_threshold:
            return spec.VELOCITY_DECELERATING
        return spec.VELOCITY_STEADY

    def stability_trend(self, early: float, late: float) -> str:
        if late < early - self.stability_delta_threshold:
            return spec.STABILITY_DEGRADING
        if late > early + self.stability_delta_threshold:
            return spec.STABILITY_IMPROVING
        return spec.STABILITY_HOLDING

    def spread_dynamics(self, mean_approach: float) -> str:
        if mean_approach < -self.approach_threshold:
            return spec.SPREAD_CONVERGING
        if mean_approach > self.approach_threshold:
            return spec.SPREAD_DISPERSING
        return spec.SPREAD_STABLE


@dataclass
class PercentileCalibrator(Calibrator):
    """Cut-points fitted from the spread (P_high - P_low) of a distribution of
    real STGT outputs, scaled by `sensitivity`. Unit-invariant by construction: if
    STGT's output scale changes 100x, refitting on the new distribution produces
    thresholds that are ~100x different too, with zero code change.

    fit() computes velocity_threshold / approach_threshold / stability_delta_threshold
    the same way AbsoluteCalibrator uses them (compare a delta against +-threshold),
    so the classification logic in velocity_trend/stability_trend/spread_dynamics is
    IDENTICAL in shape to AbsoluteCalibrator's — only the threshold magnitudes are
    data-derived instead of hardcoded.
    """

    VERSION = "calib-v1-percentile"

    velocity_threshold: float = 0.0
    approach_threshold: float = 0.0
    stability_delta_threshold: float = 0.0
    low_pct: float = 10.0
    high_pct: float = 90.0
    sensitivity: float = 0.5
    n_fit_samples: int = 0
    fitted: bool = False
    note: Optional[str] = None

    def fit(self, predictions: list[dict]) -> "PercentileCalibrator":
        """predictions: list of raw STGT prediction dicts (as returned by
        inference.predict / sliding_window_inference), each with at least
        centroid_velocity, approach_rate, formation_stability keys."""
        velocities = [p["centroid_velocity"] for p in predictions]
        approaches = [p["approach_rate"] for p in predictions]
        stabilities = [p["formation_stability"] for p in predictions]

        def spread(values):
            return float(np.percentile(values, self.high_pct) - np.percentile(values, self.low_pct))

        self.velocity_threshold = self.sensitivity * spread(velocities)
        self.approach_threshold = self.sensitivity * spread(approaches)
        self.stability_delta_threshold = self.sensitivity * spread(stabilities)
        self.n_fit_samples = len(predictions)
        self.fitted = True
        return self

    def velocity_trend(self, delta_v: float) -> str:
        self._require_fitted()
        if delta_v > self.velocity_threshold:
            return spec.VELOCITY_ACCELERATING
        if delta_v < -self.velocity_threshold:
            return spec.VELOCITY_DECELERATING
        return spec.VELOCITY_STEADY

    def stability_trend(self, early: float, late: float) -> str:
        self._require_fitted()
        if late < early - self.stability_delta_threshold:
            return spec.STABILITY_DEGRADING
        if late > early + self.stability_delta_threshold:
            return spec.STABILITY_IMPROVING
        return spec.STABILITY_HOLDING

    def spread_dynamics(self, mean_approach: float) -> str:
        self._require_fitted()
        if mean_approach < -self.approach_threshold:
            return spec.SPREAD_CONVERGING
        if mean_approach > self.approach_threshold:
            return spec.SPREAD_DISPERSING
        return spec.SPREAD_STABLE

    def _require_fitted(self):
        if not self.fitted:
            raise RuntimeError("PercentileCalibrator used before fit() (or from_json of an "
                                "unfitted one) — thresholds are all 0.0 and every classification "
                                "would degenerate to the 'accelerating'/'converging' branches.")

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"version": self.VERSION, **asdict(self)}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PercentileCalibrator":
        with open(path) as f:
            data = json.load(f)
        data.pop("version", None)
        return cls(**data)
