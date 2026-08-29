"""
Step 2 of the "vendor teammate's retrained STGT" session (AUDIT.md sec V).

Measures the REAL regression-label distribution from the teammate's retrained
data (swarm_data/X_train.npy, denormalised, n=5879) and checks it against the
three AbsoluteCalibrator thresholds (calibration.py): velocity_trend (+-0.5 on
delta_v), spread_dynamics (+-0.1 on approach_rate), stability_trend (+-0.1 on
an early-vs-late delta).

IMPORTANT scale finding (see scripts/recover_reg_stats.py's assertion failure):
this repo's dataset.py::compute_regression_labels(), run on X_raw = X_norm *
train_std + train_mean, reproduces the checkpoint's embedded reg_mean/reg_std
EXACTLY for approach_rate and stability, but centroid_velocity comes out at
EXACTLY half the checkpoint's value (both mean and std scale by 2.000x, the
signature of a uniform per-sample multiplicative factor, not a data/seed
mismatch). Likeliest explanation: this repo's compute_regression_labels()
computes velocity as raw per-FRAME displacement (`deltas.mean()`), while
sliding_window_inference (inference.py) already carries an explicit dt=0.5s
per-frame convention -- i.e. the checkpoint's velocity is in metres/SECOND
(displacement / dt), and dividing by dt=0.5 is exactly a *2. Since the model's
reg_head was trained against the checkpoint's labels, "real" runtime velocity
(what actually reaches calibration.py) is on the dt-corrected (*2) scale, so
that correction is applied here to every velocity-derived number. approach_rate
and stability need no correction (verified exact match against the checkpoint).

For velocity_trend and stability_trend, which compare an EARLY-window value
against a LATE-window value, the population-level analog used here is: split
each of the 5879 independent 50-step training sequences into two 25-step
halves and run compute_regression_labels() on each half separately. This is
the closest available stand-in for "early window prediction vs late window
prediction in a stream" when all we have is a bank of independent sequences,
not a continuous multi-window stream.

Usage:
    python scripts/measure_reg_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.dataset import compute_regression_labels  # noqa: E402

DATA_DIR = REPO / "swarm_data"
DT = 0.5  # seconds/frame, per inference.py's sliding_window_inference default
VELOCITY_DT_CORRECTION = 1.0 / DT  # == 2.0, see module docstring


def percentiles(x, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def main():
    X_train_norm = np.load(DATA_DIR / "X_train.npy")
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    X_raw = X_train_norm * train_std + train_mean
    n = len(X_raw)
    print(f"n sequences = {n}")

    full = np.array([compute_regression_labels(X_raw[i]) for i in range(n)])
    velocity_frame, approach_rate, stability = full[:, 0], full[:, 1], full[:, 2]
    velocity_physical = velocity_frame * VELOCITY_DT_CORRECTION

    early = np.array([compute_regression_labels(X_raw[i][:25]) for i in range(n)])
    late = np.array([compute_regression_labels(X_raw[i][25:]) for i in range(n)])
    delta_v_physical = (late[:, 0] - early[:, 0]) * VELOCITY_DT_CORRECTION
    stab_early, stab_late = early[:, 2], late[:, 2]
    delta_stability = stab_late - stab_early

    print("\n=== full-sequence field distributions (n={}) ===".format(n))
    for name, vals in [("centroid_velocity (physical, dt-corrected)", velocity_physical),
                       ("approach_rate", approach_rate),
                       ("stability", stability)]:
        p = percentiles(vals)
        print(f"\n{name}:")
        print(f"  min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f}")
        print(f"  percentiles: {p}")

    print("\n=== early/late-half derived quantities (n={}) ===".format(n))
    p_dv = percentiles(delta_v_physical)
    print(f"\ndelta_v (late-half - early-half centroid_velocity, physical):")
    print(f"  min={delta_v_physical.min():.4f} max={delta_v_physical.max():.4f} "
          f"mean={delta_v_physical.mean():.4f} std={delta_v_physical.std():.4f}")
    print(f"  percentiles: {p_dv}")

    p_ds = percentiles(delta_stability)
    print(f"\ndelta_stability (late-half - early-half stability):")
    print(f"  min={delta_stability.min():.4f} max={delta_stability.max():.4f} "
          f"mean={delta_stability.mean():.4f} std={delta_stability.std():.4f}")
    print(f"  percentiles: {p_ds}")

    print("\n=== threshold-crossing verdicts ===")
    frac_dv = float(np.mean(np.abs(delta_v_physical) > 0.5))
    print(f"a. fraction |delta_v| > 0.5 (velocity_trend fires accelerating/decelerating): "
          f"{frac_dv:.4%} ({int(frac_dv * n)}/{n})")

    frac_approach = float(np.mean(np.abs(approach_rate) > 0.1))
    frac_conv = float(np.mean(approach_rate < -0.1))
    frac_disp = float(np.mean(approach_rate > 0.1))
    print(f"b. fraction |approach_rate| > 0.1 (spread_dynamics fires converging/dispersing): "
          f"{frac_approach:.4%} ({int(frac_approach * n)}/{n}) "
          f"[converging={frac_conv:.4%}, dispersing={frac_disp:.4%}]")

    frac_stab_delta = float(np.mean(np.abs(delta_stability) > 0.1))
    print(f"c. stability range: [{stability.min():.4f}, {stability.max():.4f}]; "
          f"fraction |delta_stability| > 0.1 (stability_trend fires degrading/improving): "
          f"{frac_stab_delta:.4%} ({int(frac_stab_delta * n)}/{n})")

    # synth_context() sampling ranges, from AUDIT.md sec F's table / build_sft_dataset.py
    print("\n=== comparison vs synth_context() sampling ranges ===")
    print(f"  velocity:   synth [0.6-2.0 approx via mean_conf-linked draw] -- see sec BB for exact "
          f"post-fix ranges; real physical mean={velocity_physical.mean():.3f} "
          f"[{percentiles(velocity_physical)['p1']:.3f}, {percentiles(velocity_physical)['p99']:.3f}] "
          f"(p1-p99)")
    print(f"  approach:   synth [-1.5, 1.5] (post sec BB widening); "
          f"real [{percentiles(approach_rate)['p1']:.3f}, {percentiles(approach_rate)['p99']:.3f}] (p1-p99)")
    print(f"  stability:  synth [0.5, 0.98] (early/late draw, sec BB); "
          f"real [{percentiles(stability)['p1']:.3f}, {percentiles(stability)['p99']:.3f}] (p1-p99)")

    np.savez(DATA_DIR / "_reg_distribution_analysis.npz",
             velocity_physical=velocity_physical, approach_rate=approach_rate, stability=stability,
             delta_v_physical=delta_v_physical, delta_stability=delta_stability)
    print(f"\nsaved raw arrays to {DATA_DIR / '_reg_distribution_analysis.npz'}")


if __name__ == "__main__":
    main()
