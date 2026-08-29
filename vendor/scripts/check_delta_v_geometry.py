"""
Step 1 of the "settle delta_v noise-vs-geometry, recalibrate synth_context,
diagnose the prior skew" session (AUDIT.md sec W).

Decisive test for AUDIT.md sec V's open question ("is delta_v's 33% ±0.5
crossing rate noise from small half-windows, or real signal?"): regenerate
transition sequences with noise_std=0.0 via the teammate's OWN
generate_transition_sequence (github.com/pizz-beep/capstone @ b139dcee71f,
already vendor-inspected in sec V/step 3) across her three blend regimes, and
recompute delta_v with her OWN compute_regression_labels (the formula that
matches best_model.pt's embedded reg_mean/reg_std exactly, sec V). With zero
sensor/motion noise, any remaining delta_v crossings must be pure geometry
(offset-mean asymmetry across the blend), not measurement noise -- noise_std=0
removes noise from the model entirely.

Geometry hypothesis: centroid = mean of 6 drone positions, and
mean(formation_offsets) is NOT zero for every formation (v_shape/column/shield
skew negative or positive on y). During the blend window, the mean offset
itself shifts from mean(offsets_a) to mean(offsets_b) -- this shift adds
apparent centroid velocity distinct from the (noise-free, constant) swarm
translation velocity. Regime 0 (blend late) concentrates that shift in the
LATE half -> inflates v_late -> positive delta_v. Regime 2 (blend early)
concentrates it in the EARLY half -> negative delta_v. Regime 1 (blend
spans the midpoint) splits the shift across both halves -> partially cancels.

Imports the teammate's data_gen.py/dataset.py/config.py directly from the
scratchpad clone (same one vendored into src/swarm_intent/stgt/ in sec V step
3) rather than reimplementing them -- if the clone isn't present, re-clones it.

Usage:
    python scripts/check_delta_v_geometry.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRATCHPAD_CLONE = Path("/tmp/claude-1004/-home-pw26-akp-01-aadhya-uav-llm/"
                        "fae9053f-5ab6-47ba-8583-d58ca0394b31/scratchpad/teammate_repo")
UPSTREAM_COMMIT = "b139dcee71feb82244ef1470a6193a628040f318"

N_PAIRS_SAMPLE = 12          # of the 42 ordered (a,b) pairs, cycle through this many
N_PER_REGIME_PER_PAIR = 14   # -> 12 * 14 * 3 regimes = 504 sequences total


def ensure_clone():
    if not (SCRATCHPAD_CLONE / "data_gen.py").exists():
        SCRATCHPAD_CLONE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/pizz-beep/capstone.git",
                        str(SCRATCHPAD_CLONE)], check=True)
    head = subprocess.run(["git", "-C", str(SCRATCHPAD_CLONE), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == UPSTREAM_COMMIT, f"clone HEAD {head} != expected {UPSTREAM_COMMIT}"


def main():
    ensure_clone()
    sys.path.insert(0, str(SCRATCHPAD_CLONE))
    sys.path.insert(0, str(REPO / "src"))

    from config import FORMATION_NAMES  # teammate's, noqa
    from data_gen import generate_transition_sequence, get_formation_offsets  # noqa

    def centroid_velocity_half(seq_raw):
        """Same formula as the teammate's dataset.py::compute_regression_labels
        (centroids -> frame-diffs -> mean speed / dt=0.5), extracted standalone
        because her function hardcodes `t = np.arange(50, ...)` for the
        approach_rate polyfit, which only works on full 50-step sequences --
        it crashes (TypeError: x/y length mismatch) on a 25-step half-window.
        Only centroid_velocity is needed here, so the broken half is skipped."""
        centroids = seq_raw.mean(axis=1)
        deltas = np.diff(centroids, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        return float(speeds.mean() / 0.5)

    base_formations = [f for f in FORMATION_NAMES if f != "transitioning"]

    # Report the mean-offset hypothesis numbers first, exactly as stated.
    print("=== mean(offset) per formation (y-component, the dominant asymmetric axis) ===")
    for f in base_formations:
        off = get_formation_offsets(f, spread=1.0)
        print(f"  {f:14s} mean_offset={off.mean(axis=0).round(2).tolist()}")

    pairs = [(a, b) for a in base_formations for b in base_formations if a != b]
    rng = np.random.default_rng(0)
    sample_pairs = [pairs[i] for i in rng.choice(len(pairs), size=N_PAIRS_SAMPLE, replace=False)]

    regime_bstart_bend = {
        # exact ranges from the teammate's generate_transition_dataset
        0: lambda: (int(rng.integers(33, 43)), None),   # blend late -> filled below
        1: lambda: (int(rng.integers(10, 25)), None),
        2: lambda: (None, int(rng.integers(8, 18))),    # blend early -> b_start filled below
    }

    records = []  # (regime, pair, delta_v)
    for pair_idx, (f_a, f_b) in enumerate(sample_pairs):
        for regime in (0, 1, 2):
            for _ in range(N_PER_REGIME_PER_PAIR):
                if regime == 0:
                    b_start = int(rng.integers(33, 43))
                    b_end = int(np.clip(b_start + int(rng.integers(5, 9)), b_start + 5, 49))
                elif regime == 1:
                    b_start = int(rng.integers(10, 25))
                    b_end = int(np.clip(b_start + int(rng.integers(14, 22)), b_start + 5, 45))
                else:
                    b_end = int(rng.integers(8, 18))
                    b_start = int(np.clip(b_end - int(rng.integers(5, 9)), 1, b_end - 5))

                spread = rng.uniform(0.7, 1.5)
                seq, _meta = generate_transition_sequence(
                    formation_a=f_a, formation_b=f_b, n_timesteps=50, dt=0.5,
                    spread=spread, noise_std=0.0, blend_start=b_start, blend_end=b_end,
                )
                v_early = centroid_velocity_half(seq[:25])
                v_late = centroid_velocity_half(seq[25:])
                delta_v = v_late - v_early
                records.append((regime, f"{f_a}->{f_b}", float(delta_v)))

    delta_v_arr = np.array([r[2] for r in records])
    regime_arr = np.array([r[0] for r in records])
    n = len(records)
    print(f"\n=== n={n} sequences, noise_std=0.0, {N_PAIRS_SAMPLE} pairs x 3 regimes x "
          f"{N_PER_REGIME_PER_PAIR} ===")

    frac_cross = float(np.mean(np.abs(delta_v_arr) > 0.5))
    print(f"\nOVERALL |delta_v|>0.5 crossing rate (zero noise): {frac_cross:.4%} "
          f"({int(frac_cross * n)}/{n})")
    print(f"OVERALL mean signed delta_v: {delta_v_arr.mean():+.4f} (std={delta_v_arr.std():.4f})")

    print("\n=== per-regime ===")
    regime_names = {0: "0 (blend LATE, mostly A)", 1: "1 (blend MID, transitioning)",
                    2: "2 (blend EARLY, mostly B)"}
    for regime in (0, 1, 2):
        mask = regime_arr == regime
        vals = delta_v_arr[mask]
        frac = float(np.mean(np.abs(vals) > 0.5))
        print(f"  regime {regime_names[regime]}: n={mask.sum()}  "
              f"mean_signed_delta_v={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"frac|delta_v|>0.5={frac:.4%}")

    # Text histogram (10 bins) -- overall and per-regime, so bimodality is visible without
    # needing to open an image.
    bins = np.linspace(delta_v_arr.min(), delta_v_arr.max(), 21)
    print(f"\n=== delta_v histogram (20 bins, range [{bins[0]:.2f}, {bins[-1]:.2f}]) ===")
    print(f"{'bin center':>12s} | {'regime0':>8s} {'regime1':>8s} {'regime2':>8s} | total")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        center = (lo + hi) / 2
        counts = []
        for regime in (0, 1, 2):
            mask = (regime_arr == regime) & (delta_v_arr >= lo) & (delta_v_arr < hi)
            counts.append(int(mask.sum()))
        total = sum(counts)
        bar = "#" * total
        print(f"{center:12.3f} | {counts[0]:8d} {counts[1]:8d} {counts[2]:8d} | {bar}")

    np.savez(REPO / "swarm_data" / "_delta_v_geometry_check.npz",
             delta_v=delta_v_arr, regime=regime_arr)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {0: "#3B82C4", 1: "#6B7280", 2: "#C4753B"}  # blue / neutral gray / orange
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for regime in (0, 1, 2):
            vals = delta_v_arr[regime_arr == regime]
            ax.hist(vals, bins=30, alpha=0.6, label=regime_names[regime], color=colors[regime])
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.axvline(-0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("delta_v (late-half - early-half centroid_velocity, m/s, noise_std=0)")
        ax.set_ylabel("count")
        ax.set_title("delta_v distribution by blend regime (zero-noise geometry test)")
        ax.legend()
        fig.tight_layout()
        out_path = REPO / "evaluation" / "delta_v_geometry_histogram.png"
        fig.savefig(out_path)
        print(f"\nsaved plot to {out_path}")
    except ImportError:
        print("\nmatplotlib not available, skipped plot (text histogram above suffices)")


if __name__ == "__main__":
    main()
