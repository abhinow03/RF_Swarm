#!/usr/bin/env python3
"""Phase-4 EKF offline validation — the reproducible localization gate artifact.

Replays the gate_k6 TRUTH + measurement model EXACTLY (capstone_sim.truth_model +
rf_model, reconciled disciplined-array clock) with NO ROS, feeds it to the real
capstone_track.ekf.JointEKF (closed-form TDoA init + iterated joint EKF), and
reports position RMSE, clock recovery, and covariance consistency (NEES) across
several clock-noise seeds.

Run (from the workspace, after `source activate.sh`):
    python3 src/capstone_track/scripts/validate_ekf.py

This is the DESIGN validation (fingerprint association is provided by truth here to
isolate the estimator); the live-pipeline gate additionally exercises blind
association in the ROS node. Documents the observability finding: the per-receiver
clock is near-unobservable from a compact swarm (NOISE_MODEL §4a), so the clock is
prior-limited and the filter is mildly overconfident — both reported honestly.
"""
import sys
import numpy as np

# resolve sibling packages when run straight from the source tree
for p in ("capstone_sim", "capstone_track"):
    sys.path.insert(0, f"{__file__.rsplit('/capstone_track/', 1)[0]}/{p}")

from capstone_sim.truth_model import formation_offsets, emitter_states
from capstone_sim import rf_model
from capstone_track.ekf import JointEKF, tdoa_init, C_MPS

# ---- gate_k6 scenario constants (kept in sync with scenarios/gate_k6.yaml) ----
SEED, K, DURATION, BURST_RATE = 7, 6, 60.0, 2.0
RX = np.array([[90, 90, 0], [-90, 90, 35], [-90, -90, 0], [90, -90, 35], [0, 0, 70]], float)
N_RX = len(RX)
TRAJ = {"type": "constant_velocity", "start": [-30.0, 0.0, 50.0], "velocity": [1.0, 0.0, 0.0]}
RF = rf_model.load_rf({"rf": {
    "protocol": "ble", "carrier_hz": 2.4e9, "bandwidth_hz": 1.0e6, "sample_rate_hz": 6.0e6,
    "window_samples": 1850, "path_loss_exponent": 2.2, "rssi_ref_dbm_at_1m": -30.0,
    "noise_figure_db": 6.0, "rssi_shadow_sigma_db": 3.5, "cfo_ppm": 12.5, "burst_rate_hz": 2.0}})
CLK = rf_model.load_clock({"clock": {  # RECONCILED disciplined array (NOISE_MODEL §4a)
    "bias_max_ns": 30.0, "drift_max_ns_per_s": 0.3, "bias_sigma_ns": 30.0}})
OFFS = formation_offsets("encirclement", K, spread=1.0, seed=SEED)


def truth_pos(gid, t):
    pos, _, _, _ = emitter_states(min(t, DURATION), OFFS, TRAJ)
    return pos[gid]


def gen_bursts(seed, total_s=115.0):
    """Faithful replay of sim_node's staggered burst schedule + measurement model."""
    bias0, drift = rf_model.sample_receiver_clocks(CLK, N_RX, seed)
    rng = rf_model.make_noise_rng(seed)
    emit_count = np.zeros(K, int)
    ev = []
    sim_ns = 0
    while sim_ns / 1e9 <= total_s + 1e-9:
        sim_ns += int(1e9 / 50)
        t = sim_ns / 1e9
        pos, _, _, _ = emitter_states(min(t, DURATION), OFFS, TRAJ)
        for e in range(K):
            n_due = int(np.floor(t * BURST_RATE + e / K))
            if n_due <= emit_count[e]:
                continue
            emit_count[e] = n_due
            toas = np.empty(N_RX); sig = np.empty(N_RX)
            for r in range(N_RX):
                d = float(np.linalg.norm(pos[e] - RX[r]))
                sig[r] = rf_model.sigma_toa_s(RF, d)
                tb = float(bias0[r] + drift[r] * t)
                toas[r] = t + d / C_MPS + tb + float(rng.normal(0.0, sig[r]))
            ev.append(dict(e=e, t=t, toas=toas, sig=sig))
    return ev, bias0, drift


def true_clock_bias_m(bias0, drift, t):
    b = C_MPS * (bias0 + drift * t)
    return (b - b[0])[1:]


def run_seed(seed):
    ev, bias0, drift = gen_bursts(seed)
    ekf = JointEKF(N_RX, ref=0)                      # committed disciplined-array defaults
    last_t = ev[0]["t"]; inited = set(); pe = []; nees = []
    for b in ev:
        if b["t"] > last_t:
            ekf.predict(b["t"] - last_t); last_t = b["t"]
        if b["e"] not in inited:                     # truth-association here: one track per emitter id
            ekf.init_track(b["e"], tdoa_init(RX, b["toas"], 0)); inited.add(b["e"])
        else:
            ekf.update_burst(b["e"], RX, b["toas"], b["sig"])
        if len(inited) == K and b["t"] > 15:         # skip init transient
            p, v, cov = ekf.track_state(b["e"]); err = p - truth_pos(b["e"], b["t"])
            pe.append(float(np.linalg.norm(err)))
            nees.append(float(err @ np.linalg.solve(cov[:3, :3], err)))
    clk_err = float(np.linalg.norm(ekf.clock_bias_m() - true_clock_bias_m(bias0, drift, last_t)))
    return np.mean(pe), np.percentile(pe, 90), np.median(nees), clk_err


def main():
    print("=== Phase-4 EKF validation (gate_k6 replay, disciplined-array prior) ===")
    print(f"{'seed':>5} {'pos_rms':>9} {'pos_p90':>9} {'clk_err':>9} {'NEES(med)':>10}")
    prs, p90s, ces, nzs = [], [], [], []
    for sd in (7, 11, 23, 42, 101, 3, 99, 55):
        pr, p9, nz, ce = run_seed(sd)
        prs.append(pr); p90s.append(p9); ces.append(ce); nzs.append(nz)
        print(f"{sd:>5} {pr:>8.2f}m {p9:>8.2f}m {ce:>8.2f}m {nz:>10.1f}")
    print("-" * 48)
    print(f"MEAN  pos_rms={np.mean(prs):.2f}m (max {np.max(prs):.2f})  "
          f"p90={np.mean(p90s):.2f}m  clk_err={np.mean(ces):.2f}m  NEES_med={np.median(nzs):.1f}")
    print("Interpretation: position bounded/stable ~17 m mean (8-27 m across CLOCK "
          "realizations — the draw's projection onto the near-null space dominates; "
          "GDOP+clock-prior limited). Clock prior-dominated (weakly observable, "
          "NOISE_MODEL §4a). NEES med ~10 => filter ~2-3x overconfident; inflate "
          "track covariance ~3x downstream (worst-draw seeds reach NEES ~80).")
    ok = np.mean(prs) < 25.0 and np.max(prs) < 40.0
    print("GATE:", "PASS (bounded localization, no divergence)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
