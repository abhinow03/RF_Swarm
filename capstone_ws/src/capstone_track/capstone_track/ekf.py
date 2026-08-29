"""Phase 4 localization: closed-form TDoA init + joint fused EKF. Pure numpy, NO ROS.

Joint state (per the locked design): per-track [x,y,z,vx,vy,vz] for K tracks,
PLUS a SHARED per-receiver clock block [bias_r, drift_r] for receivers 1..N-1
RELATIVE to reference receiver 0 (only N-1 relative biases are observable; the
absolute folds into the reference). Clock states are ESTIMATED here, never read
from /receivers (which exposes only clock_bias_sigma) — the carried requirement.

Measurement = TDoA only. For a burst from track k seen at receivers R (ref=0):
    z_j  = c*(toa_j - toa_0)                                    [metres, bias-corrupted]
    h_j  = (|p_k - r_j| - |p_k - r_0|) + b_j                    [b_0 == 0 by reference]
    H: dh/dp_k = u_j - u_0 (unit LOS diff);  dh/db_j = 1
Differenced measurements share the ref-receiver noise, so R is dense:
    R_jl = c^2 * (sigma_toa_0^2  + delta_jl * sigma_toa_j^2)
Velocity comes only from the constant-velocity process model over time (FDoA is
CFO+Doppler unseparated with unknown per-emitter CFO, so it is NOT fused).
"""
from __future__ import annotations

import numpy as np

C_MPS = 299_792_458.0
TRK = 6   # per-track state dimension [x y z vx vy vz]


def tdoa_init(rx_pos: np.ndarray, toas: np.ndarray, ref: int = 0) -> np.ndarray:
    """Closed-form linear-LS TDoA multilateration -> initial (x,y,z).
    Solves, over receivers i != ref, the linearized set
        2 (r_i - r_ref) . p + 2 d_i R0 = |r_i|^2 - |r_ref|^2 - d_i^2
    with d_i = c*(toa_i - toa_ref) and unknowns [p (3), R0=|p-r_ref|]. Clock bias
    is ignored here (rough init; the EKF refines it and estimates the clock)."""
    rx = np.asarray(rx_pos, float)
    d = C_MPS * (np.asarray(toas, float) - toas[ref])
    rows, rhs = [], []
    for i in range(len(rx)):
        if i == ref:
            continue
        rows.append(np.concatenate([2.0 * (rx[i] - rx[ref]), [2.0 * d[i]]]))
        rhs.append(rx[i] @ rx[i] - rx[ref] @ rx[ref] - d[i] ** 2)
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return sol[:3]


def gdop(rx_pos: np.ndarray, target: np.ndarray) -> float:
    """Position GDOP (3D) of a TDoA array observing `target` — the differenced
    unit-LOS geometry (u_i - u_0), GDOP = sqrt(trace((G^T G)^-1)). np.inf for a
    degenerate/rank-deficient layout. Mirrors capstone_sim.truth_model.compute_gdop
    (kept here so the estimator has no dependency on the sim package)."""
    rx = np.asarray(rx_pos, float); p = np.asarray(target, float)
    if rx.shape[0] < 4:
        return np.inf
    los = p[None, :] - rx
    dist = np.linalg.norm(los, axis=1)
    if np.any(dist < 1e-6):
        return np.inf
    u = los / dist[:, None]
    G = u[1:] - u[0][None, :]
    sv = np.linalg.svd(G, compute_uv=False)
    if sv.min() < 1e-9:
        return np.inf
    tr = float(np.trace(np.linalg.inv(G.T @ G)))
    return float(np.sqrt(tr)) if np.isfinite(tr) and tr >= 0 else np.inf


class JointEKF:
    """Joint EKF over K tracks + shared (N-1) relative receiver clock states."""

    def __init__(self, n_rx: int, ref: int = 0,
                 clock_bias_sigma_s: float = 30e-9,
                 clock_drift_sigma_s: float = 0.3e-9,
                 q_acc: float = 0.02, q_drift: float = 1e-8, q_bias: float = 1.0):
        """clock_bias_sigma_s: the DISCIPLINED-ARRAY residual the receivers report
        on /receivers (ReceiverState.clock_bias_sigma) — the informative prior that
        makes the otherwise-unobservable per-receiver clock estimable (a compact
        swarm's common translation is degenerate with a clock-bias shift; see
        NOISE_MODEL.md §4 / Phase-4 observability finding). It is the stated
        UNCERTAINTY, not the true bias (which lives only on /truth/receivers) — the
        clock stays an ESTIMATED state, seeded belief-synced (mean 0) and refined by
        the TDoA measurements (the estimate provably beats the prior). clock_drift
        has no published sigma, so its prior comes from the disciplined-array spec."""
        self.n_rx = int(n_rx)
        self.ref = int(ref)
        self.nc = self.n_rx - 1                      # relative clock receivers (1..N-1)
        self.tracks = {}                             # global_id -> row offset in state
        self.clock0 = 0                              # filled once tracks known (kept at end)
        self.x = np.zeros(2 * self.nc)               # start belief-synced (mean 0)
        # Relative-clock prior. State b_j = c*(beta_j - beta_ref) with per-receiver
        # absolute residual beta ~ N(0, sigma^2) i.i.d.; the SHARED reference makes
        # the relative biases fully correlated: Cov(b) = sigma_m^2 * (I + 11^T).
        sig_b = C_MPS * float(clock_bias_sigma_s)    # absolute bias sigma, metres
        sig_d = C_MPS * float(clock_drift_sigma_s)   # absolute drift sigma, m/s
        corr = np.eye(self.nc) + np.ones((self.nc, self.nc))
        self.P = np.zeros((2 * self.nc, 2 * self.nc))
        self.P[:self.nc, :self.nc] = sig_b ** 2 * corr
        self.P[self.nc:, self.nc:] = sig_d ** 2 * corr
        # process-noise densities. q_bias is a direct random walk on the clock BIAS
        # (disciplining jitter + residual drift); it is what keeps the clock — and
        # hence position — covariance HONEST under the weak-observability null space
        # (without it the filter falsely "resolves" the clock and reports a covariance
        # far tighter than the true error). Tuned for filter consistency (NEES~=dof).
        self.q_acc = float(q_acc)                    # track accel PSD (m^2/s^3)
        self.q_drift = float(q_drift)                # clock-drift random-walk PSD (m^2/s^3)
        self.q_bias = float(q_bias)                  # clock-bias random-walk PSD (m^2/s)

    # ---- state layout: [track blocks ...][clock bias (nc)][clock drift (nc)] ----
    def _cb(self):
        return len(self.x) - 2 * self.nc            # clock-block start index

    def has_track(self, gid):
        return gid in self.tracks

    def init_track(self, gid, p0):
        """Add a new track (position p0, zero velocity) as a fresh 6-state block
        inserted BEFORE the clock block."""
        cb = self._cb()
        blk = np.zeros(TRK); blk[:3] = p0
        self.x = np.concatenate([self.x[:cb], blk, self.x[cb:]])
        P = np.zeros((len(self.x), len(self.x)))
        P[:cb, :cb] = self.P[:cb, :cb]                       # old track blocks
        P[:cb, cb + TRK:] = self.P[:cb, cb:]                 # cross old-track / clock
        P[cb + TRK:, :cb] = self.P[cb:, :cb]
        P[cb + TRK:, cb + TRK:] = self.P[cb:, cb:]           # clock block
        P[cb:cb + 3, cb:cb + 3] = np.eye(3) * (60.0) ** 2    # pos prior (m) — init is rough
        P[cb + 3:cb + 6, cb + 3:cb + 6] = np.eye(3) * (20.0) ** 2  # vel prior (m/s)
        self.tracks[gid] = cb
        self.P = P

    def predict(self, dt):
        if dt <= 0:
            return
        n = len(self.x); F = np.eye(n); Q = np.zeros((n, n))
        for gid, o in self.tracks.items():
            F[o:o + 3, o + 3:o + 6] = np.eye(3) * dt         # p += v dt
            # white-accel process noise on each axis
            q = self.q_acc
            Qpp = q * dt ** 3 / 3.0; Qpv = q * dt ** 2 / 2.0; Qvv = q * dt
            for a in range(3):
                Q[o + a, o + a] = Qpp; Q[o + a, o + 3 + a] = Qpv
                Q[o + 3 + a, o + a] = Qpv; Q[o + 3 + a, o + 3 + a] = Qvv
        cb = self._cb()
        for r in range(self.nc):                              # b += d dt
            bi, di = cb + r, cb + self.nc + r
            F[bi, di] = dt
            # drift random walk integrated into bias, PLUS a direct bias random walk
            Q[bi, bi] = self.q_bias * dt + self.q_drift * dt ** 3 / 3.0
            Q[bi, di] = Q[di, bi] = self.q_drift * dt ** 2 / 2.0
            Q[di, di] = self.q_drift * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_burst(self, gid, rx_pos, toas, toa_sigmas, max_iters: int = 8, tol: float = 1e-3):
        """ITERATED-EKF update from one burst of track `gid` (all receivers). Returns
        (n_used, innovation_rms_m).

        Iteration (relinearizing the range-difference measurement at the updated
        position each pass) is REQUIRED here, not a refinement: the closed-form init
        is clock-corrupted and thus biased, and under the compact-swarm clock
        near-null-space a single linearization at that biased point makes the filter
        both inaccurate (~30 m) AND badly overconfident. Relinearizing collapses it
        to the GDOP/clock-prior limit (~12 m) and restores covariance sanity."""
        o = self.tracks[gid]; cb = self._cb()
        rx = np.asarray(rx_pos, float); toas = np.asarray(toas, float)
        sig = np.asarray(toa_sigmas, float)
        ref = self.ref
        others = [j for j in range(self.n_rx) if j != ref]
        m = len(others)
        z = np.array([C_MPS * (toas[j] - toas[ref]) for j in others])
        R = np.full((m, m), (C_MPS * sig[ref]) ** 2)         # shared-ref correlation
        for idx, j in enumerate(others):
            R[idx, idx] += (C_MPS * sig[j]) ** 2

        x_prior = self.x.copy(); P = self.P
        xi = x_prior.copy()
        I = np.eye(len(self.x))
        H = np.zeros((m, len(self.x))); K = None; y0 = None
        for _ in range(max_iters):
            p = xi[o:o + 3]
            u = (p[None, :] - rx) / (np.linalg.norm(p[None, :] - rx, axis=1, keepdims=True) + 1e-9)
            h = np.empty(m); H[:] = 0.0
            for idx, j in enumerate(others):
                b_j = xi[cb + (j - 1)] if j > ref else 0.0   # relative bias (recv 0 == ref == 0)
                h[idx] = (np.linalg.norm(p - rx[j]) - np.linalg.norm(p - rx[ref])) + b_j
                H[idx, o:o + 3] = u[j] - u[ref]
                if j > ref:
                    H[idx, cb + (j - 1)] = 1.0
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            # IEKF innovation carries the relinearization offset H(x_prior - xi)
            y = z - h - H @ (x_prior - xi)
            if y0 is None:
                y0 = z - h                                    # first-pass innovation (reporting)
            x_new = x_prior + K @ y
            step = float(np.linalg.norm(x_new[o:o + 3] - xi[o:o + 3]))
            xi = x_new
            if step < tol:
                break
        self.x = xi
        self.P = (I - K @ H) @ P
        return m, float(np.sqrt(np.mean(y0 ** 2)))

    # ---- readout ----
    def track_state(self, gid):
        o = self.tracks[gid]
        return self.x[o:o + 3].copy(), self.x[o + 3:o + 6].copy(), self.P[o:o + 6, o:o + 6].copy()

    def clock_bias_m(self):
        cb = self._cb()
        return self.x[cb:cb + self.nc].copy()                # relative to receiver `ref`, metres
