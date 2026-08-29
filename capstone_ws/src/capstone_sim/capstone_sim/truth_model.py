"""Phase 1 truth kinematics — pure Python/numpy, NO ROS (unit-testable).

Formation geometry is taken VERBATIM from the STGT-side repo (READ-ONLY, pinned
commit 69ac2b6): `swarm_intent.formations.get_formation_offsets`, which returns a
(6, 3) ENU-metre offset template per formation. Per the locked decision:
  * K == 6 -> the full 6-row template, byte-identical to what STGT expects (gate).
  * K in [1, 5] -> the LEADING K rows of that same template (deterministic subset,
    dev/debug only; never invents geometry STGT wouldn't produce).
  * K > 6 -> unsupported: the template only defines 6 drones (raise, don't guess).

Forward-only sim: the centroid trajectory is the INPUT; per-emitter truth pose =
centroid + rigid formation offset. This module produces TRUTH ONLY — no
observations, nothing solves backward from assumed measurements.
"""
from __future__ import annotations

import numpy as np

from swarm_intent.formations import get_formation_offsets, FORMATION_INFO

MAX_K = 6  # get_formation_offsets is authored for exactly 6 drones
# 7 static formations; "transitioning" is dynamic (a mid-sequence change) and has
# no static offset template — get_formation_offsets raises on it.
STATIC_FORMATIONS = tuple(f for f in FORMATION_INFO if f != "transitioning")
# Procedural (NON-STGT) layouts for K that the 6-drone STGT template cannot express
# (K>6). Used only to exercise the localization / K-estimation pipeline at high K;
# these are NOT STGT-recognized formations and must not be read as such.
PROCEDURAL_FORMATIONS = ("ring",)


def _procedural_ring(k: int, spread: float) -> np.ndarray:
    """k emitters evenly on a horizontal ring (ENU) — non-degenerate, any k. NOT an
    STGT formation; a synthetic layout for K>6 where the template has no rows."""
    ang = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
    r = 12.0 * spread
    return np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(k)], axis=1)


def formation_offsets(label: str, k: int, spread: float = 1.0, seed: int = 0) -> np.ndarray:
    """(k, 3) ENU offsets from the swarm centroid.

    STGT static formations: leading-K subset of the frozen 6-drone template (k<=6).
    Procedural layouts (e.g. "ring"): synthetic non-STGT geometry for any k, used for
    the K>6 sweep points the STGT template cannot represent.
    """
    if label in PROCEDURAL_FORMATIONS:
        if k < 1:
            raise ValueError(f"k={k} must be >= 1")
        return _procedural_ring(k, spread)
    if label not in STATIC_FORMATIONS:
        raise ValueError(
            f"formation_label {label!r} is not a static STGT formation "
            f"{STATIC_FORMATIONS} nor a procedural layout {PROCEDURAL_FORMATIONS}")
    if not (1 <= k <= MAX_K):
        raise ValueError(
            f"k={k} out of range for STGT formation {label!r}: the template defines "
            f"exactly {MAX_K} drones (k in [1,{MAX_K}]). For k>{MAX_K} use a procedural "
            f"layout ({PROCEDURAL_FORMATIONS}), which is non-STGT by construction.")
    rng = np.random.default_rng(seed)
    template = get_formation_offsets(label, spread=spread, rng=rng)  # (6, 3), verbatim
    return template[:k].copy()                                       # leading-K subset


def centroid_state(t: float, traj: dict):
    """(pos(3,), vel(3,)) of the swarm centroid at sim-time t seconds."""
    kind = traj.get("type", "static")
    start = np.asarray(traj.get("start", [0.0, 0.0, 50.0]), dtype=float)

    if kind == "static":
        return start.copy(), np.zeros(3)

    if kind == "constant_velocity":
        vel = np.asarray(traj["velocity"], dtype=float)
        return start + vel * t, vel.copy()

    if kind == "waypoints":
        # piecewise-linear through [start, *waypoints] at the given speed (m/s);
        # holds at the final waypoint once traversed.
        pts = [start] + [np.asarray(w, dtype=float) for w in traj["waypoints"]]
        speed = float(traj.get("speed", 5.0))
        seg_t = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            dur = seg_len / speed if speed > 0 else 0.0
            if t <= seg_t + dur and dur > 0:
                frac = (t - seg_t) / dur
                direction = (b - a) / seg_len
                return a + (b - a) * frac, direction * speed
            seg_t += dur
        return pts[-1].copy(), np.zeros(3)  # arrived; hold

    raise ValueError(f"unknown trajectory type {kind!r}")


def emitter_states(t: float, offsets: np.ndarray, traj: dict):
    """Rigid formation: every emitter shares the centroid velocity.
    Returns (positions(k,3), velocities(k,3), centroid_pos(3,), centroid_vel(3,))."""
    c_pos, c_vel = centroid_state(t, traj)
    positions = c_pos[None, :] + offsets
    velocities = np.tile(c_vel, (len(offsets), 1))
    return positions, velocities, c_pos, c_vel


def compute_gdop(rx_positions: np.ndarray, target: np.ndarray) -> float:
    """Position GDOP (3D) for a TDoA array of receivers observing `target`.

    TDoA cancels the unknown transmit time, leaving 3 position unknowns; the
    per-receiver Jacobian is the unit line-of-sight, and each range-DIFFERENCE
    row (relative to reference receiver 0) is (u_i - u_0). With N receivers that
    is an (N-1, 3) geometry matrix G, and GDOP = sqrt(trace((G^T G)^-1)).

    Returns np.inf for a degenerate/rank-deficient layout (e.g. too collinear or
    coplanar to resolve 3D position) — the caller treats that as a hard failure.
    Needs N >= 4 receivers to make G full-column-rank at all.
    """
    rx = np.asarray(rx_positions, dtype=float)
    p = np.asarray(target, dtype=float)
    if rx.shape[0] < 4:
        return np.inf  # < 4 receivers cannot resolve 3D position via TDoA

    los = p[None, :] - rx
    dist = np.linalg.norm(los, axis=1)
    if np.any(dist < 1e-6):
        return np.inf  # target coincident with a receiver
    u = los / dist[:, None]
    G = u[1:] - u[0][None, :]            # (N-1, 3) differenced unit-LOS

    sv = np.linalg.svd(G, compute_uv=False)
    if sv.min() < 1e-9:
        return np.inf                    # rank-deficient geometry
    cov = np.linalg.inv(G.T @ G)
    tr = float(np.trace(cov))
    if not np.isfinite(tr) or tr < 0:
        return np.inf
    return float(np.sqrt(tr))
