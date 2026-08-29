"""Phase 2 RF observable + receiver-clock-nuisance model. Pure Python/numpy, NO ROS.

Turns Phase-1 truth (emitter poses, receiver poses) into the synthetic
measurements the multilateration chain consumes. Every number here is
pre-registered in capstone_scenarios/NOISE_MODEL.md (BLE tier, convention A);
this module must not deviate from those formulas.

Design invariants (per the locked plan):
  * CFO is a per-EMITTER hardware fingerprint, drawn ONCE per run (not per burst).
  * Clock bias+drift is a per-RECEIVER nuisance state, drawn ONCE per run, then
    bias(t) = bias0 + drift*t. Independent RNG streams from the scenario seed.
  * Measurement noise (toa/rssi/fdoa) is per-observation, zero-mean Gaussian at
    the range-dependent sigma from the CRLB.
  * The TRUE clock bias corrupts the reported ToA; the estimator is NOT told it
    (see sim_node: /truth/receivers carries the true bias, /receivers carries
    belief 0 + sigma), so multilateration is a genuine nuisance-estimation problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_MPS = 299_792_458.0

# Independent, reproducible RNG sub-streams keyed off the scenario seed. Distinct
# tags => statistically independent draws; formation offsets keep using the bare
# `seed` (truth_model) so Phase-1 output is unchanged.
_TAG_CFO = 0xCF0
_TAG_CLOCK = 0xC10C
_TAG_NOISE = 0x0B5


@dataclass(frozen=True)
class RfParams:
    protocol: str
    carrier_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    window_samples: int
    path_loss_exponent: float
    rssi_ref_dbm_at_1m: float
    noise_figure_db: float
    rssi_shadow_sigma_db: float
    cfo_ppm: float
    burst_rate_hz: float

    @property
    def b_rms_hz(self) -> float:
        return self.bandwidth_hz / np.sqrt(12.0)          # RMS(Gabor) BW, rectangular approx

    @property
    def noise_floor_dbm(self) -> float:
        return -174.0 + 10.0 * np.log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def window_T_s(self) -> float:
        return self.window_samples / self.sample_rate_hz


@dataclass(frozen=True)
class ClockParams:
    bias_max_ns: float
    drift_max_ns_per_s: float
    bias_sigma_ns: float


def load_rf(scn: dict) -> RfParams:
    rf = scn['rf']
    return RfParams(
        protocol=rf['protocol'], carrier_hz=float(rf['carrier_hz']),
        bandwidth_hz=float(rf['bandwidth_hz']), sample_rate_hz=float(rf['sample_rate_hz']),
        window_samples=int(rf['window_samples']),
        path_loss_exponent=float(rf['path_loss_exponent']),
        rssi_ref_dbm_at_1m=float(rf['rssi_ref_dbm_at_1m']),
        noise_figure_db=float(rf['noise_figure_db']),
        rssi_shadow_sigma_db=float(rf['rssi_shadow_sigma_db']),
        cfo_ppm=float(rf['cfo_ppm']), burst_rate_hz=float(rf['burst_rate_hz']))


def load_clock(scn: dict) -> ClockParams:
    c = scn['clock']
    return ClockParams(bias_max_ns=float(c['bias_max_ns']),
                       drift_max_ns_per_s=float(c['drift_max_ns_per_s']),
                       bias_sigma_ns=float(c['bias_sigma_ns']))


# ---- link budget / CRLB (convention A; d is 3D slant range in metres) ----

def rssi_dbm(rf: RfParams, d: float) -> float:
    d = max(float(d), 1.0)
    return rf.rssi_ref_dbm_at_1m - 10.0 * rf.path_loss_exponent * np.log10(d)


def snr_db(rf: RfParams, d: float) -> float:
    return rssi_dbm(rf, d) - rf.noise_floor_dbm


def sigma_toa_s(rf: RfParams, d: float) -> float:
    snr = 10.0 ** (snr_db(rf, d) / 10.0)
    return 1.0 / (2.0 * np.pi * rf.b_rms_hz * np.sqrt(snr))


def sigma_fdoa_hz(rf: RfParams, d: float) -> float:
    snr = 10.0 ** (snr_db(rf, d) / 10.0)
    return 1.0 / (2.0 * np.pi * rf.window_T_s * np.sqrt(snr))


def doppler_hz(rf: RfParams, emitter_pos, emitter_vel, rx_pos) -> float:
    """Doppler shift at a fixed receiver. f_d = -f_c/c * d(range)/dt, and with a
    stationary receiver the range rate is the emitter velocity projected on the
    receiver->emitter unit LOS (positive = receding => negative shift)."""
    los = np.asarray(emitter_pos, float) - np.asarray(rx_pos, float)
    d = np.linalg.norm(los)
    if d < 1e-9:
        return 0.0
    range_rate = float(np.dot(np.asarray(emitter_vel, float), los / d))
    return -rf.carrier_hz * range_rate / C_MPS


# ---- once-per-run nuisance draws (independent streams from the scenario seed) ----

def sample_emitter_cfo_hz(rf: RfParams, k: int, seed: int) -> np.ndarray:
    """Per-emitter CFO ~ U(±cfo_ppm of carrier), drawn once. Hardware fingerprint."""
    rng = np.random.default_rng([int(seed), _TAG_CFO])
    max_hz = rf.cfo_ppm * 1e-6 * rf.carrier_hz
    return rng.uniform(-max_hz, max_hz, size=k)


def sample_receiver_clocks(clk: ClockParams, n_rx: int, seed: int):
    """Per-receiver (bias0, drift) drawn once, in SECONDS and seconds/second.
    bias(t) = bias0 + drift*t. Distinct per receiver by construction."""
    rng = np.random.default_rng([int(seed), _TAG_CLOCK])
    bias0 = rng.uniform(-clk.bias_max_ns, clk.bias_max_ns, size=n_rx) * 1e-9
    drift = rng.uniform(-clk.drift_max_ns_per_s, clk.drift_max_ns_per_s, size=n_rx) * 1e-9
    return bias0, drift


def make_noise_rng(seed: int) -> np.random.Generator:
    """Per-observation measurement-noise stream (drawn sequentially, deterministic
    given the fixed tick schedule)."""
    return np.random.default_rng([int(seed), _TAG_NOISE])
