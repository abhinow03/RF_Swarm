"""capstone_sim — Phase 1: the time authority + scripted TRUTH pose source.

Publishes real kinematics for K emitters (scripted trajectory through a fixed
base-station receiver array) and the receiver states, at 50 Hz. Ground truth on
/truth/scene + /truth/receivers; the estimator-visible receiver copy on
/receivers — for Phase 1 /receivers == /truth/receivers EXACTLY (no receiver
noise yet; that is Phase 2's measurement/estimation error, not kinematics).

Forward-only: the centroid trajectory is the INPUT and per-emitter truth pose =
centroid + a rigid formation offset taken VERBATIM from the STGT-side geometry
(swarm_intent.formations.get_formation_offsets, pinned commit 69ac2b6). Nothing
here solves backward from assumed observations. This phase produces TRUTH ONLY —
no observations; /rf_observations + /truth/association remain Phase-2 zero-stubs.

Scenario (k, formation_label, receivers, trajectory, seed, ...) is loaded from a
capstone_scenarios YAML — never hardcoded here. Select with the `scenario` param
(default 'gate_k6'); override the file directly with `scenario_path`.

Time model unchanged from Phase 0: a 50 Hz master timer on an explicit
SYSTEM_TIME clock advances the sim clock and publishes /clock, so this node stays
the clock authority even under use_sim_time:=true.

Phase 1 gate: poses at 50 Hz; GDOP for the receiver geometry computed and printed
per timestep and confirmed finite (a degenerate array is caught here, loudly,
before any later phase silently inherits an unsolvable geometry).
"""
import os

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Vector3
from builtin_interfaces.msg import Time as TimeMsg
from rosgraph_msgs.msg import Clock as ClockMsg
from ament_index_python.packages import get_package_share_directory
from capstone_msgs.msg import (TruthScene, EmitterTruth, ReceiverStateArray,
                               ReceiverState, RfObservationSet, RfObservation,
                               BurstAssociationTruth)

from capstone_common.health import HealthReporter, qos_best_effort, qos_reliable
from capstone_sim.truth_model import formation_offsets, emitter_states, compute_gdop
from capstone_sim import rf_model

RATE_HZ = 50
# tx_power_dbm is NOT the RSSI driver under convention A (RSSI uses the adopted
# reference scale rssi_ref_dbm_at_1m, see NOISE_MODEL.md); published as a neutral
# placeholder so no untraceable TX power is asserted.
TX_POWER_PLACEHOLDER_DBM = 0.0


class SimNode(Node):
    def __init__(self):
        super().__init__('capstone_sim')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('scenario', 'gate_k6')
        self.declare_parameter('scenario_path', '')
        pt = self.get_parameter('passthrough_truth').value

        self._scn = self._load_scenario()
        self._offsets = formation_offsets(
            self._scn['formation_label'], self._scn['k'],
            spread=float(self._scn.get('spread', 1.0)),
            seed=int(self._scn.get('seed', 0)))
        self._rx_pos = np.asarray(self._scn['receivers'], dtype=float)   # (N,3)
        self._traj = self._scn['trajectory']
        self._duration = float(self._scn.get('duration_s', 60.0))
        self._k = int(self._scn['k'])
        self._n_rx = self._rx_pos.shape[0]

        # Phase 2 RF model + once-per-run nuisance draws (independent seed streams)
        self._rf = rf_model.load_rf(self._scn)
        self._clk = rf_model.load_clock(self._scn)
        seed = int(self._scn.get('seed', 0))
        self._cfo_hz = rf_model.sample_emitter_cfo_hz(self._rf, self._k, seed)      # per-emitter, once
        self._bias0, self._drift = rf_model.sample_receiver_clocks(self._clk, self._n_rx, seed)  # per-rx, once
        self._noise = rf_model.make_noise_rng(seed)
        self._emit_count = np.zeros(self._k, dtype=int)   # bursts emitted per emitter (staggered cadence)

        # publishers (topics + QoS unchanged from Phase 0)
        self._clock_pub = self.create_publisher(ClockMsg, '/clock', qos_best_effort())
        self._truth_scene_pub = self.create_publisher(TruthScene, '/truth/scene', qos_best_effort())
        self._truth_rx_pub = self.create_publisher(ReceiverStateArray, '/truth/receivers', qos_best_effort())
        self._rx_pub = self.create_publisher(ReceiverStateArray, '/receivers', qos_best_effort())
        self._assoc_pub = self.create_publisher(BurstAssociationTruth, '/truth/association', qos_reliable())
        self._obs_pub = self.create_publisher(RfObservationSet, '/rf_observations', qos_reliable())

        self.health = HealthReporter(self, pt)

        self._sim_ns = 0
        self._dt_ns = int(1e9 / RATE_HZ)
        self._sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self._master = self.create_timer(1.0 / RATE_HZ, self._on_tick, clock=self._sys_clock)
        self._tick = 0
        self._burst_seq = 0
        self._degenerate_logged = False

        self._log_startup_geometry()

    # ---------------- scenario loading ----------------
    def _load_scenario(self):
        path = self.get_parameter('scenario_path').value
        if not path:
            name = self.get_parameter('scenario').value
            share = get_package_share_directory('capstone_scenarios')
            path = os.path.join(share, 'scenarios', name + '.yaml')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'scenario file not found: {path}')
        with open(path) as f:
            scn = yaml.safe_load(f)
        self.get_logger().info(f'loaded scenario {scn.get("scenario_id")!r} from {path}')
        return scn

    def _log_startup_geometry(self):
        c0, _ = self._centroid(0.0)
        g0 = compute_gdop(self._rx_pos, c0)
        self.get_logger().info(
            f'scenario_id={self._scn["scenario_id"]} k={self._k} '
            f'formation={self._scn["formation_label"]} n_rx={self._n_rx}')
        self.get_logger().info(f'receiver array (ENU m):\n{self._rx_pos}')
        self.get_logger().info(f'initial centroid={c0}  GDOP={g0:.3f}  finite={np.isfinite(g0)}')
        if not np.isfinite(g0):
            self.get_logger().error(
                'DEGENERATE receiver geometry at t=0 — GDOP is not finite; '
                'every downstream localization phase would be unsolvable. Fix the '
                'receiver layout in the scenario YAML before proceeding.')
        # show the per-emitter / per-receiver nuisance states are DISTINCT (not shared)
        self.get_logger().info(
            'RF: protocol=%s  noise_floor=%.1f dBm  B_rms=%.0f kHz  window_T=%.1f us'
            % (self._rf.protocol, self._rf.noise_floor_dbm, self._rf.b_rms_hz / 1e3,
               self._rf.window_T_s * 1e6))
        self.get_logger().info('per-emitter CFO (kHz): '
                               + np.array2string(self._cfo_hz / 1e3, precision=2))
        self.get_logger().info('per-receiver clock bias0 (ns): '
                               + np.array2string(self._bias0 * 1e9, precision=1))
        self.get_logger().info('per-receiver clock drift (ns/s): '
                               + np.array2string(self._drift * 1e9, precision=3))

    # ---------------- time helpers (unchanged) ----------------
    def _stamp(self):
        t = TimeMsg()
        t.sec = self._sim_ns // 1_000_000_000
        t.nanosec = self._sim_ns % 1_000_000_000
        return t

    def _header(self, frame='enu'):
        h = Header()
        h.stamp = self._stamp()
        h.frame_id = frame
        return h

    def _sim_t(self):
        # clamp trajectory time to duration: the swarm holds at its final (good-GDOP)
        # pose after duration_s, so long runs don't drift into bad geometry.
        return min(self._sim_ns / 1e9, self._duration)

    def _centroid(self, t):
        from capstone_sim.truth_model import centroid_state
        return centroid_state(t, self._traj)

    # ---------------- main loop ----------------
    def _on_tick(self):
        self._sim_ns += self._dt_ns
        cm = ClockMsg()
        cm.clock = self._stamp()
        self._clock_pub.publish(cm)

        t = self._sim_t()
        pos, vel, c_pos, c_vel = emitter_states(t, self._offsets, self._traj)
        if self._sim_ns / 1e9 > self._duration:
            # holding at the final pose after duration_s: position is frozen, so
            # truth velocity must be zero (keep vel == d(pos)/dt for the EKF).
            vel = np.zeros_like(vel)
            c_vel = np.zeros(3)

        self._publish_truth_scene(pos, vel)
        self._publish_receivers()
        self._check_and_report_gdop(c_pos)

        self._generate_bursts(pos, vel)   # per-emitter RF bursts (staggered cadence)

        self._tick += 1
        self.health.bump()

    def _publish_truth_scene(self, pos, vel):
        msg = TruthScene()
        msg.header = self._header()
        msg.k_true = self._k
        msg.formation_label = self._scn['formation_label']
        msg.scenario_id = self._scn['scenario_id']
        for i in range(self._k):
            e = EmitterTruth()
            e.header = msg.header
            e.emitter_id = i
            e.hardware_uid = ''                       # Phase 2: fingerprint binding
            e.position = Point(x=float(pos[i, 0]), y=float(pos[i, 1]), z=float(pos[i, 2]))
            e.velocity = Vector3(x=float(vel[i, 0]), y=float(vel[i, 1]), z=float(vel[i, 2]))
            e.cfo_hz = float(self._cfo_hz[i])         # TRUE per-emitter CFO (hardware fingerprint)
            e.tx_power_dbm = TX_POWER_PLACEHOLDER_DBM  # not the RSSI driver (convention A)
            msg.emitters.append(e)
        self._truth_scene_pub.publish(msg)

    def _make_rx_array(self, true_clock):
        """true_clock=True  -> /truth/receivers: the TRUE per-receiver clock nuisance.
        true_clock=False -> /receivers: estimator-visible belief (bias 0) + the stated
        residual uncertainty only. The estimator is NOT told the true bias, so
        multilateration stays a genuine per-receiver nuisance-estimation problem.
        Receiver POSITION is truth on both topics (no position noise in Phase 2)."""
        arr = ReceiverStateArray()
        arr.header = self._header()
        t = self._sim_ns / 1e9
        for i in range(self._n_rx):
            r = ReceiverState()
            r.header = arr.header
            r.rx_id = i
            r.position = Point(x=float(self._rx_pos[i, 0]),
                               y=float(self._rx_pos[i, 1]),
                               z=float(self._rx_pos[i, 2]))
            r.velocity = Vector3()                    # fixed base-station array
            r.position_covariance = [0.0] * 9         # position is truth this phase
            if true_clock:
                r.clock_bias = float(self._bias0[i] + self._drift[i] * t)  # true offset(t)
                r.clock_drift = float(self._drift[i])
                r.clock_bias_sigma = 0.0              # truth is exact
            else:
                r.clock_bias = 0.0                    # belief: receiver thinks it is synced
                r.clock_drift = 0.0
                r.clock_bias_sigma = self._clk.bias_sigma_ns * 1e-9  # stated residual
            arr.receivers.append(r)
        return arr

    def _publish_receivers(self):
        self._truth_rx_pub.publish(self._make_rx_array(true_clock=True))
        self._rx_pub.publish(self._make_rx_array(true_clock=False))

    def _check_and_report_gdop(self, c_pos):
        g = compute_gdop(self._rx_pos, c_pos)
        if not np.isfinite(g):
            self.health.set_status('DEGENERATE geometry: GDOP not finite')
            if not self._degenerate_logged:
                self.get_logger().error(
                    f'GDOP not finite at centroid={c_pos} — degenerate receiver geometry')
                self._degenerate_logged = True
            return
        if self._tick % RATE_HZ == 0:                 # ~1 Hz print
            self.get_logger().info(
                f't={self._sim_t():5.1f}s  centroid={np.round(c_pos, 2)}  GDOP={g:.3f}')

    def _time_from_s(self, t_s):
        ns = int(round(t_s * 1e9))
        m = TimeMsg()
        m.sec = ns // 1_000_000_000
        m.nanosec = ns % 1_000_000_000
        return m

    def _generate_bursts(self, pos, vel):
        """Fixed-cadence bursts at rf.burst_rate_hz per emitter, staggered by
        emitter so they interleave. Each due emitter -> one burst = one
        RfObservation per receiver (joined by burst_seq) + a BurstAssociationTruth."""
        t_now = self._sim_ns / 1e9
        rate = self._rf.burst_rate_hz
        for e in range(self._k):
            n_due = int(np.floor(t_now * rate + e / self._k))
            if n_due <= self._emit_count[e]:
                continue
            self._emit_count[e] = n_due
            self._burst_seq += 1
            self._emit_one_burst(e, pos[e], vel[e], t_now)

    def _emit_one_burst(self, e, epos, evel, t_emit):
        s = RfObservationSet()
        s.header = self._header()
        s.burst_seq = self._burst_seq
        for r in range(self._n_rx):
            rx = self._rx_pos[r]
            d = float(np.linalg.norm(epos - rx))              # 3D slant range
            sig_toa = rf_model.sigma_toa_s(self._rf, d)
            sig_fd = rf_model.sigma_fdoa_hz(self._rf, d)
            true_bias = float(self._bias0[r] + self._drift[r] * t_emit)   # per-rx nuisance
            toa_s = (t_emit + d / rf_model.C_MPS + true_bias
                     + float(self._noise.normal(0.0, sig_toa)))
            o = RfObservation()
            o.header = s.header
            o.rx_id = r
            o.burst_seq = self._burst_seq
            o.toa = self._time_from_s(toa_s)                  # receiver (biased) clock
            o.toa_sigma = sig_toa
            o.rssi_dbm = (rf_model.rssi_dbm(self._rf, d)
                          + float(self._noise.normal(0.0, self._rf.rssi_shadow_sigma_db)))
            o.rssi_sigma = self._rf.rssi_shadow_sigma_db
            o.fdoa_hz = (float(self._cfo_hz[e])               # per-emitter CFO (fingerprint)
                         + rf_model.doppler_hz(self._rf, epos, evel, rx)  # + Doppler (truth)
                         + float(self._noise.normal(0.0, sig_fd)))
            o.fdoa_sigma = sig_fd
            o.aoa_azimuth = float('nan')                      # single-antenna: no AoA
            o.aoa_elevation = float('nan')
            o.aoa_sigma = float('nan')
            o.aoa_valid = False
            o.snr_db = rf_model.snr_db(self._rf, d)
            o.nominal_freq_hz = self._rf.carrier_hz
            s.observations.append(o)
        self._obs_pub.publish(s)

        a = BurstAssociationTruth()
        a.header = s.header
        a.burst_seq = self._burst_seq
        a.truth_emitter_id = e
        self._assoc_pub.publish(a)


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
