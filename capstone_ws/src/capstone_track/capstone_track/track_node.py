"""capstone_track — Phase 4: fused TDoA localization (closed-form init + joint iterated EKF).

Blind, estimator-side node. Consumes:
  * /rf_observations  (RfObservationSet)  — per-burst ToA at each receiver (biased clock)
  * /receivers        (ReceiverStateArray) — receiver POSITIONS + the DISCIPLINED-array
                        clock_bias_sigma. The true per-rx clock bias is NEVER read here
                        (it lives only on /truth/receivers); clock_bias_sigma seeds the
                        EKF's informative clock prior (NOISE_MODEL §4a).
  * /associations     (BurstAssociationSet) — the base station's blind per-burst ->
                        global_id map + confidence, used as a soft, confidence-gated
                        data-association PRIOR (fingerprint identity, NOT truth).

Produces /tracks (TrackArray): one TrackState per discovered global_id.

Estimator design (see ekf.py + NOISE_MODEL §4a):
  * Joint state = per-track [x y z vx vy vz] + SHARED relative per-receiver clock
    [bias, drift]; the clock is ESTIMATED (with the disciplined-array prior), never read.
  * Measurement = TDoA only (range differences). Velocity comes from the CV model.
  * ITERATED EKF update (relinearize at the clock-corrupted init) — required for both
    accuracy and covariance sanity.
  * The per-receiver clock is only WEAKLY observable from a compact swarm (common
    formation translation ~= clock shift); the informative prior regularizes it, so the
    filter is known to be mildly overconfident — the published covariance is inflated by
    `covariance_inflation` so downstream threat assessment gets calibrated uncertainty.

Data association: a burst is fed to the EKF only once the base publishes a CONFIDENT
association for it (confidence > min_assoc_confidence); withheld / under-accumulated
bursts are skipped (they carry no trustworthy identity yet). This realizes the carried
Phase-3 caution: treat a fresh FingerprintAssignment as a prior, not ground truth.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from capstone_msgs.msg import (TrackArray, TrackState, RfObservationSet,
                               ReceiverStateArray, BurstAssociationSet)

from capstone_common.health import (HealthReporter, qos_best_effort,
                                    qos_reliable, qos_latched)
from capstone_track.ekf import JointEKF, tdoa_init, gdop


class TrackNode(Node):
    def __init__(self):
        super().__init__('capstone_track')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('publish_period_s', 0.1)          # /tracks at 10 Hz
        self.declare_parameter('min_assoc_confidence', 0.3)      # gate: below => skip burst
        self.declare_parameter('covariance_inflation', 3.0)      # published cov scale (filter is ~overconfident)
        self.declare_parameter('coast_timeout_s', 2.0)           # no update within => COASTING
        self.declare_parameter('lost_timeout_s', 8.0)            # => LOST
        self.declare_parameter('burst_drop_age_s', 30.0)         # incomplete/unassociated burst max wait
        pt = self.get_parameter('passthrough_truth').value
        self._min_conf = float(self.get_parameter('min_assoc_confidence').value)
        self._cov_infl = float(self.get_parameter('covariance_inflation').value)
        self._coast_t = float(self.get_parameter('coast_timeout_s').value)
        self._lost_t = float(self.get_parameter('lost_timeout_s').value)
        self._drop_age = float(self.get_parameter('burst_drop_age_s').value)

        # receiver geometry + disciplined-array clock prior (set on first /receivers)
        self._rx_pos = None          # (n_rx, 3)
        self._n_rx = None
        self._clock_sigma_s = None

        self._ekf = None
        self._pending = {}           # burst_seq -> {'t': sim_s, 'rx': {rx_id: (toa_s, sig)}}
        self._assoc = {}             # burst_seq -> (global_id, confidence)  (latest map)
        self._processed = set()      # burst_seqs already consumed (or skipped) by the EKF
        self._last_proc_t = None     # sim-time of the last EKF predict/update
        self._track_last_upd = {}    # global_id -> sim-time of last update
        self._track_nrx = {}         # global_id -> receivers used in last update
        self._track_updates = {}     # global_id -> count of updates (for INITIALIZING)

        self._pub = self.create_publisher(TrackArray, '/tracks', qos_reliable())
        self.health = HealthReporter(self, pt)
        self.create_subscription(RfObservationSet, '/rf_observations', self._on_obs, qos_reliable())
        self.create_subscription(ReceiverStateArray, '/receivers', self._on_receivers, qos_best_effort())
        self.create_subscription(BurstAssociationSet, '/associations', self._on_assoc, qos_latched())
        self.create_timer(float(self.get_parameter('publish_period_s').value), self._tick)

    # ---------------- helpers ----------------
    @staticmethod
    def _stamp_to_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    # ---------------- subscriptions ----------------
    def _on_receivers(self, msg):
        n = len(msg.receivers)
        if n == 0:
            return
        pos = np.zeros((n, 3))
        sig = None
        for r in msg.receivers:
            pos[r.rx_id] = [r.position.x, r.position.y, r.position.z]
            # disciplined-array residual (same for all rx here); belief bias is 0 and NOT used
            sig = r.clock_bias_sigma if r.clock_bias_sigma > 0 else sig
        self._rx_pos = pos
        if self._n_rx is None:
            self._n_rx = n
            self._clock_sigma_s = float(sig) if sig else 30e-9
            self._ekf = JointEKF(self._n_rx, ref=0, clock_bias_sigma_s=self._clock_sigma_s)
            self.get_logger().info(
                f'EKF initialized: n_rx={n} clock_bias_sigma={self._clock_sigma_s*1e9:.1f} ns '
                f'(disciplined-array prior; clock is estimated, not read)')

    def _on_assoc(self, msg):
        m = {}
        for a in msg.associations:
            m[a.burst_seq] = (a.global_id, float(a.confidence))
        self._assoc = m

    def _on_obs(self, msg):
        b = msg.burst_seq
        t = self._stamp_to_s(msg.header.stamp)
        e = self._pending.setdefault(b, {'t': t, 'rx': {}})
        for o in msg.observations:
            e['rx'][o.rx_id] = (self._stamp_to_s(o.toa), float(o.toa_sigma))

    # ---------------- main loop ----------------
    def _tick(self):
        if self._ekf is not None:
            self._process_pending()
            self._publish_tracks()
        self.health.bump()

    def _process_pending(self):
        now = None
        # process complete, confidently-associated bursts in burst_seq (≈time) order
        for b in sorted(self._pending):
            if b in self._processed:
                continue
            e = self._pending[b]
            have_full = len(e['rx']) >= self._n_rx
            assoc = self._assoc.get(b)
            if not have_full or assoc is None:
                # not ready yet — keep waiting unless it has aged out (then drop, in order)
                if self._oldest_age(e) > self._drop_age:
                    self._processed.add(b)          # give up on this burst; don't block the queue
                continue
            gid, conf = assoc
            if conf <= self._min_conf or not gid:
                self._processed.add(b)              # withheld/under-accumulated identity: skip
                continue
            self._feed_ekf(b, e, gid)
            self._processed.add(b)
        # bound memory: forget processed pending entries
        for b in list(self._pending):
            if b in self._processed:
                self._pending.pop(b, None)

    def _oldest_age(self, e):
        # sim-time age of a pending burst relative to the newest burst we've seen
        newest = max((x['t'] for x in self._pending.values()), default=e['t'])
        return newest - e['t']

    def _feed_ekf(self, b, e, gid):
        toas = np.array([e['rx'][r][0] for r in range(self._n_rx)])
        sigs = np.array([e['rx'][r][1] for r in range(self._n_rx)])
        t = e['t']
        if self._last_proc_t is None:
            self._last_proc_t = t
        dt = t - self._last_proc_t
        if dt > 0:
            self._ekf.predict(dt)
            self._last_proc_t = t
        if not self._ekf.has_track(gid):
            p0 = tdoa_init(self._rx_pos, toas, ref=0)
            self._ekf.init_track(gid, p0)
            self._track_updates[gid] = 0
            self.get_logger().info(f'new track {gid} init at {np.round(p0,1)} (burst {b})')
        else:
            self._ekf.update_burst(gid, self._rx_pos, toas, sigs)
        self._track_last_upd[gid] = t
        self._track_nrx[gid] = self._n_rx
        self._track_updates[gid] = self._track_updates.get(gid, 0) + 1

    def _status(self, gid):
        last = self._track_last_upd.get(gid)
        n = self._track_updates.get(gid, 0)
        if last is None or self._last_proc_t is None:
            return TrackState.STATUS_INITIALIZING
        age = self._last_proc_t - last
        if n < 2:
            return TrackState.STATUS_INITIALIZING
        if age <= self._coast_t:
            return TrackState.STATUS_TRACKING
        if age <= self._lost_t:
            return TrackState.STATUS_COASTING
        return TrackState.STATUS_LOST

    def _publish_tracks(self):
        arr = TrackArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'enu'
        for gid in sorted(self._ekf.tracks):
            p, v, cov6 = self._ekf.track_state(gid)
            t = TrackState()
            t.header = arr.header
            t.global_id = str(gid)
            t.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            t.velocity = Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))
            # publish INFLATED covariance — the filter is known to be ~2-3x overconfident
            # under the weak-observability null space (NOISE_MODEL §4a).
            t.covariance = [float(x) for x in (cov6 * self._cov_infl).flatten()]
            t.gdop = float(gdop(self._rx_pos, p)) if self._rx_pos is not None else 0.0
            t.n_receivers_used = int(self._track_nrx.get(gid, 0))
            t.status = self._status(gid)
            arr.tracks.append(t)
        self._pub.publish(arr)
        n_tr = len(arr.tracks)
        self.health.set_status(
            f'tracks={n_tr} assoc_map={len(self._assoc)} pending={len(self._pending)}')


def main():
    rclpy.init()
    node = TrackNode()
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
