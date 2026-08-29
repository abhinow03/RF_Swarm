"""capstone_bridge — publishes the Option-C /scene_update JSON for the AR client.

/scene_update is std_msgs/String (JSON), RELIABLE + TRANSIENT_LOCAL depth 1, at
10 Hz. This node flattens the typed /tracks (joined with /fingerprints for
per-track confidence) into the frozen wire shape, doing the ENU->Y-up conversion
server-side. It is the ONLY place frame maths happens; the AR client never does any.

Option-C topic split (FROZEN, see scene-update-wire-format): /scene_update carries
TRACKS ONLY. /formation and /narration are NOT bundled into this 10 Hz envelope —
they are their own TRANSIENT_LOCAL (latched) topics with different rates + latching
semantics, and the client subscribes to them DIRECTLY over rosbridge (which JSON-
serializes the typed messages; they carry no frame maths, so no server flattening
is needed). The bridge still SUBSCRIBES to /predictions /formation /narration only
to drive the fault-isolation waiting-list below.

Fault isolation for upstreams that may lag on startup:
  * a one-shot startup log names which of /predictions /formation /narration have
    never been seen.
  * NodeHealth.status_text reports the waiting list while true, and clears to 'ok'
    once every upstream has published at least once.
"""
import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_msgs.msg import (TrackArray, FingerprintSet, PathPrediction,
                               FormationEstimate, Narration)

from capstone_common.health import HealthReporter, qos_reliable, qos_latched

SCHEMA_VERSION = 1
STATUS_NAME = {0: 'init', 1: 'tracking', 2: 'coasting', 3: 'lost'}


class BridgeNode(Node):
    def __init__(self):
        super().__init__('capstone_bridge')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('scenario_id', 'gate_k6')
        pt = self.get_parameter('passthrough_truth').value
        self._scenario_id = self.get_parameter('scenario_id').value

        self._scene_pub = self.create_publisher(String, '/scene_update', qos_latched())
        self.health = HealthReporter(self, pt)

        self._last_tracks = None
        self._last_fingerprints = None
        self._got_predictions = False
        self._got_formation = False
        self._last_formation = None
        self._got_narration = False
        self._last_narration = None
        self._startup_logged = False

        self.create_subscription(TrackArray, '/tracks', self._on_tracks, qos_reliable())
        self.create_subscription(FingerprintSet, '/fingerprints', self._on_fp, qos_latched())
        self.create_subscription(PathPrediction, '/predictions', self._on_pred, qos_reliable())
        self.create_subscription(FormationEstimate, '/formation', self._on_form, qos_latched())
        self.create_subscription(Narration, '/narration', self._on_narr, qos_latched())

        self.create_timer(0.1, self._tick)  # 10 Hz

    # --- upstream latches ---
    def _on_tracks(self, m):
        self._last_tracks = m

    def _on_fp(self, m):
        self._last_fingerprints = m

    def _on_pred(self, m):
        self._got_predictions = True

    def _on_form(self, m):
        self._got_formation = True
        self._last_formation = m

    def _on_narr(self, m):
        self._got_narration = True
        self._last_narration = m

    # --- helpers ---
    def _waiting(self):
        w = []
        if not self._got_predictions:
            w.append('/predictions')
        if not self._got_formation:
            w.append('/formation')
        if not self._got_narration:
            w.append('/narration')
        return w

    @staticmethod
    def _enu_to_yup(p):
        # ENU [E, N, U] -> Three.js Y-up [E, U, -N]
        return [p.x, p.z, -p.y]

    def _confidence_for(self, global_id):
        if self._last_fingerprints is None:
            return 0.0
        for a in self._last_fingerprints.assignments:
            if a.global_id == global_id:
                return float(a.confidence)
        return 0.0  # no assignment yet -> 0 (never hardcode a happy-path value)

    # --- main loop ---
    def _tick(self):
        now_ns = self.get_clock().now().nanoseconds

        env = {
            'schema_version': SCHEMA_VERSION,
            'scenario_id': self._scenario_id,
            'k_estimated': (int(self._last_fingerprints.k_estimated)
                            if self._last_fingerprints is not None else 0),
            'tracks': [],
        }

        if self._last_tracks is not None:
            for t in self._last_tracks.tracks:
                c = t.covariance
                pos_sigma = math.sqrt(max(0.0, (c[0] + c[7] + c[14]) / 3.0))
                env['tracks'].append({
                    'id': t.global_id,
                    'cls': 'swarm',
                    'confidence': self._confidence_for(t.global_id),
                    'pos': self._enu_to_yup(t.position),
                    'vel': self._enu_to_yup(t.velocity),
                    'status': STATUS_NAME.get(int(t.status), 'init'),
                    'pos_sigma': pos_sigma,
                    'gdop': float(t.gdop),
                })
            ts = self._last_tracks.header.stamp
            track_ns = ts.sec * 1_000_000_000 + ts.nanosec
            env['pipeline_latency_ms'] = (now_ns - track_ns) / 1e6
        else:
            env['pipeline_latency_ms'] = -1.0  # sentinel: no upstream tracks yet

        # Option C: formation/narration are NOT bundled here — the client subscribes
        # to the latched /formation and /narration topics directly. The bridge only
        # tracks their arrival (above) for the waiting-list health below.

        msg = String()
        msg.data = json.dumps(env)
        self._scene_pub.publish(msg)
        self.health.bump()

        waiting = self._waiting()
        if not self._startup_logged:
            if waiting:
                self.get_logger().warn(
                    'no upstream yet on: ' + ' '.join(waiting)
                    + ' — normal at startup until predict/formation/narrate warm up')
            self._startup_logged = True
        if waiting:
            self.health.set_status('waiting (no producer yet): ' + ' '.join(waiting))
        else:
            self.health.set_status('ok')


def main():
    rclpy.init()
    node = BridgeNode()
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
