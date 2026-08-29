"""capstone_formation — Phase 5: STGT formation classifier hookup.

Feeds REAL EKF localization output (/tracks) into the trained STGT classifier and
publishes FormationEstimate on /formation. This is the whole point of the phase's
gate: classify from real localization, not parametric truth. A `from_truth` param
swaps the input to /truth/scene (debug/isolation only) — the gate itself runs on
/tracks.

Sequence assembly: snapshot the K=6 emitter positions every `sample_dt_s` of SIM
time into a rolling window of `window` frames (the model's max_seq_len=50 at the
training dt of 0.5 s => a 25 s window), then classify. The model is permutation-
invariant over nodes (global_mean_pool), so per-frame drone ordering is irrelevant;
it needs exactly 6 nodes (trained on 6 drones), so this runs at the K=6 scenarios.
"""
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from capstone_msgs.msg import FormationEstimate, TrackArray, TruthScene

from capstone_common.health import HealthReporter, qos_best_effort, qos_reliable, qos_latched
from capstone_formation.classifier import FormationClassifier

# Self-contained in-RF STGT weights (Part-2 packaging). best_model.pt + train_mean/std
# copied byte-for-byte from ~/aadhya/uav-llm/swarm_data (see DEPENDENCY_MANIFEST.md).
DEFAULT_MODEL_DIR = '/home/pw26_akp_01/RF_Swarm/models/stgt'


class FormationNode(Node):
    def __init__(self):
        super().__init__('capstone_formation')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('from_truth', False)          # debug: classify /truth/scene
        self.declare_parameter('model_dir', DEFAULT_MODEL_DIR)
        self.declare_parameter('sample_dt_s', 0.5)           # matches training dt
        self.declare_parameter('publish_period_s', 0.5)
        self.declare_parameter('min_confidence_report', 0.0)  # always publish; consumer gates
        pt = self.get_parameter('passthrough_truth').value
        self._from_truth = bool(self.get_parameter('from_truth').value)
        self._sample_dt = float(self.get_parameter('sample_dt_s').value)
        model_dir = self.get_parameter('model_dir').value

        self.get_logger().info(f'loading STGT classifier from {model_dir} (CPU torch)...')
        self._clf = FormationClassifier(model_dir)
        self._win = self._clf.window
        self._buf = deque(maxlen=self._win)                  # each: (member_ids, pos(6,3))
        self._last_snap_t = None

        self._pub = self.create_publisher(FormationEstimate, '/formation', qos_latched())
        self.health = HealthReporter(self, pt)
        if self._from_truth:
            self.create_subscription(TruthScene, '/truth/scene', self._on_truth, qos_best_effort())
            self.get_logger().warn('from_truth=TRUE — classifying PARAMETRIC TRUTH (debug/isolation, not the gate)')
        else:
            self.create_subscription(TrackArray, '/tracks', self._on_tracks, qos_reliable())
        self.create_timer(float(self.get_parameter('publish_period_s').value), self._tick)

    @staticmethod
    def _stamp_to_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _maybe_snapshot(self, t, member_ids, pos):
        if self._last_snap_t is None or (t - self._last_snap_t) >= self._sample_dt - 1e-6:
            self._last_snap_t = t
            self._buf.append((member_ids, pos))

    def _on_tracks(self, msg):
        # need exactly the 6-drone set with finite positions
        items = [(t.global_id, [t.position.x, t.position.y, t.position.z]) for t in msg.tracks]
        items = [(g, p) for g, p in items if np.all(np.isfinite(p))]
        if len(items) < self._clf.n_nodes:
            return
        items = sorted(items)[:self._clf.n_nodes]
        member_ids = [g for g, _ in items]
        pos = np.array([p for _, p in items], float)
        self._maybe_snapshot(self._stamp_to_s(msg.header.stamp), member_ids, pos)

    def _on_truth(self, msg):
        items = [(e.emitter_id, [e.position.x, e.position.y, e.position.z]) for e in msg.emitters]
        if len(items) < self._clf.n_nodes:
            return
        items = sorted(items)[:self._clf.n_nodes]
        member_ids = [f'e{i}' for i, _ in items]
        pos = np.array([p for _, p in items], float)
        self._maybe_snapshot(self._stamp_to_s(msg.header.stamp), member_ids, pos)

    def _tick(self):
        n = len(self._buf)
        if n < self._win:
            self.health.set_status(f'buffering ({n}/{self._win} frames)')
            self.health.bump()
            return
        seq = np.stack([p for _, p in self._buf], axis=0)     # (T, 6, 3)
        out = self._clf.classify(seq)
        member_ids = self._buf[-1][0]

        m = FormationEstimate()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'enu'
        m.formation_class = out['formation_type']
        m.confidence = float(out['formation_confidence'])
        m.class_probabilities = self._clf.prob_vector(out)
        m.member_ids = list(member_ids)
        m.from_truth = self._from_truth
        self._pub.publish(m)
        self.health.set_status(
            f'{"TRUTH" if self._from_truth else "EKF"} formation={m.formation_class} '
            f'conf={m.confidence:.3f}')
        self.health.bump()
        self.get_logger().info(
            f'[{"truth" if self._from_truth else "ekf"}] formation={m.formation_class} '
            f'conf={m.confidence:.3f}')


def main():
    rclpy.init()
    node = FormationNode()
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
