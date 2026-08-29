"""capstone_base — Phase 3 base station: blind fingerprint discovery.

Consumes /embeddings (NO truth), pools each burst_seq's per-receiver embeddings
into one per-burst track (same emission => same unit), accumulates tracks, and
runs the canonical eigengap K-estimation + spectral partition to discover K and
assign global IDs d1..dK. Publishes FingerprintSet with REAL eigenvalues /
eigengap_value / n_accumulated.

ENFORCED ACCUMULATION GATE (real logic, not a sim guarantee): a discovered unit
whose accumulated clean-track count is below MIN_TRACKS gets confidence 0 — an
explicitly-withheld assignment, visibly distinct from a normal one — because the
eigengap K-estimate is only trustworthy once each unit is well-accumulated (the
K̂=3 collapse was insufficient accumulation, not a BLE ceiling). This must hold on
live hardware too, where a quiet emitter or dropped receiver can under-accumulate a
unit; the sweep DEMONSTRATES the threshold rather than running long enough to dodge it.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from capstone_msgs.msg import (FingerprintSet, FingerprintAssignment, Embedding,
                               BurstAssociation, BurstAssociationSet)

from capstone_common.health import HealthReporter, qos_latched, qos_reliable
from capstone_base import fingerprint as fp

CLUSTER_FLOOR = 12        # min total tracks before attempting eigengap at all


class BaseNode(Node):
    def __init__(self):
        super().__init__('capstone_base')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('min_tracks_confident', 15)   # enforced accumulation threshold
        self.declare_parameter('publish_period_s', 3.0)
        pt = self.get_parameter('passthrough_truth').value
        self._min_tracks = int(self.get_parameter('min_tracks_confident').value)

        self._pub = self.create_publisher(FingerprintSet, '/fingerprints', qos_latched())
        # Full per-burst association map (blind) for the tracker's data association —
        # the base already computes `labels` for every accumulated burst; publishing
        # the whole current map (idempotent, latched) lets the EKF route each burst by
        # fingerprint identity and treat confidence as a soft prior (Phase-4 decision).
        self._assoc_pub = self.create_publisher(BurstAssociationSet, '/associations', qos_latched())
        self.health = HealthReporter(self, pt)

        self._acc = {}            # burst_seq -> list[np.ndarray] (per-rx vectors)
        self._max_seq = -1
        self._tracks = []         # list[(burst_seq, pooled_vec)]
        self._sha = ''
        self.create_subscription(Embedding, '/embeddings', self._on_emb, qos_reliable())
        self.create_timer(float(self.get_parameter('publish_period_s').value), self._tick)

    def _on_emb(self, m):
        self._acc.setdefault(m.burst_seq, []).append(np.asarray(m.vector, np.float32))
        self._max_seq = max(self._max_seq, m.burst_seq)
        if not self._sha:
            self._sha = m.encoder_sha256

    def _finalize_old_bursts(self, guard=2):
        # a burst_seq <= max_seq-guard has surely received all its receivers
        done = [b for b in self._acc if b <= self._max_seq - guard]
        for b in sorted(done):
            v = np.mean(self._acc.pop(b), axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)               # per-burst track: pool rx + L2
            self._tracks.append((b, v))

    def _tick(self):
        self._finalize_old_bursts()
        n = len(self._tracks)

        m = FingerprintSet()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'enu'
        m.encoder_sha256 = self._sha or 'unknown'
        m.n_accumulated = n

        if n < CLUSTER_FLOOR:
            m.k_estimated = 0                # still accumulating; no discovery yet
            m.eigenvalues = []
            m.eigengap_value = 0.0
            m.assignments = []
            self._pub.publish(m)
            self._publish_associations([], {}, {}, {})   # empty map (still accumulating)
            self.health.set_status(f'accumulating ({n} tracks < {CLUSTER_FLOOR})')
            self.health.bump()
            return

        X = np.array([v for _, v in self._tracks])
        khat, vals, gap, labels, k_raw = fp.discover(X)
        m.k_estimated = int(khat)
        m.eigenvalues = [float(x) for x in vals]   # raw normalized-Laplacian spectrum
        m.eigengap_value = float(gap)

        # global IDs by first-seen cluster order; per-cluster confidence + accum gate
        order, counts = [], {}
        for lb in labels:
            if lb not in counts:
                order.append(lb)
            counts[lb] = counts.get(lb, 0) + 1
        gid = {lb: f'd{i+1}' for i, lb in enumerate(order)}
        rep = {}                              # representative (latest) burst_seq per cluster
        for (bseq, _), lb in zip(self._tracks, labels):
            rep[lb] = max(rep.get(lb, -1), bseq)

        sep = fp.separability(X, labels)      # per-cluster silhouette = identity confidence
        assigns = []
        withheld = 0
        for lb in order:
            n_lb = counts[lb]
            # withhold (conf 0) if UNDER-ACCUMULATED (count gate) OR not yet SEPARABLE
            # (silhouette gate) — the latter catches the low-diversity failure the
            # count gate alone misses (finding 2). Confidence is the separability.
            conf = 0.0 if n_lb < self._min_tracks else sep.get(lb, 0.0)
            if n_lb < self._min_tracks:
                withheld += 1
            a = FingerprintAssignment()
            a.burst_seq = int(rep[lb])
            a.global_id = gid[lb]
            a.confidence = float(conf)        # 0 => explicitly withheld (under-accumulated)
            assigns.append(a)
        m.assignments = assigns
        self._pub.publish(m)

        # Publish the FULL per-burst association map (same conf/gate logic as the
        # per-cluster assignments): each accumulated burst -> its cluster's global_id
        # and confidence (0 => withheld). Idempotent snapshot the tracker adopts.
        conf_by_lb = {lb: (0.0 if counts[lb] < self._min_tracks else sep.get(lb, 0.0))
                      for lb in order}
        self._publish_associations(labels, gid, conf_by_lb, {'k': int(khat)})

        self.health.set_status(
            f'K_hat={khat} tracks={n} withheld(<{self._min_tracks})={withheld}')
        self.health.bump()
        self.get_logger().info(
            f'K_hat={khat} (raw_eigengap={k_raw}->merged) gap={gap:.4f} tracks={n} '
            f'cluster_counts={[counts[lb] for lb in order]} '
            f'conf={[round(a.confidence,3) for a in assigns]}')


    def _publish_associations(self, labels, gid, conf_by_lb, meta):
        """Publish the current blind per-burst -> global_id map on /associations.
        `labels` aligns with self._tracks (each (burst_seq, vec)); `gid` maps cluster
        label -> 'dK'; `conf_by_lb` maps label -> confidence. Empty when accumulating."""
        s = BurstAssociationSet()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = 'enu'
        s.encoder_sha256 = self._sha or 'unknown'
        s.k_estimated = int(meta.get('k', 0))
        for (bseq, _), lb in zip(self._tracks, labels):
            a = BurstAssociation()
            a.burst_seq = int(bseq)
            a.global_id = gid.get(lb, '')
            a.confidence = float(conf_by_lb.get(lb, 0.0))
            s.associations.append(a)
        self._assoc_pub.publish(s)


def main():
    rclpy.init()
    node = BaseNode()
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
