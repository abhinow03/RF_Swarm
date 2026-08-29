"""Record CV-baseline path-prediction RMSE (the permanent comparison number).

Runs alongside the pipeline (sim rff base track predict). Buffers /predictions
(cv_baseline waypoints) and the truth timeline; maps each track global_id to its
true emitter via /truth/association o /associations; scores each predicted waypoint
against the true emitter position at the waypoint's future time, bucketed by
prediction horizon. RMSE here folds in Phase-4's EKF position/velocity error
(NOISE_MODEL §4a) plus CV extrapolation error — that is the honest baseline.

args: <duration_s>
"""
import sys, time
import numpy as np
from collections import defaultdict, Counter
import rclpy
from rclpy.node import Node
from capstone_msgs.msg import PathPrediction, TruthScene, BurstAssociationTruth, BurstAssociationSet
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

def be(d=10): return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=d)
def rel(d=50): return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=d)
def latched(d=1): return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=d)


class CVEval(Node):
    def __init__(self):
        super().__init__('cv_eval')
        self.truth = defaultdict(list)     # emitter_id -> [(t, pos)]
        self.preds = []                    # (t0, gid, [(tw, pos)...])
        self.truth_assoc = {}              # burst -> emitter
        self.blind_assoc = {}              # burst -> gid
        self.create_subscription(TruthScene, '/truth/scene', self._truth, be())
        self.create_subscription(PathPrediction, '/predictions', self._pred, rel())
        self.create_subscription(BurstAssociationTruth, '/truth/association', self._ta, rel())
        self.create_subscription(BurstAssociationSet, '/associations', self._ba, latched())

    def _truth(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        for e in m.emitters:
            self.truth[e.emitter_id].append((t, np.array([e.position.x, e.position.y, e.position.z])))

    def _pred(self, m):
        t0 = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        wps = [(wt.sec+wt.nanosec*1e-9, np.array([p.x, p.y, p.z]))
               for wt, p in zip(m.waypoint_times, m.waypoints)]
        self.preds.append((t0, m.global_id, wps))

    def _ta(self, m): self.truth_assoc[m.burst_seq] = m.truth_emitter_id
    def _ba(self, m):
        for a in m.associations:
            if a.confidence > 0: self.blind_assoc[a.burst_seq] = a.global_id

    def gid_map(self):
        votes = defaultdict(Counter)
        for b, g in self.blind_assoc.items():
            te = self.truth_assoc.get(b)
            if te is not None: votes[g][te] += 1
        return {g: c.most_common(1)[0][0] for g, c in votes.items()}

    def truth_at(self, emitter, tw):
        arr = self.truth.get(emitter)
        if not arr: return None
        ts = np.array([t for t, _ in arr])
        i = int(np.argmin(np.abs(ts - tw)))
        if abs(ts[i] - tw) > 0.5: return None      # no truth sample near tw yet
        return arr[i][1]


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    rclpy.init(); node = CVEval(); t0 = time.time()
    while time.time()-t0 < dur and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
    gmap = node.gid_map()
    buckets = defaultdict(list)                      # rounded-horizon -> [err]
    for (tp, gid, wps) in node.preds:
        em = gmap.get(gid)
        if em is None: continue
        for (tw, pp) in wps:
            tp_truth = node.truth_at(em, tw)
            if tp_truth is None: continue
            h = round(tw - tp)
            buckets[h].append(float(np.linalg.norm(pp - tp_truth)))
    print("\n===== CV-BASELINE PATH-PREDICTION RMSE (model_id=cv_baseline) =====")
    print(f"gid->emitter map: {gmap}   predictions scored: {sum(len(v) for v in buckets.values())}")
    allerr = []
    for h in sorted(buckets):
        e = np.array(buckets[h]); allerr += list(e)
        print(f"  horizon ~{h}s: RMSE={np.sqrt(np.mean(e**2)):6.2f}m  mean={e.mean():6.2f}m  n={len(e)}")
    if allerr:
        allerr = np.array(allerr)
        print(f"  OVERALL: RMSE={np.sqrt(np.mean(allerr**2)):.2f}m  (this is the number an ML path model must beat)")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
