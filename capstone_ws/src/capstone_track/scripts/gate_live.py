"""Live Phase-4 gate: score /tracks position error vs /truth/scene.

Run (two shells, after `source activate.sh` in each):
    # shell 1 — the pipeline
    bash scripts/run_pipeline.sh gate_k6 sim rff base track
    # shell 2 — the gate (let accumulation + confident tracking settle)
    python3 src/capstone_track/scripts/gate_live.py 100
Expected: 6 tracks TRACKING, association BIJECTIVE (no swaps), position RMSE <30 m.


Blind mapping: compose /truth/association (burst->truth_emitter) with /associations
(burst->global_id) to learn which truth emitter each track global_id corresponds to
(majority vote over shared bursts), then compare each published track to ITS mapped
truth emitter. Reports per-track error, RMSE, and whether the mapping is a bijection
(no two tracks claiming the same emitter => association not swapped).
"""
import sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from collections import defaultdict, Counter

from capstone_msgs.msg import (TrackArray, TruthScene, BurstAssociationTruth,
                               BurstAssociationSet)
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def be(depth=10):
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=depth)
def rel(depth=10):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=depth)
def latched(depth=1):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class Eval(Node):
    def __init__(self):
        super().__init__('phase4_eval')
        self.truth = {}                    # emitter_id -> pos(3)
        self.tracks = {}                   # global_id -> (pos(3), status, gdop, nrx)
        self.truth_assoc = {}              # burst -> truth_emitter
        self.blind_assoc = {}              # burst -> global_id
        self.create_subscription(TruthScene, '/truth/scene', self._truth, be())
        self.create_subscription(TrackArray, '/tracks', self._tracks, rel())
        self.create_subscription(BurstAssociationTruth, '/truth/association', self._ta, rel())
        self.create_subscription(BurstAssociationSet, '/associations', self._ba, latched())

    def _truth(self, m):
        for e in m.emitters:
            self.truth[e.emitter_id] = np.array([e.position.x, e.position.y, e.position.z])

    def _tracks(self, m):
        for t in m.tracks:
            self.tracks[t.global_id] = (np.array([t.position.x, t.position.y, t.position.z]),
                                        t.status, t.gdop, t.n_receivers_used)

    def _ta(self, m):
        self.truth_assoc[m.burst_seq] = m.truth_emitter_id

    def _ba(self, m):
        for a in m.associations:
            if a.confidence > 0:
                self.blind_assoc[a.burst_seq] = a.global_id

    def gid_to_emitter(self):
        votes = defaultdict(Counter)
        for burst, gid in self.blind_assoc.items():
            te = self.truth_assoc.get(burst)
            if te is not None:
                votes[gid][te] += 1
        return {gid: c.most_common(1)[0][0] for gid, c in votes.items()}


def main():
    rclpy.init()
    node = Eval()
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    t0 = time.time()
    while time.time() - t0 < dur and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)

    m = node.gid_to_emitter()
    print(f"\n===== PHASE-4 LIVE GATE (after {dur:.0f}s) =====")
    print(f"truth emitters seen: {sorted(node.truth)}   tracks published: {sorted(node.tracks)}")
    print(f"assoc bursts: blind={len(node.blind_assoc)} truth={len(node.truth_assoc)}")
    print(f"blind global_id -> truth emitter map: {m}")
    errs = []
    claimed = Counter(m.values())
    STAT = {0: 'INIT', 1: 'TRACK', 2: 'COAST', 3: 'LOST'}
    for gid in sorted(node.tracks):
        pos, status, gdop, nrx = node.tracks[gid]
        te = m.get(gid)
        if te is None or te not in node.truth:
            print(f"  {gid}: pos={np.round(pos,1)} status={STAT.get(status)} (no truth mapping)")
            continue
        err = float(np.linalg.norm(pos - node.truth[te]))
        errs.append(err)
        dup = ' !!DUP-EMITTER' if claimed[te] > 1 else ''
        print(f"  {gid}->emitter{te}: err={err:6.2f}m  status={STAT.get(status)} "
              f"gdop={gdop:.2f} nrx={nrx}{dup}")
    if errs:
        errs = np.array(errs)
        bij = (max(claimed.values()) == 1) and (len(errs) == len(node.truth))
        print(f"\n  position RMSE={np.sqrt(np.mean(errs**2)):.2f}m  mean={errs.mean():.2f}m  "
              f"max={errs.max():.2f}m  n_tracks={len(errs)}")
        print(f"  association bijective (no swaps, all emitters covered): {bij}")
        ok = np.sqrt(np.mean(errs**2)) < 30.0 and len(errs) >= len(node.truth) - 1
        print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    else:
        print("  GATE: FAIL (no tracks with truth mapping)")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
