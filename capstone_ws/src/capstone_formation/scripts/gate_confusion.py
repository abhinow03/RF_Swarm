#!/usr/bin/env python3
"""Phase-5 formation-classifier gate — TRUTH-fed reference confusion matrix.

Classifies the CLEAN parametric-truth trajectory of each K=6 scenario (the exact
formation geometry + CV centroid path the sim produces), giving the reference the
real-EKF-fed confusion matrix (gate_confusion_live.py) is compared against. If the
real-EKF matrix is meaningfully worse than THIS, the signal is Phase-4's
clock-prior-limited RMSE (NOISE_MODEL §4a), not a classifier bug.

Run: python3 src/capstone_formation/scripts/gate_confusion.py
"""
import sys
import numpy as np

sys.path.insert(0, "/home/pw26_akp_01/RF_Swarm/capstone_ws/src/capstone_formation")
from capstone_formation.classifier import FormationClassifier

from swarm_intent.formations import get_formation_offsets

MODEL_DIR = "/home/pw26_akp_01/RF_Swarm/models/stgt"
# the K=6 confusion-matrix scenarios (formation_label -> matches gate_k6* YAMLs)
SCENARIOS = ["v_shape", "encirclement", "column", "diamond", "dispersed"]
# gate_k6 trajectory (shared by all variants)
START = np.array([-30.0, 0.0, 50.0]); VEL = np.array([1.0, 0.0, 0.0])
SEED = 7; DT = 0.5; T = 50


def truth_window(formation):
    offs = get_formation_offsets(formation, 1.0, np.random.default_rng(SEED))  # (6,3)
    return np.stack([(START + VEL * (t * DT))[None, :] + offs for t in range(T)], axis=0)


def main():
    clf = FormationClassifier(MODEL_DIR)
    names = clf.FORMATION_NAMES
    print("=== Phase-5 TRUTH-fed confusion (clean scenario trajectories) ===")
    rows = {}
    correct = 0
    for f in SCENARIOS:
        out = clf.classify(truth_window(f))
        pred = out["formation_type"]; conf = out["formation_confidence"]
        rows[f] = (pred, conf, clf.prob_vector(out))
        ok = "OK" if pred == f else "MISS"
        correct += (pred == f)
        print(f"  true={f:13s} -> pred={pred:13s} conf={conf:.3f}  [{ok}]")
    print(f"\n  truth-fed accuracy: {correct}/{len(SCENARIOS)}")
    # compact confusion (true -> top-1 pred)
    print("  (top-1 predictions define the reference; live gate compares against this)")
    return rows


if __name__ == "__main__":
    main()
