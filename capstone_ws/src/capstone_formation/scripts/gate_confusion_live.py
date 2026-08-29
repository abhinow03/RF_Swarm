"""Collect /formation predictions (from REAL EKF tracks) for one scenario run,
report the modal predicted class + distribution, append to a results file.

args: <true_formation> <duration_s> <results_file>
"""
import sys, time, json
from collections import Counter
import rclpy
from rclpy.node import Node
from capstone_msgs.msg import FormationEstimate
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def latched(depth=1):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class Collector(Node):
    def __init__(self):
        super().__init__('formation_collector')
        self.preds = []           # (class, conf, from_truth)
        self.create_subscription(FormationEstimate, '/formation', self._cb, latched())

    def _cb(self, m):
        self.preds.append((m.formation_class, round(m.confidence, 3), m.from_truth))


def main():
    true_f = sys.argv[1]; dur = float(sys.argv[2]); results = sys.argv[3]
    rclpy.init(); node = Collector()
    t0 = time.time()
    while time.time() - t0 < dur and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
    classes = [c for c, _, _ in node.preds]
    counts = Counter(classes)
    mode = counts.most_common(1)[0][0] if counts else "NONE"
    mean_conf = (sum(cf for _, cf, _ in node.preds) / len(node.preds)) if node.preds else 0.0
    rec = {"true": true_f, "pred_mode": mode, "n_pred": len(classes),
           "counts": dict(counts), "mean_conf": round(mean_conf, 3)}
    with open(results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[live] true={true_f} pred_mode={mode} n={len(classes)} counts={dict(counts)} conf~{mean_conf:.3f}")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
