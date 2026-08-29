"""Collect the /narration timeline (wall-clock stamped) to show the degenerate->
formation transition. args: <duration_s>"""
import sys, time
import rclpy
from rclpy.node import Node
from capstone_msgs.msg import Narration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def latched(d=10):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=d)


class C(Node):
    def __init__(self):
        super().__init__('narration_collector')
        self.rows = []
        self.t0 = time.time()
        self.create_subscription(Narration, '/narration', self._cb, latched())

    def _cb(self, m):
        self.rows.append((round(time.time() - self.t0, 1), m.template_id, m.source, m.text))


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    rclpy.init(); n = C()
    while time.time() - n.t0 < dur and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.2)
    print("\n===== /narration TIMELINE (t = seconds since collector start) =====")
    last = None
    for t, tid, src, txt in n.rows:
        if (tid, txt) == last:
            continue                       # collapse latched re-delivery
        last = (tid, txt)
        print(f"  t={t:6.1f}s  [{src:8s}] template={tid:28s} :: {txt[:90]}")
    tids = [r[1] for r in n.rows]
    print(f"\n  templates seen: {sorted(set(tids))}")
    print(f"  degenerate narrations: {sum(t=='ctx_degenerate_no_formation' for t in tids)}"
          f"   formation narrations: {sum(t=='ctx_formation_stgt' for t in tids)}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
