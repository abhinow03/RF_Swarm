"""capstone_eval Phase 0 stub — the ONLY node allowed to read /truth/*.

Subscribes to truth + estimates and reports liveness on /health/capstone_eval.
Its passthrough_truth defaults True because reading truth is this node's job (it
scores RMSE/NEES against ground truth); every other node defaults False. Real
node computes metrics -> CSV; the stub only proves the truth-reading wiring.
"""
import rclpy
from rclpy.node import Node
from capstone_msgs.msg import TruthScene, TrackArray

from capstone_common.health import HealthReporter, qos_best_effort, qos_reliable


class EvalStub(Node):
    def __init__(self):
        super().__init__('capstone_eval')
        # eval is the designated truth reader -> True by design (not a leak).
        self.declare_parameter('passthrough_truth', True)
        pt = self.get_parameter('passthrough_truth').value

        self.health = HealthReporter(self, pt)
        self._truth_scene = None
        self._tracks = None

        self.create_subscription(TruthScene, '/truth/scene', self._on_truth, qos_best_effort())
        self.create_subscription(TrackArray, '/tracks', self._on_tracks, qos_reliable())

    def _on_truth(self, m):
        self._truth_scene = m
        self.health.bump()

    def _on_tracks(self, m):
        self._tracks = m


def main():
    rclpy.init()
    node = EvalStub()
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
