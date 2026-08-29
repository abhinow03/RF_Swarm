"""Shared helpers: QoS profiles + a 1 Hz NodeHealth reporter.

Single canonical copy for the whole workspace. Imported by every capstone_*
node as `from capstone_common.health import ...`. (Previously duplicated per
package during Phase 0; consolidated here so real nodes share one definition.)
"""
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                       QoSHistoryPolicy)
from capstone_msgs.msg import NodeHealth


def qos_best_effort(depth=10):
    return QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def qos_reliable(depth=10):
    return QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def qos_latched(depth=1):
    # RELIABLE + TRANSIENT_LOCAL: late-joining subscribers get the last message.
    return QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


class HealthReporter:
    """Publishes capstone_msgs/NodeHealth on /health/<node> at 1 Hz.

    Rate is measured from a caller-incremented tick counter (call bump() on each
    primary publish). status_text defaults to 'ok'; callers may override it.
    """

    def __init__(self, node, passthrough_truth, status_text='ok'):
        self.node = node
        self.passthrough_truth = bool(passthrough_truth)
        self._pub = node.create_publisher(
            NodeHealth, '/health/' + node.get_name(), qos_best_effort())
        self._count = 0
        self._last_count = 0
        self._status_text = status_text
        self.messages_dropped = 0
        self._timer = node.create_timer(1.0, self._on_timer)

    def bump(self, n=1):
        self._count += n

    def set_status(self, text):
        self._status_text = text

    def _on_timer(self):
        rate = float(self._count - self._last_count)  # msgs in the last ~1 s
        self._last_count = self._count
        m = NodeHealth()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.node_name = self.node.get_name()
        m.passthrough_truth = self.passthrough_truth
        m.publish_rate_hz = rate
        m.messages_dropped = int(self.messages_dropped)
        m.status_text = self._status_text
        self._pub.publish(m)
