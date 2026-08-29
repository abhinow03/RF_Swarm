"""capstone_predict — Phase 5: path prediction from /tracks.

Publishes a constant-velocity (CV) baseline PathPrediction per track on
/predictions. The CV baseline is the PERMANENT comparison point — the number any
ML path model must beat.

Finding (2026-08-26): there is NO ML translational path model to wrap. The only
peer trajectory model is the STGT graph-transformer, which is TRANSLATION-INVARIANT
by construction (centroid-relative node features), so it cannot predict where the
swarm is going — its "centroid_velocity" regression is a formation-internal-dynamics
quantity (~constant regardless of ground speed), not translational velocity. So the
CV baseline stands alone here; its RMSE is still the phase's recorded path-quality
number, fed from real EKF localization (subject to Phase-4's clock-prior-limited
position error, NOISE_MODEL §4a).

CV model: waypoint(t) = p + v·t using the EKF track's estimated position/velocity;
per-waypoint 1-sigma grows as sqrt(pos_var + vel_var·t^2) from the (inflated) track
covariance — an honest, widening cone rather than a false point estimate.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time as TimeMsg
from capstone_msgs.msg import PathPrediction, TrackArray

from capstone_common.health import HealthReporter, qos_reliable


class PredictNode(Node):
    def __init__(self):
        super().__init__('capstone_predict')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('horizon_sec', 5.0)
        self.declare_parameter('n_waypoints', 10)
        pt = self.get_parameter('passthrough_truth').value
        self._horizon = float(self.get_parameter('horizon_sec').value)
        self._nwp = int(self.get_parameter('n_waypoints').value)

        self._pub = self.create_publisher(PathPrediction, '/predictions', qos_reliable())
        self.health = HealthReporter(self, pt)
        self.create_subscription(TrackArray, '/tracks', self._on_tracks, qos_reliable())

    @staticmethod
    def _s_to_time(t_s):
        ns = int(round(t_s * 1e9))
        m = TimeMsg(); m.sec = ns // 1_000_000_000; m.nanosec = ns % 1_000_000_000
        return m

    def _on_tracks(self, msg):
        t0 = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        n = 0
        for tr in msg.tracks:
            p = np.array([tr.position.x, tr.position.y, tr.position.z])
            v = np.array([tr.velocity.x, tr.velocity.y, tr.velocity.z])
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
                continue
            cov = np.array(tr.covariance, float).reshape(6, 6) if len(tr.covariance) == 36 else np.zeros((6, 6))
            pos_var = float(np.trace(cov[:3, :3]) / 3.0)
            vel_var = float(np.trace(cov[3:6, 3:6]) / 3.0)

            out = PathPrediction()
            out.header = msg.header
            out.global_id = tr.global_id
            out.horizon_sec = self._horizon
            out.model_id = 'cv_baseline'
            for k in range(1, self._nwp + 1):
                dt = self._horizon * k / self._nwp
                wp = p + v * dt
                out.waypoints.append(Point(x=float(wp[0]), y=float(wp[1]), z=float(wp[2])))
                out.waypoint_times.append(self._s_to_time(t0 + dt))
                out.position_sigma.append(float(np.sqrt(max(pos_var + vel_var * dt * dt, 0.0))))
            self._pub.publish(out)
            n += 1
        self.health.set_status(f'cv_baseline predictions for {n} tracks, horizon={self._horizon}s')
        self.health.bump()


def main():
    rclpy.init()
    node = PredictNode()
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
