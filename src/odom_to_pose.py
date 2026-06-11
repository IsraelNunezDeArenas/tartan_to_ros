#!/usr/bin/env python3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
import rclpy

class PoseReader(Node):
    def __init__(self):
        super().__init__('pose_reader')
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.cb,
            10
        )
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/pose/filtered',
            10
        )

    def cb(self, msg: Odometry):
        out = PoseWithCovarianceStamped()

        # Mismo timestamp y frame que el EKF
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        # Pose
        out.pose.pose = msg.pose.pose

        # Covarianza completa (6x6 → 36 elementos)
        out.pose.covariance = msg.pose.covariance

        self.pub.publish(out)


def main():
    rclpy.init()
    node = PoseReader()
    # El mapa ya está publicado (latched); solo necesitamos mantener el nodo vivo
    # para que los suscriptores tardíos (AMCL, rviz) reciban el QoS transient local.
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()