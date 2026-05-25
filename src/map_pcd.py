#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import open3d as o3d
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import sensor_msgs_py.point_cloud2 as pc2

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSDurabilityPolicy


class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('pcd_path', '/home/israel/tartanairpy/Office/Office_rgb.pcd')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('topic', '/pointcloud')

        pcd_path = self.get_parameter('pcd_path').value
        self.frame_id = self.get_parameter('frame_id').value
        topic = self.get_parameter('topic').value

        R = np.array([
            [0,  1, 0],
            [-1, 0, 0],
            [0,  0, 1]
        ])


        # ---------------- LOAD PCD ----------------
        self.get_logger().info(f"Cargando PCD: {pcd_path}")

        self.pcd = o3d.io.read_point_cloud(pcd_path)

        if len(self.pcd.points) == 0:
            raise RuntimeError("PCD vacío")

        self.points = np.asarray(self.pcd.points)

        self.points = (R @ self.points.T).T

        self.points[:, 2] *= -1

        self.get_logger().info(f"Puntos cargados: {len(self.points)}")

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---------------- PUBLISHER ----------------
        self.pub = self.create_publisher(
            PointCloud2,
            topic,
            latched_qos
        )

        # Publicar una vez (latched style)
        # self.timer = self.create_timer(1.0, self.publish_cloud)

        self.publish_cloud()

        # self.published = False

    # ------------------------------------------------

    def publish_cloud(self):

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        # Convertir Nx3 a PointCloud2
        cloud_msg = pc2.create_cloud_xyz32(
            header,
            self.points.tolist()
        )

        self.pub.publish(cloud_msg)

        self.get_logger().info("PointCloud publicada")

# =====================================================

def main():
    rclpy.init()
    node = PCDPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()