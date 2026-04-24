#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
import tf2_ros
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
import cv2
import os

class TartanGroundNode(Node):

    def __init__(self):
        super().__init__('tartanground_node')

        # Ruta datos
        self.declare_parameter('dataset_path', '/home/israelnunez/tartanairpy/AbandonedCable/Data_omni/P0000')    
        self.dataset_path = self.get_parameter('dataset_path').value
        if self.dataset_path == '':
            raise RuntimeError("Debes pasar dataset_path")

        # Carpetas
        self.left_rgb_path = os.path.join(self.dataset_path, 'image_lcam_front')
        self.left_depth_path = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path = os.path.join(self.dataset_path, 'lidar')
        self.right_rgb_path = os.path.join(self.dataset_path, 'image_rcam_front')
        self.right_depth_path = os.path.join(self.dataset_path, 'depth_rcam_front')

        # Archivos
        self.left_rgb_files = sorted(os.listdir(self.left_rgb_path))
        self.left_depth_files = sorted(os.listdir(self.left_depth_path))
        self.lidar_files = sorted(os.listdir(self.lidar_path))

        # Cargar timestamps
        self.load_times()

        self.index = 0
        self.bridge = CvBridge()

        # Publishers
        self.left_rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.left_depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pose_gt_pub = self.create_publisher(PoseWithCovarianceStamped,'/pose/ground_truth' ,10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.loop)

    def load_times(self):
        # Cámara
        cam_time_file_npy = os.path.join(self.dataset_path,'imu', 'cam_time.npy')
        cam_time_file_txt = os.path.join(self.dataset_path,'imu','cam_time.txt')
        self.cam_times = np.load(cam_time_file_npy) if os.path.exists(cam_time_file_npy) else np.loadtxt(cam_time_file_txt)

        # IMU
        imu_time_file_npy = os.path.join(self.dataset_path,'imu','imu_time.npy')
        imu_time_file_txt = os.path.join(self.dataset_path,'imu', 'imu_time.txt')
        self.imu_times = np.load(imu_time_file_npy) if os.path.exists(imu_time_file_npy) else np.loadtxt(imu_time_file_txt)

    def to_ros_time(self, t):
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        return Time(sec=sec, nanosec=nanosec)

    # -----------------------------
    # RGB
    # -----------------------------
    def publish_rgb(self, file, t):
        img = cv2.imread(file)
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.to_ros_time(t)
        msg.header.frame_id = 'camera_link'
        self.left_rgb_pub.publish(msg)

    # -----------------------------
    # Depth
    # -----------------------------
    def publish_depth(self, file, t):
        depth = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        msg = self.bridge.cv2_to_imgmsg(depth, encoding='passthrough')
        msg.header.stamp = self.to_ros_time(t)
        msg.header.frame_id = 'camera_link'
        self.left_depth_pub.publish(msg)

    # -----------------------------
    # LiDAR → LaserScan
    # -----------------------------
    def publish_scan(self, file, t):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        xy_points = points[:, :2]
        ranges = np.linalg.norm(xy_points, axis=1)
        n = len(ranges)

        scan = LaserScan()
        scan.header.stamp = self.to_ros_time(t)
        scan.header.frame_id = 'laser'
        scan.angle_min = -1.57
        scan.angle_max = 1.57
        scan.angle_increment = 3.14 / n
        scan.range_min = 0.1
        scan.range_max = 50.0
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

    # -----------------------------
    # Loop principal con sincronización
    # -----------------------------
    def loop(self):
        if self.index >= len(self.left_rgb_files):
            self.get_logger().info("Dataset terminado")
            rclpy.shutdown()
            return

        # Usar timestamp de cámara actual
        cam_t = self.cam_times[self.index]

        # Encontrar índice más cercano en LiDAR
        lidar_idx = np.argmin(np.abs(self.imu_times - cam_t))
        lidar_file = os.path.join(self.lidar_path, self.lidar_files[lidar_idx])

        # Archivos de cámara
        rgb_file = os.path.join(self.left_rgb_path, self.left_rgb_files[self.index])
        depth_file = os.path.join(self.left_depth_path, self.left_depth_files[self.index])

        # Publicar sincronizado
        self.publish_rgb(rgb_file, cam_t)
        self.publish_depth(depth_file, cam_t)
        self.publish_scan(lidar_file, self.imu_times[lidar_idx])

        self.index += 1

def main():
    rclpy.init()
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()