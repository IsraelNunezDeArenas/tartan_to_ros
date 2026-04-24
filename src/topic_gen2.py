#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
import tf2_ros
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
import cv2
import os
import time
import threading
from rosgraph_msgs.msg import Clock

from collections import deque
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class TartanGroundNode(Node):

    def __init__(self):
        super().__init__('tartanground_node')

        # ── Parámetros ────────────────────────────────────────────
        self.declare_parameter('dataset_path',
            '/home/israelnunez/tartanairpy/House/Data_omni/P0000')

        self.declare_parameter('topic_rgb_image', '/camera/rgb')
        self.declare_parameter('topic_depth_image', '/camera/depth')
        self.declare_parameter('topic_localization', '/amcl_pose')
        self.declare_parameter('topic_camera_info', '/camera/camera_info')
        self.declare_parameter('camera_localization', '/debug/default_camera')
        self.declare_parameter('motion_data', '/imu/motion_data')

        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('robot_frame_id', 'base_link')
        self.declare_parameter('camera_frame_id', 'camera')
        self.declare_parameter('lidar_frame_id', 'lidar')

        self.declare_parameter('publish_rate', 10.0)

        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 640)
        self.declare_parameter('fx', 320.0)
        self.declare_parameter('fy', 320.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 320.0)

        # ── Leer parámetros ─────────────────────────────────────
        self.dataset_path = self.get_parameter('dataset_path').value
        self.topic_rgb = self.get_parameter('topic_rgb_image').value
        self.topic_depth = self.get_parameter('topic_depth_image').value
        self.cam_topic_pose = self.get_parameter('camera_localization').value
        self.topic_info = self.get_parameter('topic_camera_info').value
        self.map_frame = self.get_parameter('map_frame_id').value
        self.robot_frame = self.get_parameter('robot_frame_id').value
        self.cam_frame = self.get_parameter('camera_frame_id').value
        self.lidar_frame = self.get_parameter('lidar_frame_id').value
        self.topic_robot_pose = self.get_parameter('topic_localization').value
        self.motion_data = self.get_parameter('motion_data').value
        self.rate = self.get_parameter('publish_rate').value

        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        # ── Dataset ─────────────────────────────────────────────
        self.rgb_path = os.path.join(self.dataset_path, 'image_lcam_front')
        self.depth_path = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path = os.path.join(self.dataset_path, 'lidar')

        self.rgb_files = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))
        self.lidar_files = sorted(os.listdir(self.lidar_path))

        # ── Cargar datos ────────────────────────────────────────
        self._load_times()
        self._load_camera_poses()
        self._load_robot_pose()
        self._load_velocity()

        self.bridge = CvBridge()

        # ── Publishers ─────────────────────────────────────────
        self.rgb_pub = self.create_publisher(Image, self.topic_rgb, 10)
        self.depth_pub = self.create_publisher(Image, self.topic_depth, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.cam_topic_pose, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.robot_pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.topic_robot_pose, 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.initial_pose_pub=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.info_pub = self.create_publisher(CameraInfo, self.topic_info, qos_latched)

        # QoS compatible con transient_local

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_latched)

        
        # ── Odometría ─────────────────────────────────────────
        self.x_odom = 0.0          # posición X integrada
        self.y_odom = 0.0          # posición Y integrada
        self.yaw_odom = 0.0        # orientación yaw integrada
        self.last_time = None      # timestamp de la última actualización

        # Quaternion de la orientación para TF / Odometry
        self.q_odom = R.from_euler('z', self.yaw_odom).as_quat()

        # Velocidades integradas para Odometry (opcional)
        self.vx_odom = 0.0
        self.vy_odom = 0.0
        self.wz_odom = 0.0

        # TF dinámico
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_pub = StaticTransformBroadcaster(self)

        self.mapa_recibido = False

        self.get_logger().info("Mapa no recibido,esperando posición")

        while not self.mapa_recibido and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info("Mapa recibido, arrancando threads de publicación...")

        self.get_logger().info("Sleep")
        time.sleep(10)

        self._publish_camera_info()

        self._publish_initial_state()
        

        time.sleep(100)

        # ── Threads ────────────────────────────────────────────
        self.cam_index = 1
        self.imu_index = 0

        threading.Thread(target=self._camera_loop).start()
        threading.Thread(target=self._imu_loop).start()

    # =========================================================================
    # Carga de datos
    # =========================================================================

    def _load(self, npy, txt):
        if npy and os.path.exists(npy):
            return np.load(npy)
        if txt and os.path.exists(txt):
            return np.loadtxt(txt)
        raise FileNotFoundError

    def _load_times(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.cam_times = self._load(os.path.join(base, 'cam_time.npy'),
                                   os.path.join(base, 'cam_time.txt'))
        self.imu_times = self._load(os.path.join(base, 'imu_time.npy'),
                                   os.path.join(base, 'imu_time.txt'))

    def _load_camera_poses(self):
        if os.path.isdir(os.path.join(self.dataset_path, 'metadata')):
            path = os.path.join(self.dataset_path, 'metadata', 'pose_lcam_front.txt')
        else:
            path = os.path.join(self.dataset_path, 'pose_lcam_front.txt')

        self.cam_poses = np.loadtxt(path)

    def _load_robot_pose(self):
        base = os.path.join(self.dataset_path, 'imu')

        pos = self._load(os.path.join(base, "pos_global.npy"),
                         os.path.join(base, "pos_global.txt"))
        ori = self._load(os.path.join(base, "ori_global.npy"),
                         os.path.join(base, "ori_global.txt"))

        roll, pitch, yaw = ori[:,0], ori[:,1], ori[:,2]

        cr = np.cos(roll/2); sr = np.sin(roll/2)
        cp = np.cos(pitch/2); sp = np.sin(pitch/2)
        cy = np.cos(yaw/2); sy = np.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        self.robot_poses = np.hstack((pos, np.stack((qx,qy,qz,qw), axis=1)))

    def _load_velocity(self):
        base = os.path.join(self.dataset_path, 'imu')

        vel_body = self._load(
            os.path.join(base, "vel_body.npy"),
            os.path.join(base, "vel_body.txt")
        )

        gyro = self._load(
            os.path.join(base, "gyro.npy"),
            os.path.join(base, "gyro.txt")
            )

        wx, wy, wz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

        # Construimos un array unificado:
        # [vx, vy, vz, wx, wy, wz]
        self.robot_velocities_body = np.hstack((
            vel_body,
            np.stack((wx, wy, wz), axis=1)
        ))


    def _pose_to_matrix(self, p):
        T = np.eye(4)
        T[:3,:3] = R.from_quat(p[3:]).as_matrix()
        T[:3,3] = p[:3]
        return T

    def _to_ros_time(self, t):
        return Time(sec=int(t), nanosec=int((t%1)*1e9))

    # =========================================================================

    def _publish_camera_info(self):
        msg = CameraInfo()
        msg.width = self.img_w
        msg.height = self.img_h
        msg.k = list(map(float, [
            self.fx, 0, self.cx,
            0, self.fy, self.cy,
            0, 0, 1
            ]))
        msg.header.frame_id = self.cam_frame
        msg.header.stamp = self._to_ros_time(0.0)
        self.info_pub.publish(msg)

    def _publish_rgb(self, path, t):
        img = cv2.imread(path)
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.rgb_pub.publish(msg)

    def _publish_depth(self, path, t):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if raw is None:
            self.get_logger().warn(f"No se pudo leer depth: {path}")
            return

        # Caso 1: ya es depth válido (1 canal)
        if len(raw.shape) == 2:
            depth = raw.astype(np.float32)

        # Caso 2: múltiples canales (TU CASO: 32FC4)
        elif len(raw.shape) == 3:
            depth = raw[:, :, 0].astype(np.float32)

        else:
            self.get_logger().error("Formato de depth desconocido")
            return

        # Si viene en mm → convertir a metros
        if raw.dtype == np.uint16:
            depth = depth / 1000.0

        # 🔥 Asegurar formato correcto
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        msg.header.stamp = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame

        self.depth_pub.publish(msg)

    def _publish_cam_pose(self, idx, t):
        p = self.cam_poses[idx]
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self._to_ros_time(t)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = p[0]
        msg.pose.pose.position.y = p[1]
        msg.pose.pose.position.z = p[2]
        msg.pose.pose.orientation.x = p[3]
        msg.pose.pose.orientation.y = p[4]
        msg.pose.pose.orientation.z = p[5]
        msg.pose.pose.orientation.w = p[6]
        self.pose_pub.publish(msg)

    def _publish_robot_pose(self, idx, t):
        p = self.robot_poses[idx]
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self._to_ros_time(t)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = p[0]
        msg.pose.pose.position.y = p[1]
        msg.pose.pose.position.z = p[2]
        msg.pose.pose.orientation.x = p[3]
        msg.pose.pose.orientation.y = p[4]
        msg.pose.pose.orientation.z = p[5]
        msg.pose.pose.orientation.w = p[6]
        self.robot_pose_pub.publish(msg)

    def find_closest_imu_time(self, cam_t):
        idx = np.argmin(np.abs(self.imu_times - cam_t))
        return self.imu_times[idx]

    def _publish_lidar(self, i, t):
        file_path = os.path.join(self.lidar_path, self.lidar_files[i])

        # --- Cargar nube PLY ---
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            self.get_logger().warn(f"Scan vacío en índice {i}")
            return

        # --- Parámetros del LaserScan ---
        angle_min = -np.pi
        angle_max = np.pi
        angle_increment = np.deg2rad(0.5)
        num_beams = int((angle_max - angle_min) / angle_increment)
        ranges = np.full(num_beams, np.inf)

        # --- Filtrado y proyección XY ---
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        mask = (zs > -0.2) & (zs < 1.5)
        xs, ys = xs[mask], ys[mask]

        # --- Conversión a polares ---
        angles = np.arctan2(ys, xs)
        distances = np.sqrt(xs**2 + ys**2)
        indices = ((angles - angle_min) / angle_increment).astype(int)

        for idx, dist in zip(indices, distances):
            if 0 <= idx < num_beams:
                ranges[idx] = min(ranges[idx], dist)

        ranges[np.isinf(ranges)] = 50.0

        # --- Crear mensaje LaserScan ---
        scan = LaserScan()
        scan.header.stamp = self._to_ros_time(t)
        scan.header.frame_id = self.lidar_frame
        scan.angle_min = angle_min
        scan.angle_max = angle_max
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.rate
        scan.range_min = 0.1
        scan.range_max = 50.0
        scan.ranges = ranges.tolist()

        self.scan_pub.publish(scan)
        self.get_logger().debug(f"LiDAR frame {i} publicado como LaserScan")

    def _compute_odometry(self, idx, t):

        if self.last_time is None:
            self.last_time = t
            return  # primera iteración: sin dt válido, salir
        dt = t - self.last_time
        self.last_time = t

        # 🔹 Velocidades en body frame
        v = self.robot_velocities_body[idx]
        vx = float(v[0])
        vy = float(v[1])

        wz = float(v[5])

        # 🔹 Precalcular
        cos_yaw = np.cos(self.yaw_odom)
        sin_yaw = np.sin(self.yaw_odom)

        # 🔹 Integración correcta (holonómica)
        self.x_odom += (vx * cos_yaw - vy * sin_yaw) * dt
        self.y_odom += (vx * sin_yaw + vy * cos_yaw) * dt
        self.yaw_odom += wz * dt

        # 🔄 Normalizar yaw
        self.yaw_odom = np.arctan2(np.sin(self.yaw_odom), np.cos(self.yaw_odom))

        # 🔄 Cuaternión
        q = R.from_euler('z', self.yaw_odom).as_quat()

        self._publish_odometry(t, vx, vy, wz, q)

    def _publish_odometry(self, t, vx, vy, wz, q):

        odom = Odometry()

        stamp = self._to_ros_time(t)

        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # 📍 Pose
        odom.pose.pose.position.x = self.x_odom
        odom.pose.pose.position.y = self.y_odom
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # 🚀 Velocidades
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        # 📊 Covarianza
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.1

        odom.twist.covariance[0] = 0.1
        odom.twist.covariance[7] = 0.1
        odom.twist.covariance[35] = 0.2

        self.odom_pub.publish(odom)

        # TF (OBLIGATORIO)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"

        tf.transform.translation.x = self.x_odom
        tf.transform.translation.y = self.y_odom
        tf.transform.translation.z = 0.0

        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(tf)
    def matrix_to_transform(self, T, parent_frame, child_frame, t):
        """
        Convierte una matriz 4x4 en un TransformStamped

        Args:
            T (np.ndarray): matriz homogénea 4x4
            parent_frame (str): frame padre (ej: "body_link")
            child_frame (str): frame hijo (ej: "lidar")
            node: nodo ROS2 (para timestamp)

        Returns:
            TransformStamped
        """

        msg = TransformStamped()

        msg.header.stamp = self._to_ros_time(t)

        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame

        # 🔹 Traslación
        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])

        # 🔹 Rotación (matriz → cuaternión)
        rot = R.from_matrix(T[0:3, 0:3])
        q = rot.as_quat()  # (x, y, z, w)

        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])

        return msg



    def _publish_initial_state(self):
        """Publica la primera pose ground truth como initialpose para AMCL"""
        p = self.robot_poses[0]  # Primera pose: [x, y, z, qx, qy, qz, qw]

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self._to_ros_time( self.imu_times[0] )
        msg.header.frame_id = self.map_frame  # normalmente 'map'

        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])

        msg.pose.pose.orientation.x = float(p[3])
        msg.pose.pose.orientation.y = float(p[4])
        msg.pose.pose.orientation.z = float(p[5])
        msg.pose.pose.orientation.w = float(p[6])

        # Covarianza diagonal baja (confianza alta en pose inicial)
        cov = [0.0]*36
        cov[0] = 0.001  # x
        cov[7] = 0.001  # y
        cov[35] = 0.001  # yaw
        msg.pose.covariance = cov

        self.initial_pose_pub.publish(msg)

        p_cam = self.cam_poses[0]

        T_cam = self._pose_to_matrix(p_cam)
        T_robot = self._pose_to_matrix(p)

        T_base_cam = np.linalg.inv(T_robot) @ T_cam

        tf_base_cam = self.matrix_to_transform(T_base_cam,
                                                self.robot_frame,
                                                self.cam_frame,
                                                self.imu_times[0])

        tf_cam_lidar = self.matrix_to_transform(np.eye(4),
                                                self.cam_frame,
                                                self.lidar_frame,
                                                self.imu_times[0])                                        

        self.tf_static_pub.sendTransform([tf_base_cam,tf_cam_lidar])      


        self.get_logger().info(
            f"Pose inicial ground truth publicada"
        )

    def map_callback(self, msg):
        if not self.mapa_recibido:
            self.get_logger().info("Mapa recibido, activando publicación de sensores")
            self.mapa_recibido = True

    def _publish_clock(self, t):
        msg = Clock()
        msg.clock = self._to_ros_time(t)
        self.clock_pub.publish(msg)

    # =========================================================================

    def _camera_loop(self):
        while rclpy.ok() and self.cam_index < len(self.rgb_files):
            i = self.cam_index
            self.cam_index += 1

            t = self.find_closest_imu_time(self.cam_times[i]) #Sincronizacion

            self._publish_rgb(os.path.join(self.rgb_path, self.rgb_files[i]), t)
            self._publish_depth(os.path.join(self.depth_path, self.depth_files[i]), t)
            self._publish_cam_pose(i, t)
            self._publish_lidar(i, t)

            time.sleep(1.0/self.rate)

        self.get_logger().info(f"Fin del bucle de cámara")

    def _imu_loop(self):
        
        while rclpy.ok() and self.imu_index < len(self.imu_times):

            i = self.imu_index
            self.imu_index += 1

            t = self.imu_times[i]

            self._publish_clock(t)

            self._publish_robot_pose(i, t)
            self._compute_odometry(i,t)

            time.sleep(1.0/(10*self.rate))

        self.get_logger().info(f"Fin del bucle de imu")


def main():
    rclpy.init()
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()