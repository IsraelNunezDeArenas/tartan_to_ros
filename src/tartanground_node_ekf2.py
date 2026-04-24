#!/usr/bin/env python3
"""
TartanGroundNode — Opción A: AMCL puro
=======================================

Pipeline honesto dado que todo el contenido de imu/ es GT derivado:

  robot_poses[0]  ──► /initialpose      (único uso de GT, solo frame 0)
  vel_body + gyro ──► /odom             (dead-reckoning, sabemos que es GT derivado)
  LiDAR PLY       ──► /scan             (único sensor independiente)

Lo que se elimina respecto a versiones anteriores:
  - /imu/data          (sería publicar GT derivado como si fuera sensor físico)
  - /odom/vel_body     (redundante sin EKF)
  - EKF                (no hay sensores independientes que fusionar)
  - _publish_robot_pose / _publish_cam_pose  (GT en frames > 0)
  - vel_global         (GT directo, no se usa)
  - accel_no_gravity   (GT derivado, no se usa)

Convenciones del dataset:
  MARCO NED MUNDO  (robot_poses):  x=North, y=East,  z=Down
  MARCO NED BODY   (vel_body, gyro): x=fwd, y=right, z=down
  MARCO ÓPTICO     (LiDAR PLY):    x=right, y=down,  z=fwd
  MARCO ENU ROS    (map, odom):    x=East,  y=North,  z=Up
  MARCO ROS BODY   (base_link):    x=fwd,   y=left,   z=up
"""

import os
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                       HistoryPolicy, DurabilityPolicy)
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, LaserScan, CameraInfo, Imu
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from rosgraph_msgs.msg import Clock
import tf2_ros
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cv_bridge import CvBridge

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# NED mundo → ENU mundo: (E, N, U) = (y_NED, x_NED, -z_NED)
R_NED2ENU = np.array([
    [0.,  1.,  0.],
    [1.,  0.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)

# NED body (x=fwd, y=right, z=down) → ROS body (x=fwd, y=left, z=up)
R_BODY_NED2ROS = np.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)

# Umbral máximo de dt para descartar integraciones espurias
DT_MAX = 0.5

# LaserScan
RANGE_MIN       = 0.3
RANGE_MAX       = 50.0
ANGLE_MIN       = -np.pi
ANGLE_MAX       =  np.pi
ANGLE_INCREMENT = np.deg2rad(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de conversión
# ─────────────────────────────────────────────────────────────────────────────

def ned_pose_to_enu(p: np.ndarray) -> np.ndarray:
    """
    [x,y,z, qx,qy,qz,qw] NED mundo + NED body → ENU mundo + ROS body.
    Resultado: base_link expresado en ENU (REP-103).
    """
    pos_enu    = R_NED2ENU @ p[:3]
    R_body_ned = R.from_quat(p[3:]).as_matrix()
    R_body_enu = R_NED2ENU @ R_body_ned @ R_BODY_NED2ROS
    q_enu      = R.from_matrix(R_body_enu).as_quat()
    return np.array([*pos_enu, *q_enu], dtype=np.float64)


def extract_yaw_enu(p_enu: np.ndarray) -> float:
    """
    Yaw ENU (CCW desde East) proyectando el vector forward en XY-ENU.
    Correcto aunque haya roll/pitch.
    """
    fwd = R.from_quat(p_enu[3:]).as_matrix()[:, 0]
    return float(np.arctan2(fwd[1], fwd[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Nodo
# ─────────────────────────────────────────────────────────────────────────────

class TartanGroundNode(Node):

    def __init__(self):
        super().__init__('tartanground_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('dataset_path',
            '/home/user/tartanground/House/Data_omni/P0000')
        self.declare_parameter('map_frame_id',   'map')
        self.declare_parameter('robot_frame_id', 'base_link')
        self.declare_parameter('lidar_frame_id', 'laser')
        self.declare_parameter('camera_frame_id', 'camera')
        self.declare_parameter('publish_rate',   10.0)
        self.declare_parameter('img_width',  640)
        self.declare_parameter('img_height', 640)
        self.declare_parameter('fx', 320.0)
        self.declare_parameter('fy', 320.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 320.0)

        self.dataset_path = self.get_parameter('dataset_path').value
        self.map_frame    = self.get_parameter('map_frame_id').value
        self.robot_frame  = self.get_parameter('robot_frame_id').value
        self.lidar_frame  = self.get_parameter('lidar_frame_id').value
        self.rate         = self.get_parameter('publish_rate').value
        self.cam_frame      = self.get_parameter('camera_frame_id').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        # ── Rutas ─────────────────────────────────────────────────────────────
        self.rgb_path    = os.path.join(self.dataset_path, 'image_lcam_front')
        self.depth_path  = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path  = os.path.join(self.dataset_path, 'lidar')
        self.rgb_files   = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))
        self.lidar_files = sorted(os.listdir(self.lidar_path))

        # - Carga de Datos ────────────────────────────────────────────────────────────

        self._load_times()
        self._load_robot_poses()
        self._load_motion_data()
        self._load_camera_poses()

        self.bridge = CvBridge()


        # ── Publishers ────────────────────────────────────────────────────────
        self.clock_pub        = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub         = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub         = self.create_publisher(LaserScan, '/scan', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.rgb_pub          = self.create_publisher(Image, '/camera/rgb', 10)
        self.depth_pub        = self.create_publisher(Image, '/camera/depth', 10)
        self.imu_pub          = self.create_publisher(Imu,'/imu/data',10)

        qos_latched = QoSProfile(
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.info_pub = self.create_publisher(
            CameraInfo, '/camera/camera_info', qos_latched)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.map_sub  = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, qos_latched)

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_pub  = StaticTransformBroadcaster(self)


        self._publish_camera_info() 
   
        # ── Control de flujo ──────────────────────────────────────────────────
        self.cam_index       = 0
        self.imu_index       = 0
        self._last_lidar_idx = -1
        self._imu_lock       = threading.Lock()
        self.mapa_recibido   = False
        self._map_timer      = None

        self.get_logger().info("TartanGroundNode (Opción A) listo. Esperando /map...")

    # =========================================================================
    # Carga de datos
    # =========================================================================

    def _load(self, npy: str, txt: str) -> np.ndarray:
        for p in (npy, txt):
            if p and os.path.exists(p):
                return np.load(p) if p.endswith('.npy') else np.loadtxt(p)
        raise FileNotFoundError(f"No se encontró: {npy} ni {txt}")

    def _load_times(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.cam_times = self._load(
            f'{base}/cam_time.npy', f'{base}/cam_time.txt')
        self.imu_times = self._load(
            f'{base}/imu_time.npy', f'{base}/imu_time.txt')

    def _load_robot_poses(self):
        """
        Carga robot_poses en NED. Solo se usa robot_poses[0] para /initialpose.
        """
        base = os.path.join(self.dataset_path, 'imu')
        pos  = self._load(f'{base}/pos_global.npy', f'{base}/pos_global.txt')
        ori  = self._load(f'{base}/ori_global.npy', f'{base}/ori_global.txt')

        roll, pitch, yaw = ori[:, 0], ori[:, 1], ori[:, 2]
        cr, sr = np.cos(roll/2),  np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
        qw =  cr*cp*cy + sr*sp*sy
        qx =  sr*cp*cy - cr*sp*sy
        qy =  cr*sp*cy + sr*cp*sy
        qz =  cr*cp*sy - sr*sp*cy
        self.robot_poses = np.hstack(
            (pos, np.stack((qx, qy, qz, qw), axis=1)))


    def _load_camera_poses(self):
        meta  = os.path.join(self.dataset_path, 'metadata', 'pose_lcam_front.txt')
        plain = os.path.join(self.dataset_path, 'pose_lcam_front.txt')
        self.cam_poses = np.loadtxt(meta if os.path.exists(meta) else plain)


    def _load_motion_data(self):
        """
        Carga vel_body y gyro.
        Sabemos que son GT derivados, pero los usamos como odometría
        de dead-reckoning dado que no hay encoders reales en el dataset.
        """
        base = os.path.join(self.dataset_path, 'imu')
        self.vel_body = self._load(
            f'{base}/vel_body.npy', f'{base}/vel_body.txt')
        self.gyro     = self._load(
            f'{base}/gyro.npy',     f'{base}/gyro.txt')

        self.acc    = self._load(
            f'{base}/acc.npy', f'{base}/acc.txt')
        self.get_logger().info(
            f"Cargados {len(self.vel_body)} pasos de vel_body + gyro "
            f"(GT derivado, usado como odometría).")

    # =========================================================================
    # Utilidades
    # =========================================================================

    def _to_ros_time(self, t: float) -> Time:
        return Time(sec=int(t), nanosec=int((t % 1) * 1e9))


    def matrix_to_transform(self, T, parent, child, t) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = parent
        msg.child_frame_id  = child
        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])
        q = R.from_matrix(T[:3, :3]).as_quat()
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        return msg

    def _pose_to_matrix(self, p: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R.from_quat(p[3:]).as_matrix()
        T[:3, 3]  = p[:3]
        return T
    # =========================================================================
    # Publicadores
    # =========================================================================

    def _publish_clock(self, t: float):
        msg = Clock()
        msg.clock = self._to_ros_time(t)
        self.clock_pub.publish(msg)

    def _publish_camera_info(self):
        msg = CameraInfo()
        msg.width  = self.img_w
        msg.height = self.img_h
        msg.k = [float(v) for v in [
            self.fx, 0, self.cx,
            0, self.fy, self.cy,
            0, 0, 1]]
        msg.header.stamp    = self._to_ros_time(0.0)
        msg.header.frame_id = 'camera'
        self.info_pub.publish(msg)

    def _publish_rgb(self, path: str, t: float):
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"RGB no leído: {path}")
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.rgb_pub.publish(msg)

    def _publish_depth(self, path: str, t: float):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            self.get_logger().warn(f"Depth no leído: {path}")
            return
        depth = raw.view(np.float32).reshape(raw.shape[:2])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        msg   = self.bridge.cv2_to_imgmsg(
            np.ascontiguousarray(depth, np.float32), encoding='32FC1')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.depth_pub.publish(msg)

    def _publish_lidar(self, cam_idx: int, t: float):
        """
        PLY (frame óptico: x=right, y=down, z=fwd) → LaserScan (base_link ENU).
        """
        # [F5] Guard duplicados
        if cam_idx == self._last_lidar_idx:
            return
        self._last_lidar_idx = cam_idx

        # [F5] Guard índice fuera de rango
        if cam_idx >= len(self.lidar_files):
            self.get_logger().warn(
                f"cam_idx {cam_idx} fuera de rango lidar ({len(self.lidar_files)})")
            return

        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[cam_idx]))
        pts = np.asarray(pcd.points)

        if pts.shape[0] == 0:
            self.get_logger().warn(f"PLY vacío en índice {cam_idx}")
            return

        # Frame óptico: x=right, y=down, z=fwd
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

        # Filtro de altura en frame óptico (y = abajo)
        mask = (ys > -0.3) & (ys < 0.3)
        xs, zs = xs[mask], zs[mask]

        if xs.size == 0:
            self.get_logger().warn(f"Slice vacío en índice {cam_idx}")
            return

        # Frame óptico → ROS body: x_ros = z_opt, y_ros = -x_opt
        x_ros = zs
        y_ros = -xs

        # [F4] Vectorizado
        num_beams = int(round((ANGLE_MAX - ANGLE_MIN) / ANGLE_INCREMENT))
        angles    = np.arctan2(y_ros, x_ros)
        distances = np.hypot(x_ros, y_ros)
        beam_idx  = ((angles - ANGLE_MIN) / ANGLE_INCREMENT).astype(np.int32)

        valid = (
            (beam_idx >= 0) & (beam_idx < num_beams) &
            (distances >= RANGE_MIN) & (distances <= RANGE_MAX)
        )
        beam_idx  = beam_idx[valid]
        distances = distances[valid]

        ranges = np.full(num_beams, np.inf, dtype=np.float32)
        np.minimum.at(ranges, beam_idx, distances)
        ranges[np.isinf(ranges)] = RANGE_MAX

        # [F3] Timestamp sincronizado con el último TF odom→base_link
        # angle_max ajustado a REP-117
        angle_max_msg = ANGLE_MIN + (num_beams - 1) * ANGLE_INCREMENT

        scan = LaserScan()
        scan.header.stamp    = self._to_ros_time(t)
        scan.header.frame_id = self.lidar_frame
        scan.angle_min       = float(ANGLE_MIN)
        scan.angle_max       = float(angle_max_msg)
        scan.angle_increment = float(ANGLE_INCREMENT)
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / self.rate
        scan.range_min       = RANGE_MIN
        scan.range_max       = RANGE_MAX
        scan.ranges          = ranges.tolist()
        scan.intensities     = []
        self.scan_pub.publish(scan)

    def _publish_imu(self, imu_idx: int, t: float):


        acc     = self.acc[imu_idx] # Datos Tartanground - NED
        w      = self.gyro[imu_idx] # Datos Tartanground - NED


        # ---------------- NED to ENU----------------
        accx_enu = float(acc[1])   # East
        accy_enu = float(acc[0])   # North
        accz_enu = -float(acc[2])  # Up

        w_enu = [0.0]*3

        w_enu[0] = w[1]   # East
        w_enu[1] = w[0]   # North
        w_enu[2] = -w[2]  # Up (invertido)

        #------------------------------

        imu_msg = Imu()

        imu_msg.header.stamp = self._to_ros_time(t)
        imu_msg.header.frame_id = "imu_link"

        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = 1.0
        imu_msg.orientation_covariance = [0.0] * 9
        imu_msg.orientation_covariance[0] = -1.0

        imu_msg.angular_velocity.x = w_enu[0]
        imu_msg.angular_velocity.y = w_enu[1]
        imu_msg.angular_velocity.z = w_enu[2]

        imu_msg.angular_velocity_covariance = [
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01
        ]

        imu_msg.linear_acceleration.x = accx_enu
        imu_msg.linear_acceleration.y = accy_enu
        imu_msg.linear_acceleration.z = accz_enu

        imu_msg.linear_acceleration_covariance = [
        0.2, 0.0, 0.0,
        0.0, 0.2, 0.0,
        0.0, 0.0, 0.2
        ]
        
        self.imu_pub.publish(imu_msg)


    # def _publish_odometry(self, t: float,
    #                       vx: float, vy: float, vz: float,
    #                       wz: float, q: np.ndarray):
    #     stamp = self._to_ros_time(t)

    #     odom = Odometry()
    #     odom.header.stamp    = stamp
    #     odom.header.frame_id = 'odom'
    #     odom.child_frame_id  = 'base_link'
    #     odom.pose.pose.position.x    = self.x_odom
    #     odom.pose.pose.position.y    = self.y_odom
    #     odom.pose.pose.position.z    = self.z_odom
    #     odom.pose.pose.orientation.x = float(q[0])
    #     odom.pose.pose.orientation.y = float(q[1])
    #     odom.pose.pose.orientation.z = float(q[2])
    #     odom.pose.pose.orientation.w = float(q[3])
    #     odom.twist.twist.linear.x    = vx
    #     odom.twist.twist.linear.y    = vy
    #     odom.twist.twist.linear.z    = vz
    #     odom.twist.twist.angular.z   = wz

    #     # Covarianza moderada: AMCL puede corregir sin saltos
    #     # (aunque la odometría sea GT derivado, declaramos incertidumbre
    #     #  para que AMCL tenga margen de corrección)
    #     odom.pose.covariance[0]  = 0.10   # x
    #     odom.pose.covariance[7]  = 0.10   # y
    #     odom.pose.covariance[14] = 0.10   # z
    #     odom.pose.covariance[35] = 0.15   # yaw
    #     odom.twist.covariance[0]  = 0.20
    #     odom.twist.covariance[7]  = 0.20
    #     odom.twist.covariance[14] = 0.20
    #     odom.twist.covariance[35] = 0.30
    #     self.odom_pub.publish(odom)

    #     # TF odom → base_link
    #     tf = TransformStamped()
    #     tf.header.stamp    = stamp
    #     tf.header.frame_id = 'odom'
    #     tf.child_frame_id  = 'base_link'
    #     tf.transform.translation.x = self.x_odom
    #     tf.transform.translation.y = self.y_odom
    #     tf.transform.translation.z = self.z_odom
    #     tf.transform.rotation.x = float(q[0])
    #     tf.transform.rotation.y = float(q[1])
    #     tf.transform.rotation.z = float(q[2])
    #     tf.transform.rotation.w = float(q[3])
    #     self.tf_broadcaster.sendTransform(tf)

    # =========================================================================
    # Dead-reckoning
    # =========================================================================

    # def _compute_odometry(self, idx: int, t: float):
    #     """
    #     Integración dead-reckoning con vel_body + gyro.
    #     Ambos son GT derivados pero es la única fuente de movimiento disponible.

    #     NED body → ROS body:
    #         vx_ros =  vb[0]   (fwd)
    #         vy_ros = -vb[1]   (left)
    #         vz_ros = -vb[2]   (up)
    #         wz_ros = -gyro[2] (CCW positivo)
    #     """
    #     if self.last_time is None:
    #         self.last_time = t
    #         return

    #     dt = t - self.last_time
    #     self.last_time = t

    #     if dt <= 0.0 or dt > DT_MAX:
    #         return

    #     # ── Yaw ──────────────────────────────────────────────────────────────
    #     wz_ned = float(self.gyro[idx, 2])
    #     wz_ned = np.clip(wz_ned, -5.0, 5.0)
    #     wz_ros = -wz_ned
    #     self.yaw_odom += wz_ros * dt
    #     self.yaw_odom  = np.arctan2(
    #         np.sin(self.yaw_odom), np.cos(self.yaw_odom))

    #     # ── Posición: rotar vel_body al frame odom ────────────────────────────
    #     vb     = self.vel_body[idx]
    #     vx_ros =  float(vb[0])
    #     vy_ros = -float(vb[1])
    #     vz_ros = -float(vb[2])

    #     cy = np.cos(self.yaw_odom)
    #     sy = np.sin(self.yaw_odom)
    #     self.x_odom += (cy * vx_ros - sy * vy_ros) * dt
    #     self.y_odom += (sy * vx_ros + cy * vy_ros) * dt
    #     self.z_odom += vz_ros * dt

    #     q = R.from_euler('z', self.yaw_odom).as_quat()
    #     self._publish_odometry(t, vx_ros, vy_ros, vz_ros, wz_ros, q)

    # =========================================================================
    # Inicialización
    # =========================================================================

    def _publish_initial_state(self):

        t0    = float(self.imu_times[0])
        p0_robot_enu = ned_pose_to_enu(self.robot_poses[0])

        x0   = float(p0_robot_enu[0])
        y0   = float(p0_robot_enu[1])
        yaw0 = extract_yaw_enu(p0_robot_enu)

        # AMCL INIT: /initialpose con covarianza pequeña (confiamos en el GT inicial)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self._to_ros_time(t0)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x    = x0
        msg.pose.pose.position.y    = y0
        msg.pose.pose.position.z    = float(p0_robot_enu[2])
        msg.pose.pose.orientation.x = float(p0_robot_enu[3])
        msg.pose.pose.orientation.y = float(p0_robot_enu[4])
        msg.pose.pose.orientation.z = float(p0_robot_enu[5])
        msg.pose.pose.orientation.w = float(p0_robot_enu[6])
        cov = [0.0] * 36
        cov[0]  = 0.05   # x  ±0.22 m
        cov[7]  = 0.05   # y  ±0.22 m
        cov[35] = 0.02   # yaw ±8°
        msg.pose.covariance = cov
        self.initial_pose_pub.publish(msg)

        self.get_logger().info(
            f"Pose inicial GT: ({x0:.2f}, {y0:.2f}), "
            f"yaw={np.degrees(yaw0):.1f}°. GT no se usará más.")

        p_cam_enu  = ned_pose_to_enu(self.cam_poses[0])

        T_cam      = self._pose_to_matrix(p_cam_enu)
        T_robot    = self._pose_to_matrix(p0_robot_enu)
        T_base_cam = np.linalg.inv(T_robot) @ T_cam

        tf_base_cam = self.matrix_to_transform(T_base_cam, self.robot_frame,self.cam_frame, t0)
        tf_cam_lidar = self.matrix_to_transform(np.eye(4),self.cam_frame,self.lidar_frame, t0)
    
        self.tf_static_pub.sendTransform([tf_base_cam, tf_cam_lidar])

        # EKF INIT:

        # msg_EKF_init = TransformStamped()
        # msg_EKF_init.header.stamp    = self._to_ros_time(t0)
        # msg_EKF_init.header.frame_id = 'odom'
        # msg_EKF_init.child_frame_id  = self.robot_frame
        # msg_EKF_init.transform.translation.x = float(T[0, 3])
        # msg_EKF_init.transform.translation.y = float(T[1, 3])
        # msg_EKF_init.transform.translation.z = float(T[2, 3])
        # q = R.from_matrix(T[:3, :3]).as_quat()
        # msg_EKF_init.transform.rotation.x = float(q[0])
        # msg_EKF_init.transform.rotation.y = float(q[1])
        # msg_EKF_init.transform.rotation.z = float(q[2])
        # msg_EKF_init.transform.rotation.w = float(q[3])

        # self.tf_broadcaster.sendTransform(msg_EKF_init)




    # =========================================================================
    # Callbacks y bucle principal
    # =========================================================================

    def _map_callback(self, msg):
        """[F2] Timer de un solo disparo — no bloquea spin()."""
        if self.mapa_recibido:
            return
        self.mapa_recibido = True
        self.get_logger().info("Mapa recibido. Inicializando en 3 s...")
        time.sleep(3.0)
        self._on_map_ready()

    def _on_map_ready(self):
        self._publish_initial_state()
        self.cam_index = 0
        self.imu_index = 0
        # [F1] Hilo separado para no bloquear spin()
        threading.Thread(target=self._imu_loop, daemon=True).start()

    def _imu_loop(self):
        """
        Bucle principal de reproducción a frecuencia real del IMU.

        Orden en cada paso:
          1. clock       — todos los sistemas sincronizan tiempo
          2. odometría   — TF odom→base_link actualizado ANTES del scan
          3. cámara      — RGB + depth si corresponde
          4. LiDAR       — scan con timestamp del IMU (TF ya publicado)
        """
        # Sleep basado en dt real del IMU para reproducir a velocidad 1:1
        if len(self.imu_times) > 1:
            imu_dt = float(np.median(np.diff(self.imu_times[:200])))
        else:
            imu_dt = 1.0 / 100.0
        self.get_logger().info(f"IMU dt={imu_dt*1000:.1f} ms → "
                               f"{1.0/imu_dt:.0f} Hz")

        while rclpy.ok():
            with self._imu_lock:
                if self.imu_index >= len(self.imu_times):
                    break
                i = self.imu_index
                self.imu_index += 1

            t_imu = float(self.imu_times[i])

            # 1. Clock
            self._publish_clock(t_imu)
            self._publish_imu(self.imu_index,t_imu)

            # 2. Dead-reckoning + TF odom→base_link
            # self._compute_odometry(i, t_imu)

            # 3. Sincronización cámara/LiDAR
            while (self.cam_index < len(self.cam_times) and
                   self.cam_times[self.cam_index] <= t_imu):

                j = self.cam_index
                self.cam_index += 1

                self._publish_rgb(
                    os.path.join(self.rgb_path,   self.rgb_files[j]),   t_imu)
                self._publish_depth(
                    os.path.join(self.depth_path, self.depth_files[j]), t_imu)

                # 4. Scan con timestamp IMU (TF ya publicado en paso 2)
                self._publish_lidar(j, t_imu)

            time.sleep(imu_dt)

        self.get_logger().info("Fin del dataset.")


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()