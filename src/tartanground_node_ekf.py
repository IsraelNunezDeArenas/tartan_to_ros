#!/usr/bin/env python3
"""
TartanGroundNode — versión con robot_localization EKF
=====================================================

Arquitectura de localización:
                                        ┌─────────────────────────────────┐
  TartanGroundNode ──► /odom           │  ekf_node (robot_localization)  │
  TartanGroundNode ──► /imu/data  ────►│  Fusiona twist + angular_vel    │
                                        │  Publica TF: odom → base_link   │
                                        │  Publica: /odometry/filtered    │
                                        └─────────────┬───────────────────┘
                                                       │ TF odom→base_link
  TartanGroundNode ──► /scan ──────────────────────────►
                                                  amcl
                                              Publica TF: map → odom
                                        └─────────────────────────────────┘

Árbol TF resultante:
  map ──[AMCL]──► odom ──[EKF]──► base_link ──[static]──► camera
                                                       ──► laser

Cambios respecto a la versión anterior (tartanground_node_fixed.py):
  [E1] Se elimina el broadcast de TF odom→base_link del nodo
       (ahora lo gestiona ekf_node exclusivamente)
  [E2] /odom publica SÓLO twist (velocidades) — el EKF integra la posición
       La covarianza de pose se pone muy alta para que EKF la ignore
  [E3] Nuevo publisher /imu/data (sensor_msgs/Imu) con:
         angular_velocity en frame ROS (x=fwd, y=left, z=up)
         linear_acceleration derivada de diferencias finitas de vel_body
         covariances realistas
  [E4] _publish_imu() llamado en cada paso IMU del bucle principal
  [E5] _publish_initial_state() ya no publica TF odom→base_link inicial
       (EKF arranca en x=0,y=0 — AMCL corregirá map→odom con /initialpose)
  [E6] Se añade accel_prev para el cálculo incremental de aceleración
  [E7] Header frame_id del IMU = robot_frame (base_link), no imu_link,
       porque el giroscopio del dataset está en el frame del cuerpo del robot

Convenciones del dataset (TartanAir / TartanGround):
  NED mundo  : x=North, y=East, z=Down
  NED cuerpo : x=fwd,   y=right, z=down  | gyro wz+ = CW desde arriba
  Frame óptico: x=right, y=down, z=fwd   (puntos LiDAR PLY)
  ENU mundo  : x=East,  y=North, z=Up    (ROS map frame)
  ROS body   : x=fwd,   y=left,  z=up    (base_link, REP-103)
"""

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, LaserScan, CameraInfo, Imu
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
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
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de conversión de marcos de referencia
# ─────────────────────────────────────────────────────────────────────────────

# NED mundo → ENU mundo:  (E, N, U) = (y_NED, x_NED, -z_NED)
R_NED2ENU = np.array([
    [0., 1.,  0.],
    [1., 0.,  0.],
    [0., 0., -1.]
], dtype=np.float64)

# NED body (x=fwd, y=right, z=down) → ROS body (x=fwd, y=left, z=up)
# Rotación 180° alrededor del eje x (forward)
R_BODY_NED2ROS = np.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)

# Umbral de dt para descartar integraciones espurias (s)
DT_MAX = 0.5

# Covarianza diagonal para twist en /odom (input al EKF)
# El EKF confía en estas velocidades para integrar la posición
ODOM_TWIST_COV_VXY  = 0.05   # [m/s]²
ODOM_TWIST_COV_WZ   = 0.02   # [rad/s]²

# Covarianza del IMU — angular velocity
IMU_ANG_COV = 0.005           # [rad/s]²

# Covarianza del IMU — linear acceleration (derivada numérica, menos precisa)
IMU_LIN_COV = 0.5             # [m/s²]²


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de conversión
# ─────────────────────────────────────────────────────────────────────────────

def ned_pose_to_enu(p: np.ndarray) -> np.ndarray:
    """
    Convierte pose [x,y,z, qx,qy,qz,qw] de NED mundo + NED body a ENU.
    Resultado: body_ROS (x=fwd, y=left, z=up) en ENU — convención base_link.
    """
    pos_enu    = R_NED2ENU @ p[:3]
    R_body_ned = R.from_quat(p[3:]).as_matrix()
    R_body_enu = R_NED2ENU @ R_body_ned @ R_BODY_NED2ROS
    q_enu      = R.from_matrix(R_body_enu).as_quat()
    return np.array([*pos_enu, *q_enu], dtype=np.float64)


def extract_yaw_enu(p_enu: np.ndarray) -> float:
    """
    Yaw ENU (CCW desde East) proyectando el vector forward (col 0 de R)
    en el plano XY-ENU. Correcto aunque haya roll/pitch no nulos.
    """
    fwd = R.from_quat(p_enu[3:]).as_matrix()[:, 0]
    return float(np.arctan2(fwd[1], fwd[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Nodo principal
# ─────────────────────────────────────────────────────────────────────────────

class TartanGroundNode(Node):

    def __init__(self):
        super().__init__('tartanground_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('dataset_path',
            '/home/israelnunez/tartanairpy/House/Data_omni/P0000')
        self.declare_parameter('topic_rgb_image',    '/camera/rgb')
        self.declare_parameter('topic_depth_image',  '/camera/depth')
        self.declare_parameter('topic_localization_gt', '/amcl_pose')
        self.declare_parameter('topic_camera_info',  '/camera/camera_info')
        self.declare_parameter('camera_localization','/debug/default_camera')
        self.declare_parameter('map_frame_id',    'map')
        self.declare_parameter('robot_frame_id',  'base_link')
        self.declare_parameter('camera_frame_id', 'camera')
        self.declare_parameter('lidar_frame_id',  'laser')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('img_width',  640)
        self.declare_parameter('img_height', 640)
        self.declare_parameter('fx', 320.0)
        self.declare_parameter('fy', 320.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 320.0)

        self.dataset_path   = self.get_parameter('dataset_path').value
        self.topic_rgb      = self.get_parameter('topic_rgb_image').value
        self.topic_depth    = self.get_parameter('topic_depth_image').value
        self.cam_topic_pose = self.get_parameter('camera_localization').value
        self.topic_info     = self.get_parameter('topic_camera_info').value
        self.map_frame      = self.get_parameter('map_frame_id').value
        self.robot_frame    = self.get_parameter('robot_frame_id').value
        self.cam_frame      = self.get_parameter('camera_frame_id').value
        self.lidar_frame    = self.get_parameter('lidar_frame_id').value
        self.topic_robot_pose = self.get_parameter('topic_localization_gt').value
        self.rate           = self.get_parameter('publish_rate').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.fx    = self.get_parameter('fx').value
        self.fy    = self.get_parameter('fy').value
        self.cx    = self.get_parameter('cx').value
        self.cy    = self.get_parameter('cy').value

        # ── Rutas del dataset ─────────────────────────────────────────────────
        self.rgb_path    = os.path.join(self.dataset_path, 'image_lcam_front')
        self.depth_path  = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path  = os.path.join(self.dataset_path, 'lidar')
        self.rgb_files   = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))
        self.lidar_files = sorted(os.listdir(self.lidar_path))

        self._load_times()
        self._load_camera_poses()
        self._load_robot_pose()
        self._load_velocities()

        self.bridge = CvBridge()

        # ── Publishers ────────────────────────────────────────────────────────
        self.rgb_pub          = self.create_publisher(Image, self.topic_rgb, 10)
        self.depth_pub        = self.create_publisher(Image, self.topic_depth, 10)
        self.pose_pub         = self.create_publisher(
            PoseWithCovarianceStamped, self.cam_topic_pose, 10)
        self.scan_pub         = self.create_publisher(LaserScan, '/scan', 10)
        self.robot_pose_pub   = self.create_publisher(
            PoseWithCovarianceStamped, self.topic_robot_pose, 10)
        self.clock_pub        = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub         = self.create_publisher(Odometry, '/odom', 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # [E3] Publisher IMU para el EKF
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
            
        self.info_pub = self.create_publisher(CameraInfo, self.topic_info, qos_latched)
        self.map_sub  = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos_latched)

        # ── Estado de odometría (sólo para calcular twist, EKF integra pos) ──
        self.last_time     = None
        self._last_vel_body = np.zeros(3)   # [E6] para derivada numérica de accel
        self._last_t_imu   = None

        # Yaw inicial (sólo para log — el EKF arranca en yaw=0 en frame odom)
        p0_enu = ned_pose_to_enu(self.robot_poses[0])
        yaw0   = extract_yaw_enu(p0_enu)
        self.get_logger().info(
            f"Yaw inicial ENU (ground truth): {np.degrees(yaw0):.1f}°  "
            f"| Pos inicial: ({p0_enu[0]:.2f}, {p0_enu[1]:.2f}) m")

        # ── Índices de reproducción ───────────────────────────────────────────
        self.cam_index = 0
        self.imu_index = 0

        # ── Control de threading ──────────────────────────────────────────────
        self._imu_lock      = threading.Lock()
        self.mapa_recibido  = False
        self._imu_thread    = None
        self._last_lidar_idx = -1

        # ── TF (sólo estáticos — odom→base_link lo publica el EKF) ──────────
        self.tf_static_pub = StaticTransformBroadcaster(self)

        self.get_logger().info(
            "TartanGroundNode (EKF mode) listo. Esperando mapa en /map...")

    # =========================================================================
    # Carga de datos
    # =========================================================================

    def _load(self, npy: str, txt: str) -> np.ndarray:
        for p in (npy, txt):
            if p and os.path.exists(p):
                return np.load(p) if p.endswith('.npy') else np.loadtxt(p)
        raise FileNotFoundError(f"No encontrado: {npy}  ni  {txt}")

    def _load_times(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.cam_times = self._load(f'{base}/cam_time.npy', f'{base}/cam_time.txt')
        self.imu_times = self._load(f'{base}/imu_time.npy', f'{base}/imu_time.txt')

    def _load_camera_poses(self):
        meta  = os.path.join(self.dataset_path, 'metadata', 'pose_lcam_front.txt')
        plain = os.path.join(self.dataset_path, 'pose_lcam_front.txt')
        self.cam_poses = np.loadtxt(meta if os.path.exists(meta) else plain)

    def _load_robot_pose(self):
        base = os.path.join(self.dataset_path, 'imu')
        pos  = self._load(f'{base}/pos_global.npy', f'{base}/pos_global.txt')
        ori  = self._load(f'{base}/ori_global.npy', f'{base}/ori_global.txt')

        roll, pitch, yaw = ori[:, 0], ori[:, 1], ori[:, 2]
        cr, sr = np.cos(roll/2),  np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2),   np.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        self.robot_poses = np.hstack(
            (pos, np.stack((qx, qy, qz, qw), axis=1)))

    def _load_velocities(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.vel_body = self._load(f'{base}/vel_body.npy', f'{base}/vel_body.txt')
        self.gyro     = self._load(f'{base}/gyro.npy',     f'{base}/gyro.txt')
        # Intentar cargar aceleraciones proporcionadas por el dataset
        try:
            self.acc = self._load(f'{base}/acc.npy', f'{base}/acc.txt')
            self.get_logger().info("acc cargado desde archivo.")
        except FileNotFoundError:
            self.get_logger().info("acc no encontrado — se usará derivada numérica si es necesario.")
            self.acc = None
        try:
            self.vel_global = self._load(
                f'{base}/vel_global.npy', f'{base}/vel_global.txt')
            self.get_logger().info("vel_global cargado desde archivo.")
        except FileNotFoundError:
            self.get_logger().warn(
                "vel_global no encontrado — derivando de vel_body + poses.")
            self.vel_global = self._derive_vel_global()

    def _derive_vel_global(self) -> np.ndarray:
        n  = len(self.vel_body)
        vg = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            vg[i] = R.from_quat(
                self.robot_poses[i, 3:]).as_matrix() @ self.vel_body[i, :3]
        return vg

    # =========================================================================
    # Utilidades
    # =========================================================================

    def _pose_to_matrix(self, p: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R.from_quat(p[3:]).as_matrix()
        T[:3, 3]  = p[:3]
        return T

    def _to_ros_time(self, t: float) -> Time:
        return Time(sec=int(t), nanosec=int((t % 1) * 1e9))

    def find_closest_imu_index(self, cam_t: float) -> int:
        return int(np.argmin(np.abs(self.imu_times - cam_t)))

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

    def _make_pose_cov_msg(self, p_enu: np.ndarray, t: float,
                           cov_xy: float = 0.25,
                           cov_yaw: float = 0.06) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp         = self._to_ros_time(t)
        msg.header.frame_id      = self.map_frame
        msg.pose.pose.position.x    = float(p_enu[0])
        msg.pose.pose.position.y    = float(p_enu[1])
        msg.pose.pose.position.z    = float(p_enu[2])
        msg.pose.pose.orientation.x = float(p_enu[3])
        msg.pose.pose.orientation.y = float(p_enu[4])
        msg.pose.pose.orientation.z = float(p_enu[5])
        msg.pose.pose.orientation.w = float(p_enu[6])
        cov = [0.0] * 36
        cov[0]  = cov_xy
        cov[7]  = cov_xy
        cov[35] = cov_yaw
        msg.pose.covariance = cov
        return msg

    # =========================================================================
    # Publicadores individuales
    # =========================================================================

    def _publish_camera_info(self):
        msg = CameraInfo()
        msg.width  = self.img_w
        msg.height = self.img_h
        msg.k = list(map(float,
            [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]))
        msg.header.stamp    = self._to_ros_time(0.0)
        msg.header.frame_id = self.cam_frame
        self.info_pub.publish(msg)

    def _publish_clock(self, t: float):
        msg = Clock()
        msg.clock = self._to_ros_time(t)
        self.clock_pub.publish(msg)

    def _publish_rgb(self, path: str, t: float):
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"No se pudo leer RGB: {path}")
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.rgb_pub.publish(msg)

    def _publish_depth(self, path: str, t: float):
        """
        TartanAir depth = PNG RGBA donde 4 bytes/píxel = float32 LE.
        Se usa view('<f4') para reinterpretar los bits correctamente.
        """
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            self.get_logger().warn(f"No se pudo leer depth: {path}")
            return
        depth = raw.view(np.float32).reshape(raw.shape[:2])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        msg   = self.bridge.cv2_to_imgmsg(
            np.ascontiguousarray(depth, np.float32), encoding='32FC1')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.depth_pub.publish(msg)

    def _publish_cam_pose(self, idx: int, t: float):
        p_enu = ned_pose_to_enu(self.cam_poses[idx])
        self.pose_pub.publish(
            self._make_pose_cov_msg(p_enu, t))

    def _publish_robot_pose(self, idx: int, t: float):
        """Ground truth pose en ENU para comparación/debug."""
        p_enu = ned_pose_to_enu(self.robot_poses[idx])
        self.robot_pose_pub.publish(
            self._make_pose_cov_msg(p_enu, t))

    def _publish_lidar(self, cam_idx: int, t: float):
        """
        PLY (frame óptico: x=right, y=down, z=fwd) → LaserScan (base_link ENU).
        Timestamp 't' = t_imu del paso actual para garantizar TF válido en EKF.
        Relleno vectorizado con np.minimum.at.
        """
        if cam_idx == self._last_lidar_idx:
            return
        self._last_lidar_idx = cam_idx

        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[cam_idx]))
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            self.get_logger().warn(f"Scan vacío en índice {cam_idx}")
            return

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]   # right, down, fwd

        # Filtro de altura (plano horizontal del robot)
        mask = (ys > -0.3) & (ys < 0.3)
        xs, zs = xs[mask], zs[mask]
        if xs.size == 0:
            return

        # Frame óptico → base_link ROS: x_ros = z_opt, y_ros = -x_opt
        x_ros = zs
        y_ros = -xs

        angle_min       = -np.pi
        angle_max       =  np.pi
        angle_increment = np.deg2rad(0.5)
        num_beams       = int(round((angle_max - angle_min) / angle_increment))

        angles    = np.arctan2(y_ros, x_ros)
        distances = np.hypot(x_ros, y_ros)
        beam_idx  = ((angles - angle_min) / angle_increment).astype(np.int32)

        valid     = (beam_idx >= 0) & (beam_idx < num_beams) & (distances > 0.1)
        np.minimum.at(
            (ranges := np.full(num_beams, np.inf, dtype=np.float32)),
            beam_idx[valid],
            distances[valid])

        scan = LaserScan()
        scan.header.stamp    = self._to_ros_time(t)
        scan.header.frame_id = self.lidar_frame
        scan.angle_min       = float(angle_min)
        scan.angle_max       = float(angle_max)
        scan.angle_increment = float(angle_increment)
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / self.rate
        scan.range_min       = 0.1
        scan.range_max       = 50.0
        scan.ranges          = ranges.tolist()
        self.scan_pub.publish(scan)

    # =========================================================================
    # [E3][E4] Publicador IMU para robot_localization EKF
    # =========================================================================

    def _publish_imu(self, idx: int, t: float):
        """
        Publica sensor_msgs/Imu con datos del dataset en frame ROS base_link.

        Conversión NED body → ROS body (x=fwd, y=left, z=up):
          wx_ros =  gyro[0]    (roll rate: mismo eje forward)
          wy_ros = -gyro[1]    (pitch rate: flip eje y)
          wz_ros = -gyro[2]    (yaw rate: flip eje z, CW→CCW)

        Aceleración lineal:
          Derivada finita de vel_body convertida a ROS.
          Covarianza alta porque es una aproximación numérica.

        Orientación:
          No disponible directamente del IMU del dataset.
          Se indica con orientation_covariance[0] = -1.
          El EKF no usará orientación absoluta del IMU.
        """
        imu = Imu()
        imu.header.stamp    = self._to_ros_time(t)
        imu.header.frame_id = self.robot_frame   # base_link

        # ── Angular velocity (NED body → ROS body) ────────────────────────────
        wx_ned, wy_ned, wz_ned = (float(v) for v in self.gyro[idx])
        wx_ned = np.clip(wx_ned, -10.0, 10.0)
        wy_ned = np.clip(wy_ned, -10.0, 10.0)
        wz_ned = np.clip(wz_ned, -10.0, 10.0)

        imu.angular_velocity.x =  wx_ned
        imu.angular_velocity.y = -wy_ned
        imu.angular_velocity.z = -wz_ned

        av_cov = [0.0] * 9
        av_cov[0] = IMU_ANG_COV   # wx
        av_cov[4] = IMU_ANG_COV   # wy
        av_cov[8] = IMU_ANG_COV   # wz
        imu.angular_velocity_covariance = av_cov

        # ── Linear acceleration: usar acc si el dataset la aporta,
        #    en caso contrario calcular Δv/Δt (derivada numérica).
        ax_ros = ay_ros = az_ros = 0.0
        vb = self.vel_body[idx]
        if getattr(self, 'acc', None) is not None:
            ab = self.acc[idx]
            # NED body → ROS body: ax_ros=ab[0], ay_ros=-ab[1], az_ros=-ab[2]
            ax_ros = float(ab[0])
            ay_ros = -float(ab[1])
            az_ros = -float(ab[2])
        else:
            if self._last_t_imu is not None and (t - self._last_t_imu) > 0.0:
                dt_imu = t - self._last_t_imu
                dvb    = (vb - self._last_vel_body) / dt_imu
                ax_ros =  float(dvb[0])
                ay_ros = -float(dvb[1])
                az_ros = -float(dvb[2])

        # Actualizar estado temporal para la siguiente derivada
        try:
            self._last_vel_body = vb.copy()
        except Exception:
            self._last_vel_body = np.array(vb)
        self._last_t_imu    = t

        imu.linear_acceleration.x = ax_ros
        imu.linear_acceleration.y = ay_ros
        imu.linear_acceleration.z = az_ros

        la_cov = [0.0] * 9
        la_cov[0] = IMU_LIN_COV   # ax
        la_cov[4] = IMU_LIN_COV   # ay
        la_cov[8] = IMU_LIN_COV   # az
        imu.linear_acceleration_covariance = la_cov

        # ── Orientación no disponible ─────────────────────────────────────────
        # orientation_covariance[0] = -1 → EKF ignora el campo orientation
        imu.orientation_covariance[0] = -1.0

        self.imu_pub.publish(imu)

    # =========================================================================
    # [E2] Odometría — sólo twist (el EKF integra la posición)
    # =========================================================================

    def _publish_odom_twist(self, idx: int, t: float):
        """
        Publica /odom con SÓLO la información de velocidad (twist) para el EKF.

        La posición en el mensaje se pone a cero con covarianza muy alta
        para que el EKF la ignore completamente y sólo use el twist.

        Twist en frame base_link (ROS):
          vx = vel_body[0]    (forward, sin cambio)
          vy = -vel_body[1]   (left = -right)
          wz = -gyro[2]       (CCW = -CW)
        """
        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        if dt <= 0.0 or dt > DT_MAX:
            return

        vb     = self.vel_body[idx]
        vx_ros =  float(np.clip(vb[0], -20.0, 20.0))
        vy_ros = -float(np.clip(vb[1], -20.0, 20.0))
        wz_ned =  float(np.clip(self.gyro[idx, 2], -5.0, 5.0))
        wz_ros = -wz_ned

        stamp = self._to_ros_time(t)

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        # Posición a cero con covarianza altísima → EKF la ignora
        odom.pose.pose.position.x    = 0.0
        odom.pose.pose.position.y    = 0.0
        odom.pose.pose.position.z    = 0.0
        odom.pose.pose.orientation.w = 1.0
        # Covarianza de pose muy alta (EKF no confía en esta posición)
        POSE_COV_HIGH = 0.005
        pose_cov       = [0.0] * 36
        pose_cov[0]    = POSE_COV_HIGH   # x
        pose_cov[7]    = POSE_COV_HIGH   # y
        pose_cov[14]   = POSE_COV_HIGH   # z
        pose_cov[21]   = POSE_COV_HIGH   # roll
        pose_cov[28]   = POSE_COV_HIGH   # pitch
        pose_cov[35]   = POSE_COV_HIGH   # yaw
        odom.pose.covariance = pose_cov

        # Twist con covarianza realista
        odom.twist.twist.linear.x   = vx_ros
        odom.twist.twist.linear.y   = vy_ros
        odom.twist.twist.angular.z  = wz_ros

        twist_cov      = [0.0] * 36
        twist_cov[0]   = ODOM_TWIST_COV_VXY   # vx
        twist_cov[7]   = ODOM_TWIST_COV_VXY   # vy
        twist_cov[35]  = ODOM_TWIST_COV_WZ    # wz
        odom.twist.covariance = twist_cov

        self.odom_pub.publish(odom)

    # =========================================================================
    # Estado inicial
    # =========================================================================

    def _publish_initial_state(self):
        """
        [E5] Publica /initialpose para AMCL y TF estáticos.
             El EKF arranca con x=0,y=0 en odom frame.
             AMCL estimará la corrección map→odom para cerrar el bucle global.
        """
        p_ned = self.robot_poses[0]
        p_enu = ned_pose_to_enu(p_ned)
        t0    = float(self.imu_times[0])

        # /initialpose para AMCL con incertidumbre moderada (~0.5 m, ~14°)

        self.get_logger().info(
            f"/initialpose ENU: ({p_enu[0]:.2f}, {p_enu[1]:.2f}) m, "
            f"yaw={np.degrees(extract_yaw_enu(p_enu)):.1f}°")

        # ── TF estáticos ──────────────────────────────────────────────────────
        # base_link → camera (cámara frontal)
        p_cam_enu  = ned_pose_to_enu(self.cam_poses[0])
        T_cam      = self._pose_to_matrix(p_cam_enu)
        T_robot    = self._pose_to_matrix(p_enu)
        T_base_cam = np.linalg.inv(T_robot) @ T_cam

        t0 = float(self.imu_times[0])

    # ── Pose inicial desde GT (único uso) ────────────────────────────────
    p0_enu = ned_pose_to_enu(self.robot_poses[0])

    x0   = float(p0_enu[0])
    y0   = float(p0_enu[1])
    yaw0 = extract_yaw_enu(p0_enu)

    # Inicializar el dead-reckoning en la posición GT
    # A partir de aquí, _compute_odometry integra desde este punto
    self.x_odom   = x0
    self.y_odom   = y0
    self.z_odom   = float(p0_enu[2])
    self.yaw_odom = yaw0

    # ── /initialpose para AMCL con covarianza pequeña ────────────────────
    # Covarianza pequeña: le decimos a AMCL que confiamos en este punto
    msg = PoseWithCovarianceStamped()
    msg.header.stamp    = self._to_ros_time(t0)
    msg.header.frame_id = self.map_frame
    msg.pose.pose.position.x    = x0
    msg.pose.pose.position.y    = y0
    msg.pose.pose.position.z    = float(p0_enu[2])
    msg.pose.pose.orientation.x = float(p0_enu[3])
    msg.pose.pose.orientation.y = float(p0_enu[4])
    msg.pose.pose.orientation.z = float(p0_enu[5])
    msg.pose.pose.orientation.w = float(p0_enu[6])
    cov = [0.0] * 36
    cov[0]  = 0.05   # x:   ±0.22 m  — confiamos en el GT inicial
    cov[7]  = 0.05   # y:   ±0.22 m
    cov[35] = 0.02   # yaw: ±8°
    msg.pose.covariance = cov
    self.initial_pose_pub.publish(msg)

    self.get_logger().info(
        f"Pose inicial GT: ({x0:.2f}, {y0:.2f}), yaw={np.degrees(yaw0):.1f}°")

    # ── TF estático base_link → laser (sin GT, identidad) ────────────────
    tf_base_lidar = self.matrix_to_transform(
        np.eye(4), self.robot_frame, self.lidar_frame, t0)
    self.tf_static_pub.sendTransform([tf_base_lidar])

    # ── Primer TF odom→base_link arrancando en la posición GT ────────────
    q0 = R.from_euler('z', yaw0).as_quat()
    self._publish_odometry(t0, 0.0, 0.0, 0.0, 0.0, q0)
        

    tf_base_cam  = self.matrix_to_transform(
    T_base_cam, self.robot_frame, self.cam_frame, t0)

        # camera → laser (coinciden en este dataset)
    tf_cam_lidar = self.matrix_to_transform(
            np.eye(4), self.cam_frame, self.lidar_frame, t0)


    self.tf_static_pub.sendTransform([tf_base_cam, tf_cam_lidar])

    # Primer scan: esperar un momento para que EKF active su primer TF
    time.sleep(1.5)
    self._publish_lidar(0, t0)
    self.get_logger().info(
        "Estado inicial publicado. EKF gestionará TF odom→base_link.")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def map_callback(self, msg):
        """
        Se dispara al recibir el mapa de /map.
        Usa create_timer para no bloquear rclpy.spin().
        """
        if self.mapa_recibido:
            return
        self.mapa_recibido = True
        self.get_logger().info("Mapa recibido. Iniciando en 3 s...")
        self._publish_camera_info()
        time.sleep(10.0)
        # self._init_timer = self.create_timer(3.0, self._delayed_start)
        self._delayed_start()
        return 

    def _delayed_start(self):
        """Un solo disparo: publica estado inicial y arranca el bucle IMU."""
        self._publish_initial_state()

        self.cam_index = 0
        self.imu_index = 0

        self._imu_thread = threading.Thread(
            target=self._imu_loop, daemon=True)
        self._imu_thread.start()
        return

    # =========================================================================
    # Bucle principal
    # =========================================================================

    def _imu_loop(self):
        """
        Bucle de reproducción sincronizada IMU + cámara/LiDAR.

        Cada paso IMU:
          1. /clock
          2. /imu/data  (E3 — input al EKF: angular velocity)
          3. /odom      (E2 — input al EKF: twist velocity)
          4. /gt/robot_pose (ground truth para comparación)
          5. Si cam_times[j] <= t_imu → RGB, depth, cam_pose, /scan

        El EKF fusiona /odom + /imu/data y publica TF odom→base_link.
        AMCL usa /scan + TF para publicar TF map→odom.
        """
        sleep_dt = 1.0 / (self.rate * 5.0)   # ~0.02 s si rate=10

        while rclpy.ok():
            with self._imu_lock:
                if self.imu_index >= len(self.imu_times):
                    break
                i = self.imu_index
                self.imu_index += 1

            t_imu = float(self.imu_times[i])

            self.get_logger().info(f"Tiempo: {t_imu}")

            # ── Paso IMU ─────────────────────────────────────────────────────
            self._publish_clock(t_imu)
            self._publish_imu(i, t_imu)            # [E3] → EKF
            self._publish_odom_twist(i, t_imu)     # [E2] → EKF
            
            self._publish_robot_pose(i, t_imu)     # ground truth

            # ── Sincronización cámara/LiDAR ──────────────────────────────────
            while (self.cam_index < len(self.cam_times) and
                   self.cam_times[self.cam_index] <= t_imu):

                j = self.cam_index
                self.cam_index += 1

                self._publish_rgb(
                    os.path.join(self.rgb_path, self.rgb_files[j]), t_imu)
                self._publish_depth(
                    os.path.join(self.depth_path, self.depth_files[j]), t_imu)
                self._publish_cam_pose(j, t_imu)
                self._publish_lidar(j, t_imu)      # timestamp IMU → TF válido

            time.sleep(sleep_dt)

        self.get_logger().info("Fin del bucle IMU.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
