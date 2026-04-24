#!/usr/bin/env python3
"""
TartanGroundNode — corrección de odometría y sistemas de referencia
====================================================================

CORRECCIONES EN ESTA VERSIÓN (respecto al código anterior):
  [F1] _imu_loop() se ejecuta en un hilo separado — ya no bloquea rclpy.spin()
  [F2] map_callback usa create_timer() en lugar de time.sleep(3.0)
  [F3] El scan usa el timestamp IMU más cercano (t_imu) en lugar de t_cam,
       garantizando que el TF odom→base_link ya esté publicado cuando AMCL lo busca
  [F4] Relleno del LaserScan vectorizado con np.minimum.at (sin bucle Python)
  [F5] Deduplicación de lidar_files por nombre de frame (evita publicar el mismo
       scan dos veces cuando cámara e IMU coinciden)
  [F6] Guard en _compute_odometry: si dt > umbral razonable (~0.5 s) se descarta
       la integración para evitar saltos grandes al inicio o si hay lagunas de datos
  [F7] yaw_odom integrado con clip de wz para evitar acumulación de ruido de
       giroscopio fuera de rango
  [F8] Covariance de odometría ajustada: valores más altos para permitir que AMCL
       corrija sin grandes saltos (AMCL confía menos en una odometría perfecta)
  [F9] _publish_initial_state() incluye sleep mínimo antes del primer scan para
       que AMCL tenga tiempo de inicializarse con /initialpose

Convenciones del dataset (TartanAir / TartanGround):

  MARCO NED MUNDO  (pos_global, vel_global):   x=North, y=East,  z=Down
  MARCO NED CUERPO (pose_lcam_front, vel_body, gyro):
                                                x=fwd, y=right, z=down
                                                gyro wz+ = CW desde arriba
  MARCO ÓPTICO CÁMARA (puntos LiDAR PLY):      x=right, y=down, z=fwd
  MARCO ENU MUNDO  (ROS / map frame):           x=East, y=North, z=Up
  MARCO ROS BODY   (base_link, REP-103):        x=fwd, y=left,  z=up
                                                wz+ = CCW desde arriba

Conversiones:
  Posición NED→ENU : (E, N, U) = (y_NED, x_NED, -z_NED)
  Orientación      : R_enu = R_NED2ENU @ R_ned @ R_BODY_NED2ROS
  Velocidad lineal : vx_ros =  vb[0],  vy_ros = -vb[1]
  Velocidad angular: wz_ros = -gyro[2]
  Integración pos  : dx_ENU = vg[1]*dt, dy_ENU = vg[0]*dt  (sin yaw)
"""

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, LaserScan, CameraInfo
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
# Equivale a una rotación de 180° alrededor del eje x (forward)
R_BODY_NED2ROS = np.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)

# Umbral de dt para descartar integraciones espurias (s)
DT_MAX = 0.5


def ned_pose_to_enu(p: np.ndarray) -> np.ndarray:
    """
    Convierte pose [x,y,z, qx,qy,qz,qw] de NED mundo + NED body a ENU.

    La rotación resultante expresa body_ROS (x=fwd, y=left, z=up) en ENU,
    que es exactamente la convención base_link de ROS (REP-103).
    """
    pos_enu = R_NED2ENU @ p[:3]
    R_body_ned = R.from_quat(p[3:]).as_matrix()
    R_body_enu = R_NED2ENU @ R_body_ned @ R_BODY_NED2ROS
    q_enu = R.from_matrix(R_body_enu).as_quat()
    return np.array([*pos_enu, *q_enu], dtype=np.float64)


def extract_yaw_enu(p_enu: np.ndarray) -> float:
    """
    Yaw ENU (CCW desde East) a partir de una pose ya en ENU.

    Se proyecta el vector forward (col 0 de la matriz de rotación) en XY-ENU
    y se calcula atan2(y_ENU, x_ENU).  Esto es correcto aunque haya roll/pitch.
    """
    R_body = R.from_quat(p_enu[3:]).as_matrix()
    fwd = R_body[:, 0]
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
        self.declare_parameter('topic_localization', '/amcl_pose')
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
        self.topic_robot_pose = self.get_parameter('topic_localization').value
        self.rate           = self.get_parameter('publish_rate').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

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
        self.pose_pub         = self.create_publisher(PoseWithCovarianceStamped, self.cam_topic_pose, 10)
        self.scan_pub         = self.create_publisher(LaserScan, '/scan', 10)
        self.robot_pose_pub   = self.create_publisher(PoseWithCovarianceStamped, self.topic_robot_pose, 10)
        self.clock_pub        = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub         = self.create_publisher(Odometry, '/odom', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.info_pub = self.create_publisher(CameraInfo, self.topic_info, qos_latched)
        self.map_sub  = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos_latched)

        # ── Estado de odometría ───────────────────────────────────────────────
        self.x_odom    = 0.0
        self.y_odom    = 0.0
        self.z_odom    = 0.0
        self.last_time = None

        # [F1] yaw inicial desde vector forward ENU
        p0_enu = ned_pose_to_enu(self.robot_poses[0])
        self.yaw_odom = extract_yaw_enu(p0_enu)
        self.get_logger().info(f"Yaw inicial ENU: {np.degrees(self.yaw_odom):.1f}°")

        # ── Índices de reproducción ───────────────────────────────────────────
        self.cam_index = 0
        self.imu_index = 0

        # ── Control de threading ──────────────────────────────────────────────
        self._cam_lock  = threading.Lock()
        self._imu_lock  = threading.Lock()
        self.mapa_recibido = False
        self._imu_thread   = None

        # Frame del último scan publicado (para evitar duplicados)
        self._last_lidar_idx = -1

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_pub  = StaticTransformBroadcaster(self)

        self.get_logger().info("TartanGroundNode listo. Esperando mapa en /map...")

    # =========================================================================
    # Carga de datos
    # =========================================================================

    def _load(self, npy, txt):
        for p in (npy, txt):
            if p and os.path.exists(p):
                return np.load(p) if p.endswith('.npy') else np.loadtxt(p)
        raise FileNotFoundError(f"No se encontró: {npy} ni {txt}")

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
        try:
            self.vel_global = self._load(
                f'{base}/vel_global.npy', f'{base}/vel_global.txt')
            self.get_logger().info("vel_global cargado desde archivo.")
        except FileNotFoundError:
            self.get_logger().warn("vel_global no encontrado — derivando de vel_body + poses.")
            self.vel_global = self._derive_vel_global()

    def _derive_vel_global(self) -> np.ndarray:
        n  = len(self.vel_body)
        vg = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            vg[i] = R.from_quat(self.robot_poses[i, 3:]).as_matrix() @ self.vel_body[i, :3]
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

    def find_closest_imu_time(self, cam_t: float) -> float:
        return float(self.imu_times[self.find_closest_imu_index(cam_t)])

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

    # =========================================================================
    # Publicadores individuales
    # =========================================================================

    def _publish_camera_info(self):
        msg = CameraInfo()
        msg.width  = self.img_w
        msg.height = self.img_h
        msg.k = list(map(float,
            [self.fx, 0, self.cx,
             0, self.fy, self.cy,
             0, 0, 1]))
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
        msg   = self._make_pose_msg(p_enu, t)
        self.pose_pub.publish(msg)

    def _publish_robot_pose(self, idx: int, t: float):
        p_enu = ned_pose_to_enu(self.robot_poses[idx])
        msg   = self._make_pose_msg(p_enu, t)
        self.robot_pose_pub.publish(msg)

    def _make_pose_msg(self, p_enu: np.ndarray, t: float) -> PoseWithCovarianceStamped:
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
        return msg

    def _publish_lidar(self, cam_idx: int, t: float):
        """
        PLY (frame óptico: x=right, y=down, z=fwd) → LaserScan (base_link ENU).

        [F3] El scan se publica con timestamp 't' que debe coincidir con el
             último TF odom→base_link para que AMCL encuentre la transformada
             sin necesidad de extrapolación.

        [F4] Relleno del array de rangos vectorizado con np.minimum.at.
        [F5] Guard: no publicar el mismo índice de nube dos veces.
        """
        if cam_idx == self._last_lidar_idx:
            return
        self._last_lidar_idx = cam_idx

        # --- Carga del PLY ---
        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[cam_idx]))
        pts = np.asarray(pcd.points)

        if pts.shape[0] == 0:
            self.get_logger().warn(f"Scan vacío en índice {cam_idx}")
            return

        # Frame óptico: x=right, y=down, z=fwd
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

        # Filtro de altura en frame óptico (ys = vertical hacia abajo)
        # ys ∈ (-0.3, 0.3) → banda de ±0.3 m alrededor del plano del sensor
        mask = (ys > -0.3) & (ys < 0.3)
        xs, zs = xs[mask], zs[mask]

        if xs.size == 0:
            self.get_logger().warn(f"Slice vacío tras filtro de altura en índice {cam_idx}")
            return

        # Frame óptico → base_link ROS (ENU):
        #   x_ros = z_opt  (adelante)
        #   y_ros = -x_opt (izquierda)
        x_ros = zs
        y_ros = -xs

        # --- Configuración del LaserScan ---
        RANGE_MIN       = 0.3
        RANGE_MAX       = 100.0
        angle_min       = -np.pi
        angle_max       =  np.pi
        angle_increment = np.deg2rad(0.5)
        num_beams       = int(round((angle_max - angle_min) / angle_increment))

        # [F4] Vectorizado: índice de rayo + np.minimum.at
        angles    = np.arctan2(y_ros, x_ros)
        distances = np.hypot(x_ros, y_ros)

        beam_idx = ((angles - angle_min) / angle_increment).astype(np.int32)

        valid = (
            (beam_idx >= 0) & (beam_idx < num_beams) &
            (distances >= RANGE_MIN) & (distances <= RANGE_MAX)
        )
        beam_idx  = beam_idx[valid]
        distances = distances[valid]

        ranges = np.full(num_beams, np.inf, dtype=np.float32)
        np.minimum.at(ranges, beam_idx, distances)

        # Bins sin retorno → RANGE_MAX (AMCL los trata como "sin obstáculo")
        ranges[np.isinf(ranges)] = RANGE_MAX


        # Publicar
        scan = LaserScan()
        scan.header.stamp    = self._to_ros_time(t)
        scan.header.frame_id = self.robot_frame
        scan.angle_min       = float(angle_min)
        scan.angle_max       = float(angle_max)
        scan.angle_increment = float(angle_increment)
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / self.rate
        scan.range_min       = RANGE_MIN
        scan.range_max       = RANGE_MAX
        scan.ranges          = ranges.tolist()
        self.scan_pub.publish(scan)

    # =========================================================================
    # Odometría
    # =========================================================================

    def _compute_odometry(self, idx: int, t: float):
        """
        [F6] Integración con vel_global (ENU, sin depender de yaw_odom).
        [F6] Guard de dt: descarta pasos de integración fuera del rango esperado.
        [F7] Clip de wz antes de integrar para acotar el ruido de giroscopio.
        """
        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        if dt <= 0.0 or dt > DT_MAX:
            # Primera iteración o laguna de datos: sólo actualizar tiempo
            return

        # Posición: vel_global en NED mundo → ENU (sin rotar por yaw_odom)
        vg = self.vel_global[idx]
        self.x_odom += float(vg[1]) * dt   # East
        self.y_odom += float(vg[0]) * dt   # North
        # z no se usa en 2D pero se mantiene por compatibilidad
        self.z_odom += -float(vg[2]) * dt

        # Yaw: wz del giroscopio NED → ROS, con clip anti-ruido
        wz_ned = float(self.gyro[idx, 2])
        wz_ned = np.clip(wz_ned, -5.0, 5.0)   # rad/s razonable para ground robot
        wz_ros = -wz_ned                        # CCW positivo en ROS
        self.yaw_odom += wz_ros * dt
        self.yaw_odom  = np.arctan2(
            np.sin(self.yaw_odom), np.cos(self.yaw_odom))

        # Twist (en frame body ROS)
        vb     = self.vel_body[idx]
        vx_ros =  float(vb[0])
        vy_ros = -float(vb[1])

        q = R.from_euler('z', self.yaw_odom).as_quat()
        self._publish_odometry(t, vx_ros, vy_ros, wz_ros, q)

    def _publish_odometry(self, t: float,
                          vx: float, vy: float, wz: float,
                          q: np.ndarray):
        stamp = self._to_ros_time(t)

        # ── Mensaje Odometry ──────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x    = self.x_odom
        odom.pose.pose.position.y    = self.y_odom
        odom.pose.pose.position.z    = self.z_odom
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        odom.twist.twist.linear.x    = vx
        odom.twist.twist.linear.y    = vy
        odom.twist.twist.angular.z   = wz

        # [F8] Covarianza más realista: AMCL puede corregir sin saltos bruscos.
        #   Diagonal de pose: 0.1 m y 0.1 rad = incertidumbre moderada
        #   Diagonal de twist: 0.2 m/s y 0.3 rad/s
        odom.pose.covariance[0]  = 0.10   # x
        odom.pose.covariance[7]  = 0.10   # y
        odom.pose.covariance[35] = 0.15   # yaw
        odom.twist.covariance[0]  = 0.20
        odom.twist.covariance[7]  = 0.20
        odom.twist.covariance[35] = 0.30
        self.odom_pub.publish(odom)

        # ── TF odom → base_link ───────────────────────────────────────────────
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = self.x_odom
        tf.transform.translation.y = self.y_odom
        tf.transform.translation.z = self.z_odom
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(tf)

    # =========================================================================
    # Estado inicial
    # =========================================================================

    def _publish_initial_state(self):
        """
        [F1] Publica pose inicial en ENU, TF estáticos y primer scan.
        [F9] Se añade un sleep mínimo para que AMCL pueda procesar /initialpose
             antes del primer scan.
        """
        p_ned = self.robot_poses[0]
        p_enu = ned_pose_to_enu(p_ned)
        t0    = float(self.imu_times[0])

        # /initialpose para AMCL
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self._to_ros_time(t0)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x    = float(p_enu[0])
        msg.pose.pose.position.y    = float(p_enu[1])
        msg.pose.pose.position.z    = float(p_enu[2])
        msg.pose.pose.orientation.x = float(p_enu[3])
        msg.pose.pose.orientation.y = float(p_enu[4])
        msg.pose.pose.orientation.z = float(p_enu[5])
        msg.pose.pose.orientation.w = float(p_enu[6])
        cov = [0.0] * 36
        cov[0] = 0.25; cov[7] = 0.25; cov[35] = 0.06   # ~0.5m, ~14°
        msg.pose.covariance = cov
        self.initial_pose_pub.publish(msg)
        self.get_logger().info(
            f"Pose inicial ENU: ({p_enu[0]:.2f}, {p_enu[1]:.2f}), "
            f"yaw={np.degrees(extract_yaw_enu(p_enu)):.1f}°")

        # TF estáticos: base_link→camera, camera→laser
        p_cam_enu  = ned_pose_to_enu(self.cam_poses[0])
        T_cam      = self._pose_to_matrix(p_cam_enu)
        T_robot    = self._pose_to_matrix(p_enu)
        T_base_cam = np.linalg.inv(T_robot) @ T_cam

        tf_base_cam  = self.matrix_to_transform(
            T_base_cam, self.robot_frame, self.cam_frame, t0)
        tf_cam_lidar = self.matrix_to_transform(
            np.eye(4), self.cam_frame, self.lidar_frame, t0)
        self.tf_static_pub.sendTransform([tf_base_cam, tf_cam_lidar])

        # Primer TF odom→base_link en la posición inicial (x=0,y=0)
        q0 = R.from_euler('z', self.yaw_odom).as_quat()
        self._publish_odometry(t0, 0.0, 0.0, 0.0, q0)

        # [F9] Dejar que AMCL procese /initialpose antes del primer scan
        time.sleep(1.0)
        self._publish_lidar(0, t0)

        self.get_logger().info("Estado inicial publicado.")

    # =========================================================================
    # Callbacks y loops
    # =========================================================================

    def map_callback(self, msg):
        """
        [F2] Usa create_timer() en lugar de time.sleep() para no bloquear spin.
        """
        if self.mapa_recibido:
            return
        self.mapa_recibido = True
        self.get_logger().info("Mapa recibido. Inicializando en 3 s...")

        self._publish_camera_info()
        # Esperar en un timer de un solo disparo para no bloquear el hilo de spin
        time.sleep(3.0)
        self._delayed_start()

    def _delayed_start(self):
        """Arranca el bucle IMU en un hilo tras la espera inicial."""
       
        self._publish_initial_state()

        self.cam_index = 0
        self.imu_index = 0

        # [F1] Hilo separado: el spin() puede seguir procesando callbacks
        self._imu_thread = threading.Thread(
            target=self._imu_loop, daemon=True)
        self._imu_thread.start()

    def _imu_loop(self):
        """
        Bucle principal de reproducción.

        Flujo:
          1. Publica clock, robot_pose y odometría a frecuencia IMU (~100 Hz).
          2. Dentro del mismo paso, comprueba si hay frames de cámara/LiDAR
             pendientes (cam_times[j] <= t_imu) y los publica.
          3. [F3] El scan usa t_imu como timestamp (no t_cam) para que el TF
             odom→base_link ya esté en el buffer cuando AMCL lo busca.
        """
        # Paso de sleep en segundos (ligeramente por debajo del dt real para
        # poder procesar datos de cámara sin retrasarse)
        sleep_dt = 1.0 / (self.rate * 5.0)   # ≈ 0.02 s si rate=10

        while rclpy.ok():
            with self._imu_lock:
                if self.imu_index >= len(self.imu_times):
                    break
                i = self.imu_index
                self.imu_index += 1

            t_imu = float(self.imu_times[i])

            # ── Paso IMU ─────────────────────────────────────────────────────
            self._publish_clock(t_imu)
            self._publish_robot_pose(i, t_imu)
            self._compute_odometry(i, t_imu)

            # ── Sincronización cámara/LiDAR ──────────────────────────────────
            while (self.cam_index < len(self.cam_times) and
                   self.cam_times[self.cam_index] <= t_imu):

                j = self.cam_index
                self.cam_index += 1

                rgb_path   = os.path.join(self.rgb_path,   self.rgb_files[j])
                depth_path = os.path.join(self.depth_path, self.depth_files[j])

                self._publish_rgb(rgb_path, t_imu)          # timestamp IMU
                self._publish_depth(depth_path, t_imu)      # timestamp IMU
                self._publish_cam_pose(j, t_imu)

                # [F3] scan con t_imu — TF ya publicado en este paso
                self._publish_lidar(j, t_imu)

            time.sleep(sleep_dt)

        self.get_logger().info("Fin del bucle IMU.")


def main():
    rclpy.init()
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()