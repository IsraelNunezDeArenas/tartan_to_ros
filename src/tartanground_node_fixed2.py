#!/usr/bin/env python3
"""
TartanGroundNode — corrección de odometría y sistemas de referencia
====================================================================

Convenciones del dataset (documentadas en AIcrowd TartanAir challenge):

  MARCO NED MUNDO (pos_global, vel_global):
    x = North, y = East, z = Down

  MARCO NED CUERPO (pose_lcam_front, vel_body, gyro):
    x = forward (adelante del robot)
    y = right   (derecha del robot)
    z = down    (abajo)
    gyro wz+ = rotación horaria vista desde arriba (Right-Hand, eje z=down)

  MARCO ÓPTICO CÁMARA (puntos LiDAR PLY):
    x = right, y = down, z = forward
    El LiDAR se genera proyectando las imágenes de profundidad,
    por lo que los puntos están en este frame, NO en NED cuerpo.

  MARCO ENU MUNDO (ROS, AMCL, map frame):
    x = East, y = North, z = Up

  MARCO ROS BODY (REP-103, base_link):
    x = forward, y = left (izquierda), z = up
    wz+ = CCW vista desde arriba

Conversiones necesarias:

  Posición NED→ENU:
    x_ENU = y_NED (East), y_ENU = x_NED (North), z_ENU = -z_NED (Up)

  Orientación:
    R_body_enu = R_NED2ENU @ R_body_ned @ R_body2ros
    donde:
      R_NED2ENU   = [[0,1,0],[1,0,0],[0,0,-1]]
      R_body2ros  = [[1,0,0],[0,-1,0],[0,0,-1]]  (NED body → ROS body)

  Velocidad lineal para Odometry/twist:
    vx_ros =  vel_body[0]   (forward, sin cambio)
    vy_ros = -vel_body[1]   (left = -right)
    vz_ros =  0             (robot en suelo)

  Velocidad angular:
    wz_ros = -gyro[2]       (CCW desde Up = -CW desde Down)

  Integración de posición (usando vel_global — sin dependencia de yaw):
    dx_ENU = vel_global[1] * dt   (East)
    dy_ENU = vel_global[0] * dt   (North)

  Extracción de yaw ENU desde quaternión NED-pose:
    R_body_enu = ned_pose_to_enu(p)[3:]→matrix
    forward_enu = R_body_enu[:, 0]    (columna 0 = eje x body = forward)
    yaw_enu = atan2(forward_enu[1], forward_enu[0])

Bugs corregidos respecto a versiones anteriores:
  [B1] NED→ENU conversión completa (posición + orientación + body frame)
  [B2] Decodificación depth: depth_rgba.view('<f4')
  [B3] LiDAR: plano horizontal XZ (x=right, z=forward), filtro altura sobre Y
  [B4] vy y wz sign fix (right→left, CW→CCW)
  [B5] yaw inicial: atan2 del vector forward en ENU (no euler('xyz')[2])
  [B6] Integración posición con vel_global: no requiere rotar por yaw
  [B7] dt=0 en primera iteración (return early)
  [B8] time.sleep(100) → create_timer(5.0, ...)
  [B9] sendTransform doble → lista única
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
# Matrices de conversión (constantes globales)
# ─────────────────────────────────────────────────────────────────────────────

# NED mundo → ENU mundo
# (x_E, y_N, z_U) = R_NED2ENU @ (x_N, y_E, z_D)
R_NED2ENU = np.array([
    [0.,  1.,  0.],
    [1.,  0.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)

# NED body (x=fwd, y=right, z=down) → ROS body (x=fwd, y=left, z=up)
# Equivalente a una rotación de 180° alrededor del eje x (forward)
R_BODY_NED2ROS = np.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.]
], dtype=np.float64)


def ned_pose_to_enu(p: np.ndarray) -> np.ndarray:
    """
    Convierte pose [x,y,z, qx,qy,qz,qw] de NED a ENU.

    La orientación resultante tiene:
      - columna 0 (body x = forward) expresada en ENU mundo
      - se usa R_NED2ENU @ R_body_ned @ R_BODY_NED2ROS para obtener la
        rotación body_ROS→ENU_mundo, que es la convención base_link ROS.

    Para AMCL 2D, el yaw ENU correcto se extrae con:
      forward_enu = R[:, 0]
      yaw = atan2(forward_enu[1], forward_enu[0])
    """
    pos_enu = R_NED2ENU @ p[:3]

    R_body_ned = R.from_quat(p[3:]).as_matrix()
    # R completa: body_ROS (x=fwd, y=left, z=up) → ENU mundo
    R_body_enu = R_NED2ENU @ R_body_ned @ R_BODY_NED2ROS
    q_enu = R.from_matrix(R_body_enu).as_quat()

    return np.array([*pos_enu, *q_enu], dtype=np.float64)


def extract_yaw_enu(p_enu: np.ndarray) -> float:
    """
    Extrae el yaw ENU (CCW desde East) de una pose ya convertida a ENU.

    B5 FIX: euler('xyz')[2] es INCORRECTO cuando el eje z del body apunta
    hacia abajo (NED body convention). El yaw ENU se obtiene proyectando
    el vector forward (columna 0 de la matriz de rotación) en el plano XY
    de ENU y calculando atan2(y_ENU, x_ENU).
    """
    R_body = R.from_quat(p_enu[3:]).as_matrix()
    forward_enu = R_body[:, 0]   # eje x del body (forward) expresado en ENU
    return float(np.arctan2(forward_enu[1], forward_enu[0]))


class TartanGroundNode(Node):

    def __init__(self):
        super().__init__('tartanground_node')

    # ── Parámetros ────────────────────────────────────────────
        self.declare_parameter('dataset_path',
            '/home/israelnunez/tartanairpy/House/Data_omni/P0000')
        self.declare_parameter('topic_rgb_image',   '/camera/rgb')
        self.declare_parameter('topic_depth_image', '/camera/depth')
        self.declare_parameter('topic_localization', '/amcl_pose')
        self.declare_parameter('topic_camera_info', '/camera/camera_info')
        self.declare_parameter('camera_localization', '/debug/default_camera')
        self.declare_parameter('motion_data',       '/imu/motion_data')
        self.declare_parameter('map_frame_id',    'map')
        self.declare_parameter('robot_frame_id',  'base_link')
        self.declare_parameter('camera_frame_id', 'camera')
        self.declare_parameter('lidar_frame_id',  'lidar')
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

    # ── Rutas del dataset ─────────────────────────────────────
        self.rgb_path   = os.path.join(self.dataset_path, 'image_lcam_front')
        self.depth_path = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path = os.path.join(self.dataset_path, 'lidar')
        self.rgb_files   = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))
        self.lidar_files = sorted(os.listdir(self.lidar_path))

        self._load_times()
        self._load_camera_poses()
        self._load_robot_pose()
        self._load_velocities()

        self.bridge = CvBridge()

        # ── Publishers ─────────────────────────────────────────────
        self.rgb_pub        = self.create_publisher(Image, self.topic_rgb, 10)
        self.depth_pub      = self.create_publisher(Image, self.topic_depth, 10)
        self.pose_pub       = self.create_publisher(PoseWithCovarianceStamped, self.cam_topic_pose, 10)
        self.scan_pub       = self.create_publisher(LaserScan, '/scan', 10)
        self.robot_pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.topic_robot_pose, 10)
        self.clock_pub      = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub       = self.create_publisher(Odometry, '/odom', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.info_pub = self.create_publisher(CameraInfo, self.topic_info, qos_latched)
        self.map_sub  = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_latched)

        # ── Estado de odometría ───────────────────────────────────
        self.x_odom   = 0.0
        self.y_odom   = 0.0
        self.z_odom   = 0.0
        self.last_time = None

        # B5 FIX: extraer yaw ENU desde el vector forward, no desde euler('xyz')
        p0_enu = ned_pose_to_enu(self.robot_poses[0])
        self.yaw_odom = extract_yaw_enu(p0_enu)
        self.get_logger().info(f"Yaw inicial ENU: {np.degrees(self.yaw_odom):.1f}°")

        # Locks para acceso thread-safe

        self._cam_lock = threading.Lock()
        self._imu_lock = threading.Lock()

        self.mapa_recibido = False

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_pub  = StaticTransformBroadcaster(self)

        self.get_logger().info("Nodo listo. Esperando mapa en /map...")

    # =========================================================================
    # Arranque diferido
    # =========================================================================


    # =========================================================================
    # Carga de datos
    # =========================================================================

    def _load(self, npy, txt):
        for p in (npy, txt):
            if p and os.path.exists(p):
                return np.load(p) if p.endswith('.npy') else np.loadtxt(p)
        raise FileNotFoundError(f"No se encontró {npy} ni {txt}")

    def _load_times(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.cam_times = self._load(f'{base}/cam_time.npy', f'{base}/cam_time.txt')
        self.imu_times = self._load(f'{base}/imu_time.npy', f'{base}/imu_time.txt')

    def _load_camera_poses(self):
        meta = os.path.join(self.dataset_path, 'metadata', 'pose_lcam_front.txt')
        plain = os.path.join(self.dataset_path, 'pose_lcam_front.txt')
        path = meta if os.path.exists(meta) else plain
        self.cam_poses = np.loadtxt(path)

    def _load_robot_pose(self):
        base = os.path.join(self.dataset_path, 'imu')
        pos = self._load(f'{base}/pos_global.npy', f'{base}/pos_global.txt')
        ori = self._load(f'{base}/ori_global.npy', f'{base}/ori_global.txt')

        roll, pitch, yaw = ori[:, 0], ori[:, 1], ori[:, 2]
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        # Almacenamos en NED; se convierte a ENU al publicar
        self.robot_poses = np.hstack((pos, np.stack((qx, qy, qz, qw), axis=1)))

    def _load_velocities(self):
        """
        Carga vel_body  [vx_fwd, vy_right, vz_down]  en NED cuerpo
             vel_global [vx_north, vy_east, vz_down]  en NED mundo
             gyro       [wx, wy, wz]                  en NED cuerpo
        """
        base = os.path.join(self.dataset_path, 'imu')

        self.vel_body = self._load(f'{base}/vel_body.npy', f'{base}/vel_body.txt')
        self.gyro     = self._load(f'{base}/gyro.npy',     f'{base}/gyro.txt')

        # vel_global puede no estar disponible en todos los splits;
        # si no existe, lo calculamos desde vel_body y la pose.
        try:
            self.vel_global = self._load(f'{base}/vel_global.npy', f'{base}/vel_global.txt')
            self.get_logger().info("vel_global cargado desde archivo.")
        except FileNotFoundError:
            self.get_logger().warn(
                "vel_global no encontrado. Derivando de vel_body + poses.")
            self.vel_global = self._derive_vel_global()

    def _derive_vel_global(self) -> np.ndarray:
        """
        Si vel_global no existe, lo calcula como:
            v_global[i] = R_body_ned[i] @ vel_body[i]

        Esto es equivalente a diferenciar finitas de pos_global,
        pero usa los datos que ya tenemos.
        """
        n = len(self.vel_body)
        vg = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            R_body = R.from_quat(self.robot_poses[i, 3:]).as_matrix()
            vg[i] = R_body @ self.vel_body[i, :3]
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
        msg.k = list(map(float, [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]))
        msg.header.stamp    = self._to_ros_time(0.0)
        msg.header.frame_id = self.cam_frame
        self.info_pub.publish(msg)

    def _publish_rgb(self, path, t):
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"No se pudo leer RGB: {path}")
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.rgb_pub.publish(msg)

    def _publish_depth(self, path, t):
        """
        B2 FIX: TartanAir depth = PNG RGBA donde 4 bytes/píxel = float32 LE.
        Usar view('<f4') para reinterpretar bits correctamente.
        """
        depth_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            self.get_logger().warn(f"No se pudo leer depth: {path}")
            return
        depth = depth_rgba.view(np.float32).reshape(depth_rgba.shape[:2])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.cam_frame
        self.depth_pub.publish(msg)

    def _publish_cam_pose(self, idx, t):
        p_enu = ned_pose_to_enu(self.cam_poses[idx])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x    = p_enu[0]
        msg.pose.pose.position.y    = p_enu[1]
        msg.pose.pose.position.z    = p_enu[2]
        msg.pose.pose.orientation.x = p_enu[3]
        msg.pose.pose.orientation.y = p_enu[4]
        msg.pose.pose.orientation.z = p_enu[5]
        msg.pose.pose.orientation.w = p_enu[6]
        self.pose_pub.publish(msg)

    def _publish_robot_pose(self, idx, t):
        p_enu = ned_pose_to_enu(self.robot_poses[idx])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self._to_ros_time(t)
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x    = p_enu[0]
        msg.pose.pose.position.y    = p_enu[1]
        msg.pose.pose.position.z    = p_enu[2]
        msg.pose.pose.orientation.x = p_enu[3]
        msg.pose.pose.orientation.y = p_enu[4]
        msg.pose.pose.orientation.z = p_enu[5]
        msg.pose.pose.orientation.w = p_enu[6]
        self.robot_pose_pub.publish(msg)

    def _publish_lidar(self, i, t):
        """
        LiDAR TartanGround (PLY en frame óptico) → LaserScan en ROS (base_link)

        Frame óptico:
            x = right
            y = down
            z = forward

        Frame ROS (base_link):
            x = forward
            y = left
            z = up

        Transformación:
            x_ros =  z
            y_ros = -x
        """

        # ── Cargar nube de puntos ──
        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[i])
        )
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            self.get_logger().warn(f"Scan vacío en índice {i}")
            return

        # Frame óptico
        xs = points[:, 0]  # right
        ys = points[:, 1]  # down
        zs = points[:, 2]  # forward

        # ── Filtro de altura (plano horizontal) ──
        mask = (ys > -0.2) & (ys < 0.2)
        xs = xs[mask]
        zs = zs[mask]

        # ── Conversión a frame ROS (base_link) ──
        x_ros = zs
        y_ros = -xs

        # ── Configuración del scan ──
        angle_min = -np.pi
        angle_max =  np.pi
        angle_increment = np.deg2rad(0.5)
        num_beams = int((angle_max - angle_min) / angle_increment)

        ranges = np.full(num_beams, np.inf)

        # ── Cálculo de ángulos ──
        angles = np.arctan2(y_ros, x_ros)
        distances = np.hypot(x_ros, y_ros)

        indices = ((angles - angle_min) / angle_increment).astype(int)

        # ── Rellenar rayos ──
        for idx_b, dist in zip(indices, distances):
            if 0 <= idx_b < num_beams:
                if dist < ranges[idx_b]:
                    ranges[idx_b] = dist

        # ── Mensaje ROS ──
        scan = LaserScan()
        scan.header.stamp = self._to_ros_time(t)
        # scan.header.frame_id = self.robot_frame  # base_link
        scan.header.frame_id = self.lidar_frame  # base_link

        scan.angle_min = angle_min
        scan.angle_max = angle_max
        scan.angle_increment = angle_increment

        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.rate

        scan.range_min = 0.1
        scan.range_max = 50.0

        # IMPORTANTE: dejar inf (mejor para AMCL)
        scan.ranges = ranges.tolist()

        self.scan_pub.publish(scan)

    def _compute_odometry(self, idx: int, t: float):
        """
        B6 FIX — Integración de posición con vel_global (sin rotación por yaw).

        vel_global[i] = [v_North, v_East, v_Down]  en NED mundo.
        Convertir a ENU: dx_ENU = vel_global[1]*dt, dy_ENU = vel_global[0]*dt.

        Esto elimina la dependencia de yaw_odom en la integración de posición,
        lo que evita la acumulación de error cuando el yaw tiene drift.

        La integración de yaw sigue usando el giroscopio:
          wz_ros = -gyro[2]  (CCW desde Up = -CW desde Down)

        B5 FIX — yaw_odom inicializado con extract_yaw_enu() en __init__.
        B7 FIX — primera iteración retorna sin integrar.
        """
        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t
        if dt <= 0.0:
            return

        # ── Posición: vel_global en NED mundo → ENU (sin rotación por yaw) ──
        vg = self.vel_global[idx]
        self.x_odom += float(vg[1]) * dt  # East
        self.y_odom += float(vg[0]) * dt  # North
        self.z_odom += -float(vg[2]) * dt

        # ── Yaw: giroscopio en NED cuerpo → ROS (negar wz) ──────────────────
        wz_ned = float(self.gyro[idx, 2])   # CW positivo (eje z=down)
        wz_ros = -wz_ned                    # CCW positivo (eje z=up)
        self.yaw_odom += wz_ros * dt
        self.yaw_odom  = np.arctan2(np.sin(self.yaw_odom), np.cos(self.yaw_odom))

        # ── Twist para Odometry message (en frame ROS body) ──────────────────
        vb = self.vel_body[idx]
        vx_ros =  float(vb[0])    # forward, sin cambio
        vy_ros = -float(vb[1])    # left = -right

        q = R.from_euler('z', self.yaw_odom).as_quat()
        self._publish_odometry(t, vx_ros, vy_ros, wz_ros, q)

    def _publish_odometry(self, t, vx, vy, wz, q):
        stamp = self._to_ros_time(t)

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
        odom.pose.covariance[0]  = 0.05
        odom.pose.covariance[7]  = 0.05
        odom.pose.covariance[35] = 0.1
        odom.twist.covariance[0]  = 0.1
        odom.twist.covariance[7]  = 0.1
        odom.twist.covariance[35] = 0.2
        self.odom_pub.publish(odom)

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

    def _publish_initial_state(self):
        """
        Publica la pose inicial en ENU y los TF estáticos.

        B1 FIX: ned_pose_to_enu aplica R_NED2ENU Y R_BODY_NED2ROS.
        B9 FIX: ambos TF en una sola llamada sendTransform([...]).
        """
        p_ned = self.robot_poses[0]
        p_enu = ned_pose_to_enu(p_ned)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self._to_ros_time(self.imu_times[0])
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x    = float(p_enu[0])
        msg.pose.pose.position.y    = float(p_enu[1])
        msg.pose.pose.position.z    = float(p_enu[2])
        msg.pose.pose.orientation.x = float(p_enu[3])
        msg.pose.pose.orientation.y = float(p_enu[4])
        msg.pose.pose.orientation.z = float(p_enu[5])
        msg.pose.pose.orientation.w = float(p_enu[6])
        cov = [0.0] * 36
        cov[0] = 0.001; cov[7] = 0.001; cov[35] = 0.001
        msg.pose.covariance = cov
        self.initial_pose_pub.publish(msg)

        # TF estáticos en ENU
        p_cam_enu   = ned_pose_to_enu(self.cam_poses[0])
        T_cam       = self._pose_to_matrix(p_cam_enu)
        T_robot     = self._pose_to_matrix(p_enu)
        T_base_cam  = np.linalg.inv(T_robot) @ T_cam

        tf_base_cam  = self.matrix_to_transform(
            T_base_cam, self.robot_frame, self.cam_frame, self.imu_times[0])
        tf_cam_lidar = self.matrix_to_transform(
            np.eye(4), self.cam_frame, self.lidar_frame, self.imu_times[0])

        self.tf_static_pub.sendTransform([tf_base_cam, tf_cam_lidar])


        self._publish_lidar(0, self.imu_times[0])
        self.get_logger().info("Estado inicial publicado en ENU (con R_BODY_NED2ROS).")

    # =========================================================================
    # Callbacks y loops
    # =========================================================================

    def map_callback(self, msg):

        if self.mapa_recibido:
            return

        self.mapa_recibido = True
        self.get_logger().info("Mapa recibido. Inicializando sistema...")

        self._publish_camera_info()
        self._publish_initial_state()

        time.sleep(3.0)

        self.cam_index = 0
        self.imu_index = 0

        # threading.Thread(target=self._camera_loop, daemon=True).start()
        # threading.Thread(target=self._imu_loop, daemon=True).start()

        self._imu_loop()

        

    def _publish_clock(self, t):
        msg = Clock()
        msg.clock = self._to_ros_time(t)
        self.clock_pub.publish(msg)

    def _camera_loop(self):
        while rclpy.ok():
            with self._cam_lock:
                i = self.cam_index
                if i >= len(self.rgb_files):
                    break
                self.cam_index += 1
            t = self.find_closest_imu_time(self.cam_times[i])
            self._publish_rgb(os.path.join(self.rgb_path,   self.rgb_files[i]),   t)
            self._publish_depth(os.path.join(self.depth_path, self.depth_files[i]), t)
            self._publish_cam_pose(i, t)
            self._publish_lidar(i, t)
            time.sleep(1.0 / self.rate)
        self.get_logger().info("Fin del bucle de cámara.")

    def _imu_loop(self):
     while rclpy.ok():
         with self._imu_lock:
             if self.imu_index >= len(self.imu_times):
                 break

             i = self.imu_index
             self.imu_index += 1

         t_imu = self.imu_times[i]

         # ── Publicar IMU SIEMPRE ─────────────────────────────
         self._publish_clock(t_imu)
         self._publish_robot_pose(i, t_imu)
         self._compute_odometry(i, t_imu)

         # ── Sincronización con cámara ────────────────────────
         while (self.cam_index < len(self.cam_times) and
                self.cam_times[self.cam_index] <= t_imu):

             j = self.cam_index
             self.cam_index += 1

             t_cam = self.cam_times[j]

             # ✔ Paths correctos
             rgb_path   = os.path.join(self.rgb_path,   self.rgb_files[j])
             depth_path = os.path.join(self.depth_path, self.depth_files[j])

             # ✔ Publicación consistente
             self._publish_rgb(rgb_path, t_cam)
             self._publish_depth(depth_path, t_cam)
             self._publish_cam_pose(j, t_cam)
             self._publish_lidar(j, t_cam)

         time.sleep(1.0 / (5* (self.rate)))

     self.get_logger().info("Fin del bucle de IMU.")


def main():
    rclpy.init()
    node = TartanGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()