#!/usr/bin/env python3

import os
import math

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseArray, PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Path, Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from builtin_interfaces.msg import Time
from std_msgs.msg import Header, Empty
import time
from sensor_msgs.msg import PointCloud2, PointField, Image, LaserScan, Imu  
import sensor_msgs_py.point_cloud2 as pc2

from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Utilidades de conversión NED ↔ ENU
# ---------------------------------------------------------------------------
def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return (
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
                aw*bw - ax*bx - ay*by - az*bz
            )

def ned_to_enu_position(x_ned: float, y_ned: float, z_ned: float):
    """Convierte posición de NED a ENU."""
    x_enu =  y_ned   # Este  ← Y_ned (Este)
    y_enu =  x_ned   # Norte ← X_ned (Norte)
    z_enu = -z_ned   # Arriba ← -Z_ned (Abajo)
    return x_enu, y_enu, z_enu


def ned_to_enu_quaternion(qx, qy, qz, qw):
    """
    Conversión correcta NED → ENU (ROS estándar).
    Basado en cambio de base, no rotación física.
    """

    # Swap X <-> Y y flip Z
    qx_enu = qy
    qy_enu = qx
    qz_enu = -qz
    qw_enu = qw


    # --- Rotación +90° en Z (LOCAL frame) ---
    angle = math.pi / 2
    q_rot = (
        0.0,
        0.0,
        math.sin(angle / 2),
        math.cos(angle / 2)
    )

    # --- Multiplicación: q_final = q_enu ⊗ q_rot ---
    qx_enu, qy_enu, qz_enu, qw_enu = qmul((qx_enu,qy_enu,qz_enu,qw_enu), q_rot)


    return qx_enu, qy_enu, qz_enu, qw_enu


def load_cam_pose(filepath: str) -> list:
    """
    Carga un archivo de poses del dataset TartanGround.

    Formato: una pose por línea → tx ty tz qx qy qz qw
    Las líneas vacías o que empiezan por '#' se ignoran.

    Returns:
        Lista de tuplas (tx, ty, tz, qx, qy, qz, qw) en NED.
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) < 7:
                continue
            tx, ty, tz = float(vals[0]), float(vals[1]), float(vals[2])
            qx, qy, qz, qw = float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6])
            poses.append([tx, ty, tz, qx, qy, qz, qw])
    return poses


def RotZ(v):
    Rz_90 = np.array([
    [0, 1, 0],
    [-1,  0, 0],
    [0,  0, 1]
    ])

    return(Rz_90 @ v)

# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------

class TartanGroundTFPublisher(Node):

    def __init__(self):

        super().__init__('tartanground_tf_publisher')

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('dataset_path', '/home/israelnunez/tartanairpy/Office/Data_omni/P0000')
        self.declare_parameter('publish_rate', 10.0)      # Hz
        self.declare_parameter('loop', False)             # repetir al terminar
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('robot_frame_GT', 'base_link_GT')
        self.declare_parameter('camera_frame', 'camera')
        self.declare_parameter('camera_frame_GT', 'camera_GT')
        self.declare_parameter('camera_file', 'pose_lcam_front.txt')
        self.declare_parameter('lidar_frame','lidar')
        self.declare_parameter('lidar_frame_GT','lidar_GT')
        self.declare_parameter('robot_file', 'pose_body.txt')

        self.dataset_path       = self.get_parameter('dataset_path').value
        self.rate_hz            = self.get_parameter('publish_rate').value
        self.loop               = self.get_parameter('loop').value
        self.world_frame        = self.get_parameter('world_frame').value

        self.robot_frame        = self.get_parameter('robot_frame').value
        self.camera_frame       = self.get_parameter('camera_frame').value

        self.robot_frame_GT     = self.get_parameter('robot_frame_GT').value
        self.camera_frame_GT    = self.get_parameter('camera_frame_GT').value

        cam_file                = self.get_parameter('camera_file').value
        robot_file              = self.get_parameter('robot_file').value
        self.lidar_frame        = self.get_parameter('lidar_frame').value

        self.rgb_path           = os.path.join(self.dataset_path, 'image_lcam_front')
        self.depth_path         = os.path.join(self.dataset_path, 'depth_lcam_front')
        self.lidar_path         = os.path.join(self.dataset_path, 'lidar')

       
        self.rgb_files          = sorted(os.listdir(self.rgb_path))
        self.depth_files        = sorted(os.listdir(self.depth_path))
        self.lidar_files        = sorted(os.listdir(self.lidar_path))

        # ────────────────────────────────────────────────────────

        if not self.dataset_path:
            self.get_logger().fatal('Parámetro "dataset_path" vacío. Abortando.')
            raise SystemExit(1)

        self.bridge = CvBridge()

        # ── Cargar poses ────────────────────────────────────────────────────

        # Poses Cámara

        cam_path = os.path.join(self.dataset_path, 'metadata', cam_file)

        if not os.path.isfile(cam_path):
            cam_path = os.path.join(self.dataset_path, cam_file)

        if not os.path.isfile(cam_path):
            self.get_logger().fatal(f'No se encontró el archivo de poses de cámara: {cam_path}')
            raise SystemExit(1)

        self.cam_poses_ned = load_cam_pose(cam_path)
        self.get_logger().info(
            f'Cargadas {len(self.cam_poses_ned)} poses de cámara desde:\n  {cam_path}')


        # Poses Robot

        robot_path = os.path.join(self.dataset_path, 'imu','pos_global.txt')

        if os.path.isfile(robot_path):
            self._load_robot_poses()
            self.get_logger().info(
                f'Cargadas {len(self.robot_poses_ned)} poses de robot desde:\n  {robot_path}')
        else:
            self.get_logger().warn(
                f'Archivo de pose del robot no encontrado: {robot_path}\n'
                '→ Se usará la pose de cámara también para el robot.')
            self.robot_poses_ned = self.cam_poses_ned

        self.total_frames    = len(self.robot_poses_ned)
        self.current_frame   = 0

        self._load_times()
        self._load_motion_data()

        self.get_logger().info(
            f'Total de frames a reproducir: {self.total_frames}  |  '
            f'Frecuencia: {10*self.rate_hz} Hz  |  Loop: {self.loop}')

        # ── Odometría - Dead Reckoning ─────────────────────────────────────────────────

        self.last_time = None

        # ── Publicadores TF ─────────────────────────────────────────────────
        self.tf_broadcaster        = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ── Publicadores de topics ───────────────────────────────────────────
        self.robot_pose_pub  = self.create_publisher(PoseStamped, 'robot_pose', 10)
        self.camera_pose_pub = self.create_publisher(PoseStamped, 'camera_pose', 10)

        self.robot_pose_pub_GT  = self.create_publisher(PoseStamped, 'robot_pose_GT', 10)

        self.pub              = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.rgb_pub          = self.create_publisher(Image, '/camera/rgb', 10)
        self.depth_pub        = self.create_publisher(Image, '/camera/depth', 10)
        self.clock_pub        = self.create_publisher(Clock, '/clock', 10)
     
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)

        self.map_published    = self.create_subscription(Empty,'map_recv',self.map_received_cb,10)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, 
            depth=1)

        self.odom_pub        = self.create_publisher(Odometry, '/odom', 10)

        self.twist_pub  = self.create_publisher(TwistWithCovarianceStamped,'/twist',10)

        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # En __init__, añadir el publicador:
        self.set_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/set_pose', 10)

        # ── TF estático: map → odom (identidad) ─────────────────────────────

        self._publish_initial_pose()        # Debug
 
        self.closest_times()
        self.closest_map = {imu: cam for imu, cam in self.closest_pairs}

        self._publish_initial_state()   # Base_Link -> Camera


        p0_robot_enu = self._pose_ned_to_pose_enu(self.robot_poses_ned[0])
        p0_cam_enu  = self._pose_ned_to_pose_enu(self.cam_poses_ned[0])

        T_cam      = self.pose_to_matrix(p0_cam_enu)
        T_robot    = self.pose_to_matrix(p0_robot_enu)

        T_base_cam = np.linalg.inv(T_robot) @ T_cam


        tf_base_cam_GT = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0), self.robot_frame_GT, self.camera_frame_GT)
        tf_base_cam = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0), self.robot_frame, self.camera_frame)

        # Al final de __init__, DESPUÉS de calcular T_base_cam:
        tf_base_cam_GT = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0),
                                            self.robot_frame_GT, self.camera_frame_GT)
        tf_base_cam    = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0),
                                            self.robot_frame,    self.camera_frame)

        tf_lidar_gt    = self._make_identity_tf(self._to_ros_time(0.0),
                                                 self.camera_frame_GT, 'lidar_GT')
        tf_lidar       = self._make_identity_tf(self._to_ros_time(0.0),
                                         self.camera_frame,    self.lidar_frame)

        # Una sola llamada con todo
        self.static_tf_broadcaster.sendTransform(
            [tf_base_cam_GT, tf_base_cam, tf_lidar_gt, tf_lidar]
        )

        # self._publish_map_to_odom_at_p0()

        self.get_logger().info('✓ Nodo TartanGround TF Publisher iniciado.')
        self.get_logger().info(f'  Marcos TF: NED → {self.robot_frame} → {self.camera_frame}')
        self.get_logger().info('Esperando mapa')
 
        
    # -----------------------------------------------------------------------

    def _publish_map_to_odom_at_p0(self):
        p0 = self._pose_ned_to_pose_enu(self.robot_poses_ned[0])

        # map → odom en p0
        ts_map_odom = TransformStamped()
        ts_map_odom.header.stamp    = self.get_clock().now().to_msg()
        ts_map_odom.header.frame_id = 'map'
        ts_map_odom.child_frame_id  = 'odom'
        ts_map_odom.transform.translation.x = p0[0]
        ts_map_odom.transform.translation.y = p0[1]
        ts_map_odom.transform.translation.z = 0.0
        ts_map_odom.transform.rotation.x = p0[3]
        ts_map_odom.transform.rotation.y = p0[4]
        ts_map_odom.transform.rotation.z = p0[5]
        ts_map_odom.transform.rotation.w = p0[6]

        # odom → base_link identidad (placeholder hasta que el EKF arranque)
        ts_odom_bl = TransformStamped()
        ts_odom_bl.header.stamp    = self.get_clock().now().to_msg()
        ts_odom_bl.header.frame_id = 'odom'
        ts_odom_bl.child_frame_id  = 'base_link'
        ts_odom_bl.transform.rotation.w = 1.0

        # Una sola llamada para ambos
        self.static_tf_broadcaster.sendTransform([ts_map_odom, ts_odom_bl])
        self.get_logger().info(
            f'✓ map→odom en p0: x={p0[0]:.3f}  y={p0[1]:.3f}  '
            f'| odom→base_link identidad (placeholder EKF)')


    def map_received_cb(self,msg):

        self._publish_initial_pose()    

        self.get_logger().info('Mapa recibido: Inciando repoducción')

        self.timer = self.create_timer(1.0 / (10*self.rate_hz), self._timer_callback)
        return


    def _publish_initial_pose(self):

        p0_robot_enu = self._pose_ned_to_pose_enu(self.robot_poses_ned[0])

        msg = PoseWithCovarianceStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"   # 🔥 SIEMPRE "map"

        # ── Posición ──
        msg.pose.pose.position.x = p0_robot_enu[0]
        msg.pose.pose.position.y = p0_robot_enu[1]
        msg.pose.pose.position.z = p0_robot_enu[2]

        # ── Orientación (yaw → quaternion) ──

        msg.pose.pose.orientation.x = p0_robot_enu[3]
        msg.pose.pose.orientation.y = p0_robot_enu[4]
        msg.pose.pose.orientation.z = p0_robot_enu[5]
        msg.pose.pose.orientation.w = p0_robot_enu[6]

        # ── Covarianza (importante) ──
        msg.pose.covariance = [
           0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.068
        ]       

         # Al EKF (resetea estado interno)
        self.set_pose_pub.publish(msg)

    def _publish_initial_state(self):

        self.get_logger().info(
            f"Publicando pose inicial")

        tstamp = self.get_clock().now().to_msg()

        # Transformación Base - Cámara Ground Truth t = 0 

        p0_robot_enu = self._pose_ned_to_pose_enu(self.robot_poses_ned[0])

        p0_cam_enu  = self._pose_ned_to_pose_enu(self.cam_poses_ned[0])

        T_cam      = self.pose_to_matrix(p0_cam_enu)
        T_robot    = self.pose_to_matrix(p0_robot_enu)

        T_base_cam = np.linalg.inv(T_robot) @ T_cam


        tf_base_cam_GT = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0), self.robot_frame_GT, self.camera_frame_GT)
        tf_base_cam = self.matrix_to_tf(T_base_cam, self._to_ros_time(0.0), self.robot_frame, self.camera_frame)

        # self.static_tf_broadcaster.sendTransform([tf_base_cam_GT,tf_base_cam])

        # ----------------------------------------------

        # Pose Inicial del robot

        tf_init_GT = self._pose_ned_to_tf_enu(self.robot_poses_ned[0],self.world_frame,self.robot_frame_GT)
        tf_init_GT.header.stamp = tstamp
        

        # tf_ref_body = self._pose_ned_to_tf_enu(self.robot_poses_ned[0],self.world_frame,'Ref_temporal')
        # tf_ref_body.header.stamp = tstamp
        self.tf_broadcaster.sendTransform([tf_init_GT])

        #PUBLICAR TF REAL
        

        # ----------------------------------------------

        self.last_time = self.imu_times[0]

        self.current_frame += 1

        self.get_logger().info(
            f"Pose inicial publicada")

    def compute_odometry(self,tstamp,frame):

        # =========================
        # ODOMETRY
        # =========================
        odom = Odometry()
        odom.header.stamp = tstamp
        odom.header.frame_id = "odom_GT"
        odom.child_frame_id = 'base_link_ODOM'  # base_link

        # Posición
        odom.pose.pose.position.x = float(self.position_odom[0])
        odom.pose.pose.position.y = float(self.position_odom[1])
        odom.pose.pose.position.z = float(self.position_odom[2])

        # Orientación
        qx, qy, qz, qw = self.orientation_odom
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        # Velocidad (muy importante para EKF)
        odom.twist.twist.linear.x = float(self.velocity[0])
        odom.twist.twist.linear.y = float(self.velocity[1])
        odom.twist.twist.linear.z = float(self.velocity[2])

        odom.twist.twist.angular.x = float(gyro_enu[0])
        odom.twist.twist.angular.y = float(gyro_enu[1])
        odom.twist.twist.angular.z = float(gyro_enu[2])

        self.odom_pub.publish(odom)

        # =========================
        # IMU
        # =========================
        imu = Imu()
        imu.header.stamp = tstamp
        imu.header.frame_id = self.robot_frame  # base_link

        # Orientación (puedes quitarla si quieres que EKF la estime)
        imu.orientation.x = float(qx)
        imu.orientation.y = float(qy)
        imu.orientation.z = float(qz)
        imu.orientation.w = float(qw)

        # Velocidad angular (gyro)
        imu.angular_velocity.x = float(gyro_enu[0])
        imu.angular_velocity.y = float(gyro_enu[1])
        imu.angular_velocity.z = float(gyro_enu[2])

        # Aceleración
        imu.linear_acceleration.x = float(accel_enu[0])
        imu.linear_acceleration.y = float(accel_enu[1])
        imu.linear_acceleration.z = float(accel_enu[2])

        self.imu_pub.publish(imu)

    # -----------------------------------------------------------------------

    def _pose_ned_to_pose_enu(self, pose_ned: tuple) -> tuple:

        tx_n, ty_n, tz_n, qx_n, qy_n, qz_n, qw_n = pose_ned

        # Conversión de posición NED → ENU
        x_e, y_e, z_e = ned_to_enu_position(tx_n, ty_n, tz_n)

        # Conversión de orientación NED → ENU
        qx_e, qy_e, qz_e, qw_e = ned_to_enu_quaternion(qx_n, qy_n, qz_n, qw_n)
 
        return (x_e, y_e, z_e,qx_e, qy_e, qz_e, qw_e)


    def _pose_ned_to_tf_enu(self, pose_ned: tuple, parent: str, child: str) -> TransformStamped:
        """
        Convierte una pose NED (tx,ty,tz,qx,qy,qz,qw) en un TransformStamped ENU.
        """
        tx_n, ty_n, tz_n, qx_n, qy_n, qz_n, qw_n = pose_ned

        # Conversión de posición NED → ENU
        x_e, y_e, z_e = ned_to_enu_position(tx_n, ty_n, tz_n)

        # Conversión de orientación NED → ENU
        qx_e, qy_e, qz_e, qw_e = ned_to_enu_quaternion(qx_n, qy_n, qz_n, qw_n)

        ts = TransformStamped()
        ts.header.stamp    = self.get_clock().now().to_msg()
        ts.header.frame_id = parent
        ts.child_frame_id  = child

        ts.transform.translation.x = x_e
        ts.transform.translation.y = y_e
        ts.transform.translation.z = z_e

        ts.transform.rotation.x = qx_e
        ts.transform.rotation.y = qy_e
        ts.transform.rotation.z = qz_e
        ts.transform.rotation.w = qw_e

        return ts

    # -----------------------------------------------------------------------

    def _make_pose_stamped(self, tf_msg: TransformStamped) -> PoseStamped:
        """Convierte un TransformStamped a PoseStamped para visualización."""
        ps = PoseStamped()
        ps.header = tf_msg.header
        ps.pose.position.x = tf_msg.transform.translation.x
        ps.pose.position.y = tf_msg.transform.translation.y
        ps.pose.position.z = tf_msg.transform.translation.z
        ps.pose.orientation.x = tf_msg.transform.rotation.x
        ps.pose.orientation.y = tf_msg.transform.rotation.y
        ps.pose.orientation.z = tf_msg.transform.rotation.z
        ps.pose.orientation.w = tf_msg.transform.rotation.w
        return ps

    # -----------------------------------------------------------------------

    def closest_times(self):    

        # Enumeramos para NO perder índices originales
        imu_sorted = sorted(enumerate(self.imu_times), key=lambda x: x[1])
        cam_sorted = sorted(enumerate(self.cam_times), key=lambda x: x[1])

        i = j = 0

        self.closest_pairs = []  # lista de tuplas (idx_imu, idx_cam)

        while i < len(imu_sorted) and j < len(cam_sorted):

            # Avanzar en imu mientras mejore la distancia
            while (i + 1 < len(imu_sorted) and 
               abs(imu_sorted[i + 1][1] - cam_sorted[j][1]) <= 
               abs(imu_sorted[i][1] - cam_sorted[j][1])):
                i += 1

            idx_imu = imu_sorted[i][0]   # índice original en imu_times
            idx_cam = cam_sorted[j][0]   # índice original en cam_times

            self.closest_pairs.append((idx_imu, idx_cam))

            j += 1
            
    def _load_motion_data(self):

        base = os.path.join(self.dataset_path, 'imu')
        self.vel_body = self._load(
            f'{base}/vel_body.npy', f'{base}/vel_body.txt')
        self.gyro     = self._load(
            f'{base}/gyro.npy',     f'{base}/gyro.txt')
        self.acc    = self._load(
            f'{base}/acc.npy', f'{base}/acc.txt')
        self.acc_nograv    = self._load(
            f'{base}/acc_nograv.npy', f'{base}/acc_nograv.txt')

        self.get_logger().info(
            f"Cargados {len(self.vel_body)} pasos de vel_body + gyro "
            f"(GT derivado, usado como odometría).")

    def _to_ros_time(self, t: float) -> Time:
        return Time(sec=int(t), nanosec=int((t % 1) * 1e9))


    def _load(self, npy: str, txt: str) -> np.ndarray:
           for p in (npy, txt):
               if p and os.path.exists(p):
                   return np.load(p) if p.endswith('.npy') else np.loadtxt(p)
           raise FileNotFoundError(f"No se encontró: {npy} ni {txt}")

           
    def _make_identity_tf(self, stamp, parent: str, child: str) -> TransformStamped:
        ts = TransformStamped()
        ts.header.stamp    = stamp
        ts.header.frame_id = parent
        ts.child_frame_id  = child
        ts.transform.rotation.w = 1.0
        return ts


    def pose_to_matrix(self,pose:tuple):
        """
        Convierte una pose (x, y, z, qx, qy, qz, qw) en una matriz 4x4.

        :param pose: tupla/lista de 7 elementos
        :return: np.array 4x4
        """

        if len(pose) != 7:
            raise ValueError("La pose debe tener 7 elementos (x, y, z, qx, qy, qz, qw)")

        x, y, z, qx, qy, qz, qw = pose

        # Matriz de rotación a partir del cuaternión
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        # Matriz homogénea
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        return T

    # def matrix_to_tf(self, matrix, stamp, parent_frame="world", child_frame="base_link"):
    #     """
    #     Convierte una matriz homogénea 4x4 en un mensaje TransformStamped.

    #     :param matrix: np.array de 4x4
    #     :param parent_frame: frame padre
    #     :param child_frame: frame hijo
    #     :param node: nodo ROS2 (para timestamp)
    #     :return: TransformStamped
    #     """

    #     self.get_logger().info(f'{parent_frame} -> {child_frame}')

    #     if matrix.shape != (4, 4):
    #         raise ValueError("La matriz debe ser 4x4")

    #     t = TransformStamped()

    #     # Timestamp
    #     t.header.stamp = stamp
    #     t.header.frame_id = parent_frame
    #     t.child_frame_id = child_frame

    #     # Traslación
    #     t.transform.translation.x = float(matrix[0, 3])
    #     t.transform.translation.y = float(matrix[1, 3])
    #     t.transform.translation.z = float(matrix[2, 3])

    #     # Rotación (matriz -> cuaternión)
    #     rot = matrix[:3, :3]
    #     qw = math.sqrt(1.0 + rot[0,0] + rot[1,1] + rot[2,2]) / 2.0
    #     qx = (rot[2,1] - rot[1,2]) / (4.0 * qw)
    #     qy = (rot[0,2] - rot[2,0]) / (4.0 * qw)
    #     qz = (rot[1,0] - rot[0,1]) / (4.0 * qw)

    #     t.transform.rotation.x = float(qx)
    #     t.transform.rotation.y = float(qy)
    #     t.transform.rotation.z = float(qz)
    #     t.transform.rotation.w = float(qw)

    #     return t

    def matrix_to_tf(self, matrix, stamp, parent_frame="world", child_frame="base_link"):
        if matrix.shape != (4, 4):
            raise ValueError("La matriz debe ser 4x4")

        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id  = child_frame

        t.transform.translation.x = float(matrix[0, 3])
        t.transform.translation.y = float(matrix[1, 3])
        t.transform.translation.z = float(matrix[2, 3])

        # ✅ Shepperd robusto — elige el componente mayor para dividir
        rot = matrix[:3, :3]
        trace = rot[0,0] + rot[1,1] + rot[2,2]

        if trace > 0:
            s  = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rot[2,1] - rot[1,2]) * s
            qy = (rot[0,2] - rot[2,0]) * s
            qz = (rot[1,0] - rot[0,1]) * s
        elif rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]:
            s  = 2.0 * math.sqrt(1.0 + rot[0,0] - rot[1,1] - rot[2,2])
            qw = (rot[2,1] - rot[1,2]) / s
            qx = 0.25 * s
            qy = (rot[0,1] + rot[1,0]) / s
            qz = (rot[0,2] + rot[2,0]) / s
        elif rot[1,1] > rot[2,2]:
            s  = 2.0 * math.sqrt(1.0 + rot[1,1] - rot[0,0] - rot[2,2])
            qw = (rot[0,2] - rot[2,0]) / s
            qx = (rot[0,1] + rot[1,0]) / s
            qy = 0.25 * s
            qz = (rot[1,2] + rot[2,1]) / s
        else:
            s  = 2.0 * math.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1])
            qw = (rot[1,0] - rot[0,1]) / s
            qx = (rot[0,2] + rot[2,0]) / s
            qy = (rot[1,2] + rot[2,1]) / s
            qz = 0.25 * s

        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        return t

    def _load_times(self):
        base = os.path.join(self.dataset_path, 'imu')
        self.cam_times = self._load(
            f'{base}/cam_time.npy', f'{base}/cam_time.txt')
        self.imu_times = self._load(
            f'{base}/imu_time.npy', f'{base}/imu_time.txt')

    def _publish_rgb(self, path: str, time_stamp: float):
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"RGB no leído: {path}")
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp    = time_stamp
        msg.header.frame_id = self.camera_frame_GT
        self.rgb_pub.publish(msg)

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

        self.robot_poses_ned = np.hstack((pos, np.stack((qx, qy, qz, qw), axis=1)))


    def _pointcloud_to_laserscan(self, stamp):

        if self.pts_rot.shape[0] == 0:
            return

        # After _publish_pointcloud_ENU rotation: X=forward, Y=left, Z=up
        x = self.pts_rot[:, 0]   # forward  ✅
        y = self.pts_rot[:, 1]   # lateral  ✅
        z = self.pts_rot[:, 2]   # vertical ✅

        # ── Height filter ─────────────────────────────────────────────────────
        mask = (z > -0.3) & (z < 1.0)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            return

        # ── Parameters (radians) ──────────────────────────────────────────────
        # 🔥 FIX 1: LaserScan expects radians — convert everything up front
        ANGLE_MIN       = np.deg2rad(-50.0)
        ANGLE_MAX       = np.deg2rad( 50.0)
        ANGLE_INCREMENT = np.deg2rad(  0.1)

        RANGE_MIN = 0.1
        RANGE_MAX = 50.0

        # ── Polar ──────────────────────────────────────────────────────────────
        # 🔥 FIX 2: standard ROS — angle=0 points forward (+X), CCW positive
        #    atan2(y, x)  NOT  atan2(x, y)
        angles    = np.arctan2(y, x)
        distances = np.hypot(x, y)

        # ── Bin into beams ────────────────────────────────────────────────────
        num_beams = int(round((ANGLE_MAX - ANGLE_MIN) / ANGLE_INCREMENT))

        beam_idx = ((angles - ANGLE_MIN) / ANGLE_INCREMENT).astype(np.int32)

        valid = (
            (beam_idx >= 0) & (beam_idx < num_beams) &
            (distances >= RANGE_MIN) & (distances <= RANGE_MAX)
        )
        beam_idx  = beam_idx[valid]
        distances = distances[valid]

        ranges = np.full(num_beams, np.inf, dtype=np.float32)
        np.minimum.at(ranges, beam_idx, distances)
        # Leave inf for empty beams (REP-117 compliant) — or replace with RANGE_MAX:
        # ranges[np.isinf(ranges)] = RANGE_MAX

        # ── Build message ─────────────────────────────────────────────────────
        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = self.lidar_frame

        scan.angle_min       = ANGLE_MIN
        scan.angle_max       = ANGLE_MIN + (num_beams - 1) * ANGLE_INCREMENT
        scan.angle_increment = ANGLE_INCREMENT

        scan.range_min       = RANGE_MIN
        scan.range_max       = RANGE_MAX

        scan.scan_time       = 1.0 / self.rate_hz
        scan.time_increment  = 0.0

        scan.ranges = ranges.tolist()

        self.scan_pub.publish(scan)

    def _publish_pointcloud(self,idx_cam,time_stamp):
        pcd = o3d.io.read_point_cloud(
        os.path.join(self.lidar_path, self.lidar_files[idx_cam])
        )

        pts = np.asarray(pcd.points)

        if pts.shape[0] == 0:
            self.get_logger().warn("PointCloud vacío")
            return

        header = Header()
        header.stamp = time_stamp
        header.frame_id = 'lidar_GT'

        # Crear mensaje directamente (sin tocar ejes)
        cloud_msg = pc2.create_cloud_xyz32(header, pts)

        self.pub.publish(cloud_msg)

        header.frame_id = self.lidar_frame
        self.pub.publish(cloud_msg)

    def _publish_pointcloud_ENU(self, idx_cam, time_stamp):

        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[idx_cam])
        )

        pts = np.asarray(pcd.points)

        if pts.shape[0] == 0:
            self.get_logger().warn("PointCloud vacío")
            return

        # 🔥 ROTACIÓN -90° EN Z
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        x_rot = y
        y_rot = -x
        z_rot = z

        self.pts_rot = np.stack((x_rot, y_rot, z_rot), axis=1)

        header = Header()
        header.stamp = time_stamp
        header.frame_id = 'lidar_GT'

        cloud_msg = pc2.create_cloud_xyz32(header, self.pts_rot)

        self.pub.publish(cloud_msg)

        header.frame_id = self.lidar_frame
        self.pub.publish(cloud_msg)

    def _publish_lidar(self, cam_idx: int, t: float):
    
        pcd = o3d.io.read_point_cloud(
            os.path.join(self.lidar_path, self.lidar_files[cam_idx]))
        pts = np.asarray(pcd.points)

        if pts.shape[0] == 0:
            self.get_logger().warn(f"PLY vacío en índice {cam_idx}")
            return

        # Frame óptico: x=right, y=down, z=fwd
        xs, ys, zs = pts[:, 1], pts[:, 0], -pts[:, 2]

        # Filtro de altura en frame óptico (y = abajo)

        x_ros = zs
        y_ros = -xs
        z_ros = -ys 

        mask = (ys > -0.3) & (ys < 0.3)
        xs, zs = xs[mask], zs[mask]

        if xs.size == 0:
            self.get_logger().warn(f"Slice vacío en índice {cam_idx}")
            return

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
        scan.header.stamp    = self.get_clock().now().to_msg()
        scan.header.frame_id = self.lidar_frame
        scan.angle_min       = float(ANGLE_MIN)
        scan.angle_max       = float(angle_max_msg)
        scan.angle_increment = float(ANGLE_INCREMENT)
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / self.rate_hz
        scan.range_min       = RANGE_MIN
        scan.range_max       = RANGE_MAX
        scan.ranges          = ranges.tolist()
        scan.intensities     = []
        self.scan_pub.publish(scan)

    def ned_to_enu_vector(self, v):
        vx, vy, vz = v
        return np.array([vy, vx, -vz])

    def _publish_odom(self, tstamp, idx):

        # ── Rotación al frame base_link (Rz -90° sobre ENU) ──────────────
        #   base_link X (rojo) = Norte = ENU Y  →  vx_bl =  vy_enu
        #   base_link Y (verde) = Oeste = -ENU X →  vy_bl = -vx_enu
        def enu_to_baselink(ex, ey, ez):
            return ey, -ex, ez

        # ── Aceleración ───────────────────────────────────────────────────
        ax_enu, ay_enu, az_enu = ned_to_enu_position(*self.acc[idx])
        ax, ay, az = enu_to_baselink(ax_enu, ay_enu, az_enu)

        # ── Giroscopio ────────────────────────────────────────────────────
        wx_enu, wy_enu, wz_enu = ned_to_enu_position(*self.gyro[idx])
        wx, wy, wz = enu_to_baselink(wx_enu, wy_enu, wz_enu)

        # ── Velocidad lineal ──────────────────────────────────────────────
        vx_enu, vy_enu, vz_enu = ned_to_enu_position(*self.vel_body[idx])
        vx, vy, vz = enu_to_baselink(vx_enu, vy_enu, vz_enu)

        # ── IMU ───────────────────────────────────────────────────────────
        imu = Imu()
        imu.header.stamp    = tstamp
        imu.header.frame_id = self.robot_frame

        imu.linear_acceleration.x = ax
        imu.linear_acceleration.y = ay
        imu.linear_acceleration.z = az
        imu.linear_acceleration_covariance = [
            0.01, 0.0,  0.0,
            0.0,  0.01, 0.0,
            0.0,  0.0,  0.01
        ]

        imu.angular_velocity.x = wx
        imu.angular_velocity.y = wy
        imu.angular_velocity.z = wz
        imu.angular_velocity_covariance = [
            0.005, 0.0,   0.0,
            0.0,   0.005, 0.0,
            0.0,   0.0,   0.005
        ]

        imu.orientation_covariance[0] = -1.0  # sin orientación desde IMU

        self.imu_pub.publish(imu)

        # ── Odometría (solo twist) ────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = tstamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = self.robot_frame

        odom.twist.twist.linear.x  = vx
        odom.twist.twist.linear.y  = vy
        odom.twist.twist.linear.z  = vz
        odom.twist.twist.angular.x = wx
        odom.twist.twist.angular.y = wy
        odom.twist.twist.angular.z = wz

        odom.twist.covariance = [
            0.01, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.01, 0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.01, 0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.01, 0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.01, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.05
        ]

        self.odom_pub.publish(odom)


    def _timer_callback(self):

        if self.current_frame >= self.total_frames:
            if self.loop:
                self.current_frame = 0
                if self.pub_path:
                    self.robot_path_msg.poses.clear()
                    self.camera_path_msg.poses.clear()
                self.get_logger().info('↺ Reiniciando reproducción (loop=true).')
            else:
                self.get_logger().info('✓ Reproducción finalizada.')
                self.timer.cancel()
                return

        tf_lst = []

        # now = self._to_ros_time(self.imu_times[self.current_frame])
        now = self.get_clock().now().to_msg()

        # msg_clock = Clock()
        # msg_clock.clock = now
        # self.clock_pub.publish(msg_clock)

        frame = self.current_frame

        if frame == 0:
            self.get_logger().info(f'NO SE HA ACTUALIZADO')

        # ── Robot: NED → ENU ────────────────────────────────────────────────
        robot_tf_GT = self._pose_ned_to_tf_enu(
            self.robot_poses_ned[frame],
            parent=self.world_frame,
            child=self.robot_frame_GT,
        )
        
        robot_tf_GT.header.stamp = now

        tf_lst.append(robot_tf_GT)

        # --- Tiempo ---
        t = self.imu_times[frame]

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        if dt <= 0 or dt > 1.0:
            self.get_logger().info(f'OJO AQUI {t}, {dt}')
            return

        # ── Broadcast TF ────────────────────────────────────────────────────
        self.tf_broadcaster.sendTransform(tf_lst)

        self._publish_odom(now,frame)

        # ── Publicar PoseStamped ─────────────────────────────────────────────
        robot_ps  = self._make_pose_stamped(robot_tf_GT)
        
        self.robot_pose_pub.publish(robot_ps)

        if self.current_frame in self.closest_map:

            idx_cam = self.closest_map[frame]

            now = self.get_clock().now().to_msg()
            
            self._publish_rgb(os.path.join(self.rgb_path, self.rgb_files[idx_cam]), now)
            self._publish_pointcloud_ENU(idx_cam,now)
            self._pointcloud_to_laserscan(now)

        self.current_frame += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try:
        from rclpy.executors import MultiThreadedExecutor
        node = TartanGroundTFPublisher()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()