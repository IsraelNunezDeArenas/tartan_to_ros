#!/usr/bin/env python3
"""
pcd_map_publisher.py
Convierte un mapa en formato PCD a nav_msgs/OccupancyGrid y lo publica
con QoS transient local (latched) para que nav2_amcl lo reciba
independientemente del orden de arranque.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import open3d as o3d
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, TransformStamped
from builtin_interfaces.msg import Time
from std_msgs.msg import Empty



from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster



class PcdMapPublisher(Node):

    def __init__(self):
        super().__init__('pcd_map_publisher')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('pcd_path', '/home/israel/tartanairpy/Office/Office_rgb.pcd')
        self.declare_parameter('map_frame_id',  'map')
        self.declare_parameter('topic_map',     '/map')

        # Resolución del grid en metros/celda
        # Regla práctica: usar el spacing medio de la nube PCD.
        # Si no lo conoces, 0.05 m es un buen punto de partida.
        self.declare_parameter('resolution', 0.02)

        # Filtro de altura: sólo se proyectan puntos con z dentro de este rango.
        # Permite eliminar suelo (z < z_min) y techo (z > z_max).
        # Ponlos a -inf/inf para deshabilitar.
        self.declare_parameter('z_min', 0.2)
        self.declare_parameter('z_max',  0.3)

        # Margen extra alrededor del mapa (en metros) para que AMCL tenga espacio
        self.declare_parameter('padding', 0.5)

        pcd_path    = self.get_parameter('pcd_path').value
        self.frame  = self.get_parameter('map_frame_id').value
        topic       = self.get_parameter('topic_map').value
        self.res    = self.get_parameter('resolution').value
        self.z_min  = self.get_parameter('z_min').value
        self.z_max  = self.get_parameter('z_max').value
        padding     = self.get_parameter('padding').value

        if not pcd_path:
            raise RuntimeError(
                "Debes indicar la ruta al PCD:\n"
                "  ros2 run <pkg> pcd_map_publisher "
                "--ros-args -p pcd_path:=/ruta/al/mapa.pcd"
            )

        # ── QoS latched (equivalente a latched=True en ROS 1) ─────────────────
        # AMCL (y map_server) esperan transient local + reliable
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub  = self.create_publisher(OccupancyGrid, topic,          latched_qos)
        self.meta_pub = self.create_publisher(MapMetaData,   topic + '_metadata', latched_qos)

        self.map_published = self.create_publisher(Empty,'map_recv',10)


        # ── Construir y publicar el mapa ──────────────────────────────────────
        grid_msg, meta_msg = self._pcd_to_occupancy_grid(pcd_path, padding)

        self.get_logger().info(
            f"Mapa publicado en '{topic}' (transient local)\n"
            f"  Fichero     : {pcd_path}\n"
            f"  Resolución  : {self.res} m/celda\n"
            f"  Tamaño grid : {grid_msg.info.width} × {grid_msg.info.height} celdas\n"
            f"  Origen      : x={grid_msg.info.origin.position.x:.2f} "
            f"y={grid_msg.info.origin.position.y:.2f}\n"
            f"  Filtro z    : [{self.z_min}, {self.z_max}] m"
        )

        self.map_pub.publish(grid_msg)
        self.meta_pub.publish(meta_msg)

        self.map_published.publish(Empty())

    # =========================================================================
    # Conversión PCD → OccupancyGrid
    # =========================================================================

    def _pcd_to_occupancy_grid(self, pcd_path, padding):

        # ── 1. Cargar ─────────────────────────────────────────────────────────
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            raise RuntimeError(f"PCD vacío: {pcd_path}")
        self.get_logger().info(f"PCD: {len(pts)} puntos")

        # ── 2. Rotación NED→ENU ───────────────────────────────────────────────
        R = np.array([[0, 1,  0],
                      [1, 0,  0],
                      [0, 0, -1]], dtype=float)
        pts = (R @ pts.T).T

        # ── 3. Filtro de altura — captura toda la pared ───────────────────────
        mask = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        pts_filtered = pts[mask]   # ✅ aplicado
        self.get_logger().info(
            f"Tras filtro z [{self.z_min},{self.z_max}m]: "
            f"{len(pts_filtered)} puntos ({100.*len(pts_filtered)/len(pts):.1f}%)")

        if len(pts_filtered) == 0:
            raise RuntimeError("Sin puntos tras filtro z. Revisa z_min/z_max.")

        # ── 4. Voxel downsampling para homogeneizar densidad ─────────────────
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_filtered)
        pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=self.res)
        pts_filtered = np.asarray(pcd_tmp.points)
        self.get_logger().info(f"Tras voxel downsampling: {len(pts_filtered)} puntos")

        # ── 5. Grid ───────────────────────────────────────────────────────────
        x0 = pts_filtered[:, 0].min() - padding
        y0 = pts_filtered[:, 1].min() - padding
        x1 = pts_filtered[:, 0].max() + padding
        y1 = pts_filtered[:, 1].max() + padding

        W = int(np.ceil((x1 - x0) / self.res))
        H = int(np.ceil((y1 - y0) / self.res))
        self.get_logger().info(f"Grid: {W}×{H} celdas ({W*H/1e6:.1f}M)")

        grid = np.zeros((H, W), dtype=np.int8)

        col = np.clip(((pts_filtered[:, 0] - x0) / self.res).astype(int), 0, W-1)
        row = np.clip(((pts_filtered[:, 1] - y0) / self.res).astype(int), 0, H-1)
        grid[row, col] = 100

        # ── 6. Dilatación en dos pasadas ──────────────────────────────────────
        try:
            import cv2

            # Pasada 1: rellena huecos entre puntos de la misma pared
            fill_px = max(3, int(0.15 / self.res))
            k_fill  = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*fill_px+1, 2*fill_px+1))
            occupied = cv2.dilate((grid == 100).astype(np.uint8), k_fill)

            # Pasada 2: grosor mínimo para que AMCL detecte la pared
            wall_px = max(1, int(0.05 / self.res))
            k_wall  = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*wall_px+1, 2*wall_px+1))
            occupied = cv2.dilate(occupied, k_wall)

            grid[occupied > 0] = 100

        except ImportError:
            self.get_logger().warn("cv2 no disponible; sin dilatación.")

        # ── 7. Mensaje ────────────────────────────────────────────────────────
        now = self.get_clock().now().to_msg()   # ✅ timestamp real

        info                       = MapMetaData()
        info.map_load_time         = now
        info.resolution            = self.res
        info.width                 = W
        info.height                = H
        info.origin                = Pose()
        info.origin.position.x     = x0
        info.origin.position.y     = y0
        info.origin.orientation.w  = 1.0

        msg                 = OccupancyGrid()
        msg.header.stamp    = now
        msg.header.frame_id = self.frame
        msg.info            = info
        msg.data            = grid.flatten().tolist()

        return msg, info


def main():
    rclpy.init()
    node = PcdMapPublisher()
    # El mapa ya está publicado (latched); solo necesitamos mantener el nodo vivo
    # para que los suscriptores tardíos (AMCL, rviz) reciban el QoS transient local.
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()