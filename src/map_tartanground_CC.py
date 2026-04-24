#!/usr/bin/env python3
"""
pcd_map_publisher.py
Convierte un mapa PCD a nav_msgs/OccupancyGrid con QoS transient-local.
Soporta rotación en Z (NED→ENU u otras) antes de rasterizar.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import open3d as o3d
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time


class PcdMapPublisher(Node):

    def __init__(self):
        super().__init__('pcd_map_publisher')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('pcd_path',     '')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('topic_map',    '/map')
        self.declare_parameter('resolution',   0.05)
        self.declare_parameter('z_min',       -0.1)
        self.declare_parameter('z_max',        2.0)
        self.declare_parameter('padding',      0.5)

        # 🔥 Rotación en Z aplicada a la nube antes de rasterizar (grados).
        #    Ejemplo NED→ENU: rotation_deg = -90.0
        #    Por defecto 0.0 (sin rotación).
        self.declare_parameter('rotation_deg', -90.0)

        # 🔥 Radio de dilatación morfológica en metros (0 = desactivado).
        self.declare_parameter('inflate_radius', 0.05)

        pcd_path    = self.get_parameter('pcd_path').value
        self.frame  = self.get_parameter('map_frame_id').value
        topic       = self.get_parameter('topic_map').value
        self.res    = self.get_parameter('resolution').value
        self.z_min  = self.get_parameter('z_min').value
        self.z_max  = self.get_parameter('z_max').value
        padding     = self.get_parameter('padding').value
        rot_deg     = self.get_parameter('rotation_deg').value
        inflate_m   = self.get_parameter('inflate_radius').value

        if not pcd_path:
            raise RuntimeError(
                "Debes indicar la ruta al PCD:\n"
                "  ros2 run <pkg> pcd_map_publisher "
                "--ros-args -p pcd_path:=/ruta/al/mapa.pcd"
            )

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub  = self.create_publisher(OccupancyGrid, topic,               latched_qos)
        self.meta_pub = self.create_publisher(MapMetaData,   topic + '_metadata', latched_qos)

        grid_msg, meta_msg = self._pcd_to_occupancy_grid(pcd_path, padding, rot_deg, inflate_m)

        self.map_pub.publish(grid_msg)
        self.meta_pub.publish(meta_msg)

        self.get_logger().info(
            f"Mapa publicado en '{topic}' (transient local)\n"
            f"  Fichero      : {pcd_path}\n"
            f"  Rotación Z   : {rot_deg}°\n"
            f"  Resolución   : {self.res} m/celda\n"
            f"  Tamaño grid  : {grid_msg.info.width} × {grid_msg.info.height} celdas\n"
            f"  Origen       : x={grid_msg.info.origin.position.x:.2f} "
            f"y={grid_msg.info.origin.position.y:.2f}\n"
            f"  Filtro z     : [{self.z_min}, {self.z_max}] m\n"
            f"  Inflate      : {inflate_m} m"
        )

    # =========================================================================

    @staticmethod
    def _rotation_matrix_z(deg: float) -> np.ndarray:
        """Matriz de rotación 2-D alrededor del eje Z (aplica sólo a XY)."""
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float64)

    def _pcd_to_occupancy_grid(
        self,
        pcd_path: str,
        padding: float,
        rot_deg: float,
        inflate_m: float,
    ) -> tuple[OccupancyGrid, MapMetaData]:

        # ── 1. Cargar nube ────────────────────────────────────────────────────
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points, dtype=np.float64)

        if len(pts) == 0:
            raise RuntimeError(f"La nube PCD está vacía: {pcd_path}")

        self.get_logger().info(f"PCD cargado: {len(pts)} puntos desde {pcd_path}")

        # ── 2. Rotación en Z (NED→ENU u otra) ────────────────────────────────
        if rot_deg != 0.0:
            R = self._rotation_matrix_z(rot_deg)
            pts[:, :2] = pts[:, :2] @ R.T   # rotación in-place sobre XY
            self.get_logger().info(f"Rotación aplicada: {rot_deg}° en Z")

        # ── 3. Filtrar por altura ─────────────────────────────────────────────
        # 🔥 FIX 1: aplicar la máscara que antes quedaba comentada
        mask = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        pts_filtered = pts[mask]

        if len(pts_filtered) == 0:
            raise RuntimeError(
                f"Ningún punto pasa el filtro de altura "
                f"[{self.z_min}, {self.z_max}] m. "
                "Revisa z_min / z_max."
            )

        self.get_logger().info(
            f"Puntos tras filtro z: {len(pts_filtered)} "
            f"({100. * len(pts_filtered) / len(pts):.1f}%)"
        )

        # ── 4. Extensión + padding ────────────────────────────────────────────
        x_min = pts_filtered[:, 0].min() - padding
        x_max = pts_filtered[:, 0].max() + padding
        y_min = pts_filtered[:, 1].min() - padding
        y_max = pts_filtered[:, 1].max() + padding

        width  = int(np.ceil((x_max - x_min) / self.res))
        height = int(np.ceil((y_max - y_min) / self.res))

        # ── 5. Rasterizar ─────────────────────────────────────────────────────
        grid = np.zeros((height, width), dtype=np.int8)

        col = ((pts_filtered[:, 0] - x_min) / self.res).astype(int)
        row = ((pts_filtered[:, 1] - y_min) / self.res).astype(int)
        col = np.clip(col, 0, width  - 1)
        row = np.clip(row, 0, height - 1)
        grid[row, col] = 100

        # ── 6. Dilatación morfológica ─────────────────────────────────────────
        # 🔥 FIX 2: inflate_px derivado del parámetro en metros, sin forzar mínimo 1
        inflate_px = int(round(inflate_m / self.res))
        if inflate_px > 0:
            try:
                import cv2
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1)
                )
                occupied = (grid == 100).astype(np.uint8)
                occupied = cv2.dilate(occupied, kernel)
                grid[occupied > 0] = 100
            except ImportError:
                self.get_logger().warn(
                    "cv2 no disponible: se omite la dilatación. "
                    "Instala opencv-python para mejores resultados con AMCL."
                )

        # ── 7. Mensaje OccupancyGrid ──────────────────────────────────────────
        now = Time(sec=0, nanosec=0)

        info = MapMetaData()
        info.map_load_time      = now
        info.resolution         = self.res
        info.width              = width
        info.height             = height
        info.origin             = Pose()
        info.origin.position.x  = x_min
        info.origin.position.y  = y_min
        info.origin.position.z  = 0.0
        info.origin.orientation.w = 1.0

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp    = now
        grid_msg.header.frame_id = self.frame
        grid_msg.info            = info
        grid_msg.data            = grid.flatten().tolist()

        return grid_msg, info


def main():
    rclpy.init()
    node = PcdMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()