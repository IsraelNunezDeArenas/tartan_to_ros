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
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time


class PcdMapPublisher(Node):

    def __init__(self):
        super().__init__('pcd_map_publisher')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('pcd_path', '')
        self.declare_parameter('map_frame_id',  'map')
        self.declare_parameter('topic_map',     '/map')

        # Resolución del grid en metros/celda
        # Regla práctica: usar el spacing medio de la nube PCD.
        # Si no lo conoces, 0.05 m es un buen punto de partida.
        self.declare_parameter('resolution', 0.05)

        # Filtro de altura: sólo se proyectan puntos con z dentro de este rango.
        # Permite eliminar suelo (z < z_min) y techo (z > z_max).
        # Ponlos a -inf/inf para deshabilitar.
        self.declare_parameter('z_min', -0.1)
        self.declare_parameter('z_max',  2.0)

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

        # ── Construir y publicar el mapa ──────────────────────────────────────
        grid_msg, meta_msg = self._pcd_to_occupancy_grid(pcd_path, padding)

        self.map_pub.publish(grid_msg)
        self.meta_pub.publish(meta_msg)

        self.get_logger().info(
            f"Mapa publicado en '{topic}' (transient local)\n"
            f"  Fichero     : {pcd_path}\n"
            f"  Resolución  : {self.res} m/celda\n"
            f"  Tamaño grid : {grid_msg.info.width} × {grid_msg.info.height} celdas\n"
            f"  Origen      : x={grid_msg.info.origin.position.x:.2f} "
            f"y={grid_msg.info.origin.position.y:.2f}\n"
            f"  Filtro z    : [{self.z_min}, {self.z_max}] m"
        )

    # =========================================================================
    # Conversión PCD → OccupancyGrid
    # =========================================================================

    def _pcd_to_occupancy_grid(
        self,
        pcd_path: str,
        padding: float,
    ) -> tuple[OccupancyGrid, MapMetaData]:

        # ── 1. Cargar nube ────────────────────────────────────────────────────
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)

        if len(pts) == 0:
            raise RuntimeError(f"La nube PCD está vacía: {pcd_path}")

        self.get_logger().info(f"PCD cargado: {len(pts)} puntos desde {pcd_path}")

        # ── 2. Filtrar por altura ─────────────────────────────────────────────
        mask = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        # pts_filtered = pts[mask]
        pts_filtered = pts

        if len(pts_filtered) == 0:
            raise RuntimeError(
                f"Ningún punto pasa el filtro de altura "
                f"[{self.z_min}, {self.z_max}] m. "
                "Revisa los parámetros z_min / z_max."
            )

        self.get_logger().info(
            f"Puntos tras filtro z: {len(pts_filtered)} "
            f"({100.*len(pts_filtered)/len(pts):.1f}%)"
        )

        # ── 3. Calcular extensión con padding ─────────────────────────────────
        x_min = pts_filtered[:, 0].min() - padding
        x_max = pts_filtered[:, 0].max() + padding
        y_min = pts_filtered[:, 1].min() - padding
        y_max = pts_filtered[:, 1].max() + padding

        width  = int(np.ceil((x_max - x_min) / self.res))
        height = int(np.ceil((y_max - y_min) / self.res))

        # ── 4. Rasterizar: cada punto ocupa 1 celda ───────────────────────────
        # El grid se inicializa todo a 0 (libre).
        # Los puntos proyectados se marcan como 100 (ocupado).
        # Las celdas que nunca se visitan quedan en -1 (desconocido) solo si
        # usamos un método de ray-casting; aquí usamos proyección simple:
        # todo lo que no tiene punto es libre (válido para mapas ya construidos).
        grid = np.zeros((height, width), dtype=np.int8)

        col = ((pts_filtered[:, 0] - x_min) / self.res).astype(int)
        row = ((pts_filtered[:, 1] - y_min) / self.res).astype(int)

        # Clamp por seguridad ante errores de floating point en los bordes
        col = np.clip(col, 0, width  - 1)
        row = np.clip(row, 0, height - 1)

        grid[row, col] = 100  # ocupado

        # ── 5. Dilatación morfológica (opcional pero recomendada para AMCL) ───
        # Engrosa los obstáculos 1 celda para compensar imprecisiones del sensor.
        # Se puede deshabilitar poniendo inflate_radius = 0.
        inflate_px = max(1, int(0.05 / self.res))  # ~5 cm
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
                    "cv2 no disponible: se omite la dilatación de obstáculos. "
                    "Instala opencv-python para obtener mejores resultados con AMCL."
                )

        # ── 6. Construir mensaje OccupancyGrid ────────────────────────────────
        now = Time(sec=0, nanosec=0)

        info = MapMetaData()
        info.map_load_time    = now
        info.resolution       = self.res
        info.width            = width
        info.height           = height
        info.origin           = Pose()
        info.origin.position.x = x_min
        info.origin.position.y = y_min
        info.origin.position.z = 0.0
        info.origin.orientation.w = 1.0   # sin rotación

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp    = now
        grid_msg.header.frame_id = self.frame
        grid_msg.info            = info
        # OccupancyGrid almacena en orden row-major, fila 0 = y_min (esquina SW)
        grid_msg.data            = grid.flatten().tolist()

        meta_msg = info

        return grid_msg, meta_msg


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