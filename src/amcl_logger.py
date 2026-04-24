#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class AMCLTimestampLogger(Node):

    def __init__(self):
        super().__init__('amcl_timestamp_logger')

        # Suscriptor a AMCL
        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.callback,
            100
        )

        # Archivo de salida
        self.file = open("amcl_timestamps.txt", "w")

        self.get_logger().info("Nodo de logging de timestamps de AMCL iniciado")

    def callback(self, msg):
        # Obtener timestamp
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        # Guardar en archivo
        self.file.write(f"{t:.9f}\n")
        self.file.flush()

        self.get_logger().debug(f"Timestamp guardado: {t:.9f}")

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = AMCLTimestampLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()