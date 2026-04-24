#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry

from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R

import numpy as np
import os

import tf2_ros


class TransformationListener(Node):

    def __init__(self):
        super().__init__('Transformation_Listener')

        self.declare_parameter('dataset_path', '/home/israel/tartanairpy/House/Data_omni/P0000')
        
        self.dataset_path = self.get_parameter('dataset_path').value
        
        self.cam_times = np.loadtxt(os.path.join(self.dataset_path,'imu','cam_time.txt'))
        self.imu_times = np.loadtxt(os.path.join(self.dataset_path, 'imu', 'imu_time.txt'))

        self.common_idxs = self.find_closest_values_fast(self.cam_times,self.imu_times)


        self.cam_poses = np.loadtxt(os.path.join(self.dataset_path,'pose_lcam_top.txt'))

        pos = np.loadtxt(os.path.join(self.dataset_path,'imu','pos_global.txt'))
        ori =  np.loadtxt(os.path.join(self.dataset_path,'imu','ori_global.txt'))

        roll, pitch, yaw = ori[:,0], ori[:,1], ori[:,2]

        cr = np.cos(roll/2); sr = np.sin(roll/2)
        cp = np.cos(pitch/2); sp = np.sin(pitch/2)
        cy = np.cos(yaw/2); sy = np.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        self.robot_poses = np.hstack((pos, np.stack((qx,qy,qz,qw), axis=1)))


        self.cam_index = 0
        self.imu_index = 0
        


        self._main_loop()


    
    def find_closest_values_fast(self, data1, data2):
    
        # 🔹 Ordenar pero guardando índices originales
        sorted_indices = np.argsort(data2)
        data2_sorted = data2[sorted_indices]

        results = []

        for i, v in enumerate(data1):
            idx = np.searchsorted(data2_sorted, v)

            if idx == 0:
                closest_idx_sorted = 0
            elif idx == len(data2_sorted):
                closest_idx_sorted = len(data2_sorted) - 1
            else:
                before = data2_sorted[idx - 1]
                after = data2_sorted[idx]

                if abs(v - before) < abs(v - after):
                    closest_idx_sorted = idx - 1
                else:
                    closest_idx_sorted = idx


            idx_data2 = sorted_indices[closest_idx_sorted]

            closest_value = data2[idx_data2]
            diff = v - closest_value

            results.append({
                "idx1": i,
                "idx2": idx_data2,
                "t1": v,
                "t2": closest_value,
                "diff": diff
            })

        return results
    
    def _pose_to_matrix(self, p):
        T = np.eye(4)
        T[:3,:3] = R.from_quat(p[3:]).as_matrix()
        T[:3,3] = p[:3]
        return T
        
    def transformation_distance(self, T1, T2):

        T_error = np.linalg.inv(T1) @ T2

        # 📍 traslación
        trans_error = np.linalg.norm(T_error[:3, 3])

        # 🔄 rotación
        rot_error = R.from_matrix(T_error[:3, :3]).magnitude()

        return trans_error, rot_error

    def evaluate_consistency(self, T_list):

        ref = T_list[0]
    
        trans_errors = []
        rot_errors = []
    
        for T in T_list:
        
            et, er = self.transformation_distance(ref, T)
    
            trans_errors.append(et)
            rot_errors.append(er)
    
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
    
        self.get_logger().info(
            f"Trans error → mean: {trans_errors.mean()}, max: {trans_errors.max()}"
        )
    
        self.get_logger().info(
            f"Rot error → mean: {rot_errors.mean()}, max: {rot_errors.max()}"
        )

    def _main_loop(self):

        T_list = []

        for dic in self.common_idxs:

            idx_cam = dic["idx1"]
            idx_robot = dic["idx2"]

            p_cam = self.cam_poses[idx_cam]
            p_robot = self.robot_poses[idx_robot]

            T_cam = self._pose_to_matrix(p_cam)
            T_robot = self._pose_to_matrix(p_robot)

            T_base_cam = np.linalg.inv(T_robot) @ T_cam

            T_list.append(T_base_cam)

        self.evaluate_consistency(T_list)






        

    
                                     

# ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TransformationListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()