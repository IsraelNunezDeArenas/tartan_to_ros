# tartan_to_ros

ROS2 - Humble package for replaying the [TartanGround](https://tartanair.org/tartanground/) dataset scenes by publishing its data as ROS messages.

## Description

This ROS2 package provides tools to replay the TartanGround dataset as ROS2 messages, enabling its data to be integrated into ROS-based applications and experiments. It reads the dataset files and publishes the corresponding sensor data through standard ROS topics, allowing algorithms such as localization, mapping, odometry estimation, and point cloud registration to be evaluated using reproducible data. The package is intended to facilitate the testing and benchmarking of ROS-based perception and mapping systems without requiring the original sensor setup.

## Features

- Camera and LiDAR data replay: Reproduces camera and LiDAR data through ROS messages using the correct reference frame, including the required NED-to-ENU coordinate frame conversion.
- Velocity and acceleration data: Publishes velocity and acceleration measurements through IMU and ODOM topics, enabling integration with other ROS packages.
- 3D to 2D map conversion: Converts the 3D map into a 2D occupancy grid map for use with localization algorithms.
- No semantic information: This version does not include semantic information integration.
- Examples are included for integration with [Voxeland](https://github.com/MAPIRlab/Voxeland) and [EKF](https://docs.ros.org/en/melodic/api/robot_localization/html/index.html)

## Requirements

- ROS 2 Humble
- CMake >= 3.8
- OpenCV - cv_bridge
- Open3D
- Sensor, Nav, Geometry msgs

## Instalation

Clone this repo:

```bash
cd ~/ros2_ws/src
git clone git@github.com:IsraelNunezDeArenas/tartan_to_ros.git
```

Modify:
- *dataset_path* parameter to choose the trajectory you wat to replay: tartanground_player node
- *pcd_path* parameter to choose the scene map: map_tartanground node

Information about parameters is included in launch files.