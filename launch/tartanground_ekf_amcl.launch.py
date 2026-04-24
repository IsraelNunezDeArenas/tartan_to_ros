from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ================================================================
    # PATHS
    # ================================================================
    pkg_share = get_package_share_directory('tartan_to_ros')

    ekf_yaml = os.path.join(
        pkg_share,
        'config',
        'ekf_tartanground.yaml'
    )

    # ================================================================
    # NODO 1 — DATASET
    # ================================================================
    tartan_node = Node(
        package='tartan_to_ros',
        executable='tartanground_node_ekf2.py',
        name='tartanground_node',
        output='screen',
        parameters=[{
            'dataset_path': '/home/israelnunez/tartanairpy/Office/Data_omni/P0000',

            'topic_rgb_image': '/camera/rgb',
            'topic_depth_image': '/camera/depth',
            'topic_camera_info': '/camera/camera_info',
            'topic_localization_gt': '/gt/robot_pose',

            'map_frame_id': 'map',
            'robot_frame_id': 'base_link',
            'camera_frame_id': 'camera',
            'lidar_frame_id': 'laser',

            'publish_rate': 10.0,

            'img_width': 640,
            'img_height': 640,
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 320.0,

            'use_sim_time': True
        }]
    )

    # ================================================================
    # NODO 2 — MAPA
    # ================================================================
    map_node = Node(
        package='tartan_to_ros',
        executable='map_tartanground.py',
        name='pcd_map_publisher',
        output='screen',
        parameters=[{
            'pcd_path': '/home/israelnunez/tartanairpy/Office/Office_rgb.pcd',
            'topic_map': '/map',
            'map_frame_id': 'map',
            'resolution': 0.05,
            'z_min': -0.5,
            'z_max': 3.0,
            'padding': 0.5,
            'use_sim_time': True
        }]
    )

    # ================================================================
    # EKF
    # ================================================================
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_yaml,
            {'use_sim_time': True}
        ]
    )

    # ================================================================
    # AMCL
    # ================================================================
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',

            'scan_topic': '/scan',

            'set_initial_pose': True,

            'min_particles': 500,
            'max_particles': 2000,

            'update_min_d': 0.05,
            'update_min_a': 0.05,

            'transform_tolerance': 1.0,
            'tf_broadcast': True,

            'laser_model_type': 'likelihood_field',
        }]
    )

    # ================================================================
    # LIFECYCLE MANAGER
    # ================================================================
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['amcl'],
            'bond_timeout': 4.0
        }]
    )

    # ================================================================
    # RVIZ
    # ================================================================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            os.path.join(pkg_share, 'rviz', 'TartanGroundRVIZ.rviz')
        ],
        parameters=[{'use_sim_time': True}]
    )

    # ================================================================
    # EVENT-DRIVEN STARTUP
    # ================================================================

    ekf_event = RegisterEventHandler(
        OnProcessStart(
            target_action=tartan_node,
            on_start=[ekf_node]
        )
    )

    amcl_event = RegisterEventHandler(
        OnProcessStart(
            target_action=ekf_node,
            on_start=[amcl_node]
        )
    )

    lifecycle_event = RegisterEventHandler(
        OnProcessStart(
            target_action=amcl_node,
            on_start=[lifecycle_manager]
        )
    )

    # ================================================================
    # LAUNCH
    # ================================================================
    return LaunchDescription([
        tartan_node,
        map_node,
        rviz_node,

        ekf_event,
        amcl_event,
        lifecycle_event
    ])