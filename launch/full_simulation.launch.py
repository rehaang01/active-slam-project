"""
Full simulation launch file for Active SLAM project.

Launches: MicroXRCEAgent, ROS-GZ bridges (camera, depth, LiDAR),
          static TF publishers, MAVSDK heartbeat script, RTAB-Map.

NOTE: PX4 SITL is launched separately via tmuxinator (see rtabmap_ros2_gazebo.yml).
      The px4_process is defined here but commented out — tmuxinator handles it.

Fixes applied (v3):
  - FIX 1: PX4_GZ_WORLD changed from 'palace' to 'warehouse' (matches tmuxinator)
  - FIX 2: PX4_GZ_MODEL_POSE corrected for warehouse spawn (was palace coordinates)
  - FIX 3: LiDAR bridge topic uses world name variable (was hardcoded 'warehouse')
  - FIX 4 REVERTED: drone_control_script RESTORED.  It was removed in v2 thinking
    it conflicted with RL offboard control.  In reality, it serves a CRITICAL
    hidden purpose: it connects to PX4 via MAVLink and sends GCS heartbeats.
    Without these heartbeats, PX4 refuses to arm with:
      "Preflight Fail: No connection to the ground control station"
    The script does NOT conflict with RL control — it sits idle waiting for
    keyboard input (which never comes during autonomous runs).  The RL agent
    controls the drone via a SEPARATE channel (ROS2/DDS offboard commands).
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define paths
    px4_dir = '/root/PX4-Autopilot'
    mavsdk_script = '/root/PX4-ROS2-Gazebo-YOLOv8/keyboard-mavsdk-test.py'

    # ============================================
    # World configuration
    # FIX 1 & 2: Match the tmuxinator yml settings.
    # The tmuxinator yml launches PX4 with warehouse world.
    # If you use this launch file directly instead of tmuxinator,
    # these values must be consistent.
    # ============================================
    world_name = 'warehouse'

    px4_env = {
        'PX4_GZ_WORLD': world_name,
        'PX4_SYS_AUTOSTART': '4002',
        # FIX 2: Warehouse spawn: center of floor, 1m altitude, facing -X
        'PX4_GZ_MODEL_POSE': '0.0,0.0,1.0,0.00,0,3.14',
        'PX4_GZ_MODEL': 'x500_depth'
    }

    # Launch PX4 SITL (commented out — launched by tmuxinator instead)
    px4_process = ExecuteProcess(
        cmd=[os.path.join(px4_dir, 'build/px4_sitl_default/bin/px4')],
        cwd=px4_dir,
        env=px4_env,
        output='screen'
    )

    # Launch MicroXRCEAgent
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )

    # MAVSDK heartbeat script (CRITICAL — PX4 won't arm without this)
    # This script connects to PX4 via MAVLink and sends GCS heartbeats
    # as a side effect of the MAVSDK connection.  PX4 requires a GCS
    # heartbeat to pass preflight arming checks.
    # It does NOT interfere with RL offboard control — it just sits
    # idle waiting for keyboard input that never comes.
    drone_control_script = ExecuteProcess(
        cmd=['python3', mavsdk_script],
        output='screen'
    )

    # ============================================
    # ROS-GZ bridges — camera info + LiDAR
    # ============================================
    # FIX 3: LiDAR Gazebo topic path includes the world name.
    lidar_gz_topic = (
        f'/world/{world_name}/model/x500_depth_0'
        f'/link/link/sensor/lidar_2d_v2/scan'
    )

    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
            f'{lidar_gz_topic}@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
        ],
        remappings=[
            (lidar_gz_topic, '/scan'),
        ],
        output='screen'
    )

    image_bridge_camera = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera'],
        output='screen'
    )

    image_bridge_depth = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/depth_camera'],
        output='screen'
    )

    # ============================================
    # Static transform publishers
    # ============================================
    static_tf_imx214 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'x500_depth_0/OakD-Lite/base_link',
                   '--child-frame-id', 'x500_depth_0/OakD-Lite/base_link/IMX214'],
        output='screen'
    )

    static_tf_stereo = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'x500_depth_0/OakD-Lite/base_link',
                   '--child-frame-id', 'x500_depth_0/OakD-Lite/base_link/StereoOV7251'],
        output='screen'
    )

    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0.12', '--y', '0', '--z', '0.26',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'x500_depth_0/base_link',
                   '--child-frame-id', 'x500_depth_0/link/lidar_2d_v2'],
        output='screen'
    )

    # ============================================
    # RTAB-Map
    # ============================================
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')
    rtabmap_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch_dir, '/launch/rtabmap.launch.py']),
        launch_arguments={
            'rtabmap_args': '--delete_db_on_start',
            'rgbd': 'true',
            'rgb_topic': '/camera',
            'depth_topic': '/depth_camera',
            'camera_info_topic': '/camera_info',
            'frame_id': 'x500_depth_0/OakD-Lite/base_link',
            'use_sim_time': 'true'
        }.items()
    )

    return LaunchDescription([
        #px4_process,  # Launched by tmuxinator instead
        micro_xrce_agent,
        parameter_bridge,
        image_bridge_camera,
        image_bridge_depth,
        drone_control_script,  # RESTORED — provides MAVLink GCS heartbeats for arming
        static_tf_imx214,
        static_tf_stereo,
        static_tf_lidar,
        rtabmap_include
    ])