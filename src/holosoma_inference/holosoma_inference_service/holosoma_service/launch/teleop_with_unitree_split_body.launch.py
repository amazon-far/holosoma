"""Split-body Unitree controller (arm_sdk + loco) driven by CmdExoskeleton.

    CmdExoskeleton ─▶ unitree_split_controller ─▶ arm_sdk + LocoClient ─▶ G1

Usage:
    ros2 launch holosoma_service teleop_with_unitree_split_body.launch.py iface:=eth0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    iface = LaunchConfiguration("iface")
    return LaunchDescription(
        [
            DeclareLaunchArgument("iface", default_value="eth0"),
            Node(
                package="holosoma_service",
                executable="unitree_split_controller",
                name="unitree_split_controller",
                arguments=["--iface", iface],
                output="screen",
            ),
        ]
    )
