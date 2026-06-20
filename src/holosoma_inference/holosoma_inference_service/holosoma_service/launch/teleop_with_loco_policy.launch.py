"""Live teleop -> locomotion policy, served over ROS2 (Variant A).

    /cmd_vel (Twist) ──────────────▶ policy_service_node ─▶ robot
    state (joystick or ROS2 String) ─┘  (publishes executed_cmd + heartbeat)

``policy_service_node`` builds the policy the same way ``run_policy.py`` does
(no extension-only ``policy_type``) and runs it as a ROS2 service. The Unitree
SDK is run through the ``unitree_mp`` multiprocess proxy automatically (the node
forces it), so the SDK's CycloneDDS is isolated from this process's rclpy.

Fully parameterized for composition into a larger launch: every knob is a
``DeclareLaunchArgument`` so a parent launch can override it.

Default config below: velocity over ROS2 ``/cmd_vel`` + state on the native G1
joystick (so the joystick L1+R1 kill is available). A velocity publisher drives
motion, e.g. ``demo_scripts/ros2_velocity_publisher.py --pattern shuttle``.

Usage:
    ros2 launch holosoma_service teleop_with_loco_policy.launch.py \
        model_path:=/path/model.onnx interface:=eth0
    # ROS2 state input instead of joystick:
    ros2 launch holosoma_service teleop_with_loco_policy.launch.py \
        model_path:=/path/model.onnx state_input:=ros2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Input sources accepted by holosoma_inference's input factory.
_INPUT_CHOICES = ["ros2", "interface", "joystick", "keyboard", "injected"]


def generate_launch_description() -> LaunchDescription:
    preset = LaunchConfiguration("preset")
    model = LaunchConfiguration("model_path")
    interface = LaunchConfiguration("interface")
    velocity_input = LaunchConfiguration("velocity_input")
    state_input = LaunchConfiguration("state_input")
    domain_id = LaunchConfiguration("domain_id")
    node_name = LaunchConfiguration("node_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "preset",
                default_value="inference:g1-29dof-loco",
                description="Inference preset (tyro positional), e.g. inference:g1-29dof-loco.",
            ),
            DeclareLaunchArgument("model_path", description="Path to the ONNX policy. Required."),
            DeclareLaunchArgument(
                "interface",
                default_value="eth0",
                description="Network interface to the robot SDK (e.g. eth0). Use 'auto' to auto-detect.",
            ),
            DeclareLaunchArgument(
                "velocity_input",
                default_value="ros2",
                choices=_INPUT_CHOICES,
                description="Source for velocity commands. Default 'ros2' (/cmd_vel TwistStamped).",
            ),
            DeclareLaunchArgument(
                "state_input",
                default_value="interface",
                choices=_INPUT_CHOICES,
                description="Source for discrete state (start/stop/walk/kill). Default 'interface' "
                "(native G1 joystick, so L1+R1 kill works). Use 'ros2' for /holosoma/state_input String.",
            ),
            DeclareLaunchArgument(
                "domain_id",
                default_value="0",
                description="DDS domain ID for the robot SDK communication.",
            ),
            DeclareLaunchArgument(
                "node_name",
                default_value="policy_service",
                description="ROS2 node name (override when composing multiple instances).",
            ),
            Node(
                package="holosoma_service",
                executable="policy_service_node",
                name=node_name,
                # tyro CLI: positional preset + --task.* overrides.
                arguments=[
                    preset,
                    "--task.model-path",
                    model,
                    "--task.velocity-input",
                    velocity_input,
                    "--task.state-input",
                    state_input,
                    "--task.interface",
                    interface,
                    "--task.domain-id",
                    domain_id,
                ],
                output="screen",
            ),
        ]
    )
