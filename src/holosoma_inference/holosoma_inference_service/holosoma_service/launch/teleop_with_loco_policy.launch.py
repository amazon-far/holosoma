"""Live teleop -> locomotion policy, served over ROS2 (Variant B).

    /cmd_vel (Twist) ──────────────▶ ServiceIONode ─▶ policy ─▶ robot
    /holosoma/state_input (String) ──┘  (one node owns ALL ROS2 I/O;
                                         publishes executed_cmd + heartbeat)

``policy_service_node`` builds the policy the same way ``run_policy.py`` does
and runs it as a ROS2 service. Unlike Variant A, a single ``ServiceIONode`` owns
every subscription/publisher and injects the velocity+state providers into the
policy (input source ``injected``). The Unitree SDK runs through the
``unitree_mp`` multiprocess proxy automatically (the node forces it), isolating
its CycloneDDS from this process's rclpy.

Fully parameterized for composition into a larger launch: every knob is a
``DeclareLaunchArgument`` so a parent launch can override it.

State commands (start/stop/walk/kill) arrive on ``/holosoma/state_input`` as
Strings; the in-band emergency kill is
``ros2 topic pub /holosoma/state_input std_msgs/String "{data: kill}"``. Velocity
is driven by e.g. ``demo_scripts/ros2_velocity_publisher.py --pattern shuttle``.

Usage:
    ros2 launch holosoma_service teleop_with_loco_policy.launch.py \
        model_path:=/path/model.onnx interface:=eth0
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
                default_value="injected",
                choices=_INPUT_CHOICES,
                description="Source for velocity commands. Variant B default 'injected' "
                "(node-owned /cmd_vel subscription).",
            ),
            DeclareLaunchArgument(
                "state_input",
                default_value="injected",
                choices=_INPUT_CHOICES,
                description="Source for discrete state (start/stop/walk/kill). Variant B default "
                "'injected' (node-owned /holosoma/state_input String; 'kill' triggers exit). "
                "Use 'interface' for the native G1 joystick (L1+R1 kill).",
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
