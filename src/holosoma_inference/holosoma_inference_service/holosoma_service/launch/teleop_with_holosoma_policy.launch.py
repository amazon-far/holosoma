"""Live teleop -> WBT policy, served over ROS2 (unified service node).

    (smplh) CmdSMPLH -> retargeter_node -> CmdDense -> policy_service_node (WBT) -> robot
    (dense)                               CmdDense -> policy_service_node (WBT) -> robot

This launches the unified ``policy_service_node`` (Variant B): a single
``ServiceIONode`` owns every subscription/publisher and injects providers into
the policy. For WBT it attaches itself as the dense ``TargetSource`` (it already
subscribes ``CmdDense`` and serves ``get_target()``), so the same node that runs
locomotion runs WBT -- there is no separate WBT entrypoint. The policy class is
resolved from ``config.task.policy_type`` via the ``holosoma.policies.by_type``
entry-point group (see ``dual_mode._select_policy_class``), so the chosen
``preset`` must be registered by the installed policy extension.

``input_type`` selects how CmdDense is produced:

* ``smplh`` (default): launch the retargeter so a SMPL-H source (``CmdSMPLH``)
  is retargeted into ``CmdDense``. Requires ``urdf_path`` (the retargeter's IK
  model).
* ``dense``: skip the retargeter; an external publisher feeds ``CmdDense``
  directly (e.g. ``scripts/publish_from_npz.py`` or any teleop already in dense
  29-DOF convention). ``urdf_path`` is not needed.

State commands (start/stop/start_motion_clip/kill) arrive on
``/holosoma/state_input`` as Strings; the in-band emergency kill is
``ros2 topic pub /holosoma/state_input std_msgs/String "{data: kill}"``.

Usage:
    # SMPL-H teleop (retargeter on):
    ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
        preset:=<inference-preset> urdf_path:=/path/g1_29dof.urdf model_path:=/path/model.onnx

    # Dense input (retargeter off):
    ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
        preset:=<inference-preset> input_type:=dense model_path:=/path/model.onnx
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node

# Input sources accepted by holosoma_inference's input factory.
_INPUT_CHOICES = ["ros2", "interface", "joystick", "keyboard", "injected"]


def generate_launch_description() -> LaunchDescription:
    urdf = LaunchConfiguration("urdf_path")
    model = LaunchConfiguration("model_path")
    preset = LaunchConfiguration("preset")
    rl_rate = LaunchConfiguration("rl_rate_hz")
    input_type = LaunchConfiguration("input_type")
    retargeter = LaunchConfiguration("retargeter")
    velocity_input = LaunchConfiguration("velocity_input")
    state_input = LaunchConfiguration("state_input")
    interface = LaunchConfiguration("interface")
    domain_id = LaunchConfiguration("domain_id")
    node_name = LaunchConfiguration("node_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument(
                "input_type",
                default_value="smplh",
                choices=["smplh", "dense"],
                description="smplh: run the retargeter (CmdSMPLH->CmdDense). "
                "dense: skip it, an external publisher feeds CmdDense directly.",
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value="",
                description="Fixed-base 29-DOF URDF for the retargeter IK. Required for input_type:=smplh.",
            ),
            DeclareLaunchArgument(
                "retargeter",
                default_value="g1-smpl",
                description="Retargeter impl registered under the 'holosoma.retargeter' entry-point group. "
                "Extensions can register their own and select it here.",
            ),
            DeclareLaunchArgument(
                "preset",
                description="Inference preset registered in the installed policy extension "
                "(resolves the policy via config.task.policy_type). Required.",
            ),
            DeclareLaunchArgument("rl_rate_hz", default_value="50.0"),
            DeclareLaunchArgument(
                "velocity_input",
                default_value="injected",
                choices=_INPUT_CHOICES,
                description="Source for velocity commands. Variant B default 'injected' "
                "(node-owned /cmd_vel subscription). WBT ignores velocity but the base loop still polls it.",
            ),
            DeclareLaunchArgument(
                "state_input",
                default_value="injected",
                choices=_INPUT_CHOICES,
                description="Source for discrete state (start/stop/start_motion_clip/kill). Variant B "
                "default 'injected' (node-owned /holosoma/state_input String; 'kill' triggers exit). "
                "Use 'interface' for the native G1 joystick (L1+R1 kill).",
            ),
            DeclareLaunchArgument(
                "interface",
                default_value="eth0",
                description="Network interface to the robot SDK (e.g. eth0). Use 'auto' to auto-detect.",
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
            # Retargeter: only when consuming SMPL-H input.
            Node(
                package="holosoma_service",
                executable="retargeter_node",
                name="retargeter",
                arguments=["--urdf-path", urdf, "--rl-rate-hz", rl_rate, "--retargeter", retargeter],
                output="screen",
                condition=IfCondition(EqualsSubstitution(input_type, "smplh")),
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
