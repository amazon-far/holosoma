"""Synthetic (offline) interface for ``--task.debug.dryer-run``.

Fabricates a plausible standing robot state instead of reading it from a sim
bridge or hardware driver, so the policy loop runs with nothing else present.
Used to smoke-test config loading, ONNX inference, and the observation pipeline
off-robot.

For measurement, ``send_low_command`` publishes the commands the policy *would*
send to a **separate synthetic topic** (``/dryer_run/cmd``, ``sensor_msgs/
JointState``) that no real driver subscribes to — so you can ``ros2 topic hz /
dryer_run/cmd`` / ``echo`` / ``bag record`` the loop's output while nothing is
ever sent to the actual robot. If ROS2 is unavailable (pure offline run) it
degrades to a no-op and stays fully offline.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface

# Deliberately NOT a real driver topic (e.g. /alpha/joints/cmd). Nothing on the
# robot subscribes here, so publishing is safe — it's for measurement only.
_SYNTHETIC_CMD_TOPIC = "/dryer_run/cmd"


class SyntheticInterface(BaseInterface):
    """Returns a fixed, upright standing state; publishes cmds to a fake topic.

    State layout matches :meth:`BaseInterface.get_low_state`:
    ``[base_pos(3), quat(4), dof_pos(N), lin_vel(3), ang_vel(3), dof_vel(N)]``
    followed by an appended ``projected_gravity(3)`` (upright).
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        domain_id: int = 0,
        interface_str: str | None = None,
        use_joystick: bool = False,
        publish_cmd: bool = True,
        cmd_topic: str = _SYNTHETIC_CMD_TOPIC,
    ):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        default_angles = getattr(robot_config, "default_dof_angles", None)
        if default_angles is not None and len(default_angles) > 0:
            self._dof_pos = np.asarray(default_angles, dtype=float)
        else:
            n = len(getattr(robot_config, "dof_names", []) or [])
            self._dof_pos = np.zeros(n if n > 0 else 1)
        self._num_dof = self._dof_pos.shape[0]

        dof_names = list(getattr(robot_config, "dof_names", []) or [])
        if len(dof_names) != self._num_dof:
            dof_names = [f"joint_{i}" for i in range(self._num_dof)]
        self._dof_names = dof_names

        # Upright: identity quaternion (w,x,y,z), gravity straight down in the
        # base frame, zero linear/angular velocity, zero joint velocity.
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._projected_gravity = np.array([0.0, 0.0, -1.0])

        self._kp_level = 1.0
        self._kd_level = 1.0

        logger.warning(
            "DRYER RUN: using SyntheticInterface — state is fabricated (upright, "
            f"default pose, zero velocities across {self._num_dof} DOF) and NO "
            "connection to any bridge/driver is opened."
        )

        # Optional command publishing to a synthetic topic for measurement.
        self._cmd_topic = cmd_topic
        self._cmd_pub = None
        self._node = None
        self._JointState = None
        if publish_cmd:
            self._setup_cmd_publisher()

    def _setup_cmd_publisher(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import JointState
        except Exception as exc:  # ROS2 not sourced → stay fully offline.
            logger.warning(
                f"DRYER RUN: command publishing requested but ROS2 is unavailable "
                f"({exc}); running fully offline — no synthetic cmd topic."
            )
            return

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("dryer_run_synthetic")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self._node.create_publisher(JointState, self._cmd_topic, qos)
        self._JointState = JointState
        logger.warning(
            f"DRYER RUN: publishing synthetic commands on {self._cmd_topic} "
            "(sensor_msgs/JointState: name/position/velocity/effort) for "
            "measurement. No real driver subscribes to this topic — nothing is "
            "sent to the robot."
        )

    def get_low_state(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(3),  # base_pos
                self._quat,  # quat (w,x,y,z)
                self._dof_pos,  # dof_pos
                np.zeros(3),  # base lin_vel
                np.zeros(3),  # base ang_vel
                np.zeros(self._num_dof),  # dof_vel
                self._projected_gravity,  # projected_gravity (upright)
            ]
        ).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ) -> None:
        # Only ever publishes to the synthetic measurement topic — never to a
        # real driver. If publishing is disabled/unavailable, this is a no-op.
        if self._cmd_pub is None:
            return
        # On shutdown (Ctrl-C / SIGTERM) the rclpy context can be torn down
        # while a final publish is in flight — guard so exit stays clean.
        try:
            import rclpy

            if not rclpy.ok():
                return
        except Exception:
            return
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = self._dof_names
        msg.position = np.asarray(cmd_q, dtype=float).reshape(-1).tolist()
        msg.velocity = np.asarray(cmd_dq, dtype=float).reshape(-1).tolist()
        msg.effort = np.asarray(cmd_tau, dtype=float).reshape(-1).tolist()
        try:
            self._cmd_pub.publish(msg)
        except Exception:
            # Context invalidated mid-publish during shutdown — ignore.
            return

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = float(value)

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = float(value)
