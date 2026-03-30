"""ROS2 input providers."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

# ---------------------------------------------------------------------------
# ROS2 string-to-command mapping
# ---------------------------------------------------------------------------

ROS2_COMMAND_MAP: dict[str, StateCommand] = {
    "start": StateCommand.START,
    "stop": StateCommand.STOP,
    "init": StateCommand.INIT,
    "walk": StateCommand.WALK,
    "stand": StateCommand.STAND,
}


def _ensure_ros2_init() -> None:
    """Call rclpy.init() if not already initialized."""
    import rclpy

    try:
        rclpy.init(args=None)
    except RuntimeError:
        pass  # Already initialized


class Ros2VelCmdProvider:
    """Subscribes to ROS2 TwistStamped topic for velocity commands."""

    def __init__(self, topic: str):
        self._topic = topic
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))

    def start(self) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped

        _ensure_ros2_init()
        node = rclpy.create_node("vel_cmd_input")
        node.create_subscription(TwistStamped, self._topic, self._callback, 10)
        node.get_logger().info(f"Subscribed to ROS2 velocity topic: {self._topic}")
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def _callback(self, msg):
        """Store velocity from ROS2. Clamps to training range."""
        self._lin_vel[0, 0] = max(-1.0, min(1.0, msg.twist.linear.x))
        self._lin_vel[0, 1] = max(-1.0, min(1.0, msg.twist.linear.y))
        self._ang_vel[0, 0] = max(-1.0, min(1.0, msg.twist.angular.z))

    def zero(self) -> None:
        self._lin_vel[:] = 0.0
        self._ang_vel[:] = 0.0

    def poll_velocity(self) -> VelCmd:
        return VelCmd(
            (float(self._lin_vel[0, 0]), float(self._lin_vel[0, 1])),
            float(self._ang_vel[0, 0]),
        )


class Ros2StateCommandProvider:
    """Subscribes to ROS2 String topic for discrete commands.

    Incoming string commands are mapped to enum values via ``ROS2_COMMAND_MAP``
    and queued.  The main loop drains them via ``poll_commands()``.
    """

    def __init__(self, topic: str):
        self._topic = topic
        self._queue: deque[StateCommand] = deque()
        self._logger = None

    def start(self) -> None:
        import rclpy
        from std_msgs.msg import String

        _ensure_ros2_init()
        node = rclpy.create_node("state_cmd_input")
        self._logger = node.get_logger()
        node.create_subscription(String, self._topic, self._callback, 10)
        self._logger.info(f"Subscribed to ROS2 state_input topic: {self._topic}")
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def _callback(self, msg):
        """Map ROS2 string command to enum and queue it."""
        cmd_str = msg.data.strip().lower()
        cmd = ROS2_COMMAND_MAP.get(cmd_str)
        if cmd is not None:
            self._queue.append(cmd)
        elif self._logger is not None:
            self._logger.warning(f"ROS2 command: unknown command '{cmd_str}'")

    def poll_commands(self) -> list[StateCommand]:
        """Drain all queued commands."""
        commands: list[StateCommand] = []
        while True:
            try:
                commands.append(self._queue.popleft())
            except IndexError:
                break
        return commands
