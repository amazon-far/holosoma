"""ROS2 input provider."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from loguru import logger

from holosoma_inference.inputs.api.base import InputProvider
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
    "switch_mode": StateCommand.SWITCH_MODE,
}


# Guards rclpy.init() against concurrent calls from multiple provider threads.
_ros2_init_lock = threading.Lock()


def _ensure_ros2_init() -> None:
    """Call rclpy.init() if not already initialized."""
    import rclpy

    with _ros2_init_lock:
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass  # Already initialized


class Ros2Input(InputProvider):
    """Single ROS2 node providing both velocity and state command inputs.

    Subscribes to a TwistStamped topic for velocity and a String topic for
    discrete state commands.  One node, one spin thread.
    """

    def __init__(self, vel_topic: str, cmd_topic: str, vel_timeout: float = 1.0):
        self._vel_topic = vel_topic
        self._cmd_topic = cmd_topic
        self._vel_timeout = vel_timeout
        # Velocity state
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))
        self._last_vel_time: float = 0.0
        self._lock = threading.Lock()
        # Command queue
        self._queue: deque[StateCommand] = deque()

    def start(self) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from std_msgs.msg import String

        _ensure_ros2_init()
        node = rclpy.create_node("holosoma_input")
        node.create_subscription(TwistStamped, self._vel_topic, self._vel_callback, 10)
        node.create_subscription(String, self._cmd_topic, self._cmd_callback, 10)
        logger.info(f"Subscribed to ROS2 velocity: {self._vel_topic}, commands: {self._cmd_topic}")
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def _vel_callback(self, msg):
        """Store velocity from ROS2. Clamps to training range."""
        with self._lock:
            self._lin_vel[0, 0] = max(-1.0, min(1.0, msg.twist.linear.x))
            self._lin_vel[0, 1] = max(-1.0, min(1.0, msg.twist.linear.y))
            self._ang_vel[0, 0] = max(-1.0, min(1.0, msg.twist.angular.z))
            self._last_vel_time = time.monotonic()

    def _cmd_callback(self, msg):
        """Map ROS2 string command to enum and queue it."""
        cmd_str = msg.data.strip().lower()
        cmd = ROS2_COMMAND_MAP.get(cmd_str)
        if cmd is not None:
            self._queue.append(cmd)
        else:
            logger.warning(f"ROS2 command: unknown command '{cmd_str}'")

    def zero(self) -> None:
        with self._lock:
            self._lin_vel[:] = 0.0
            self._ang_vel[:] = 0.0

    def poll_velocity(self) -> VelCmd:
        with self._lock:
            if (
                self._vel_timeout > 0
                and self._last_vel_time > 0
                and (time.monotonic() - self._last_vel_time) > self._vel_timeout
            ):
                self._lin_vel[:] = 0.0
                self._ang_vel[:] = 0.0
                self._last_vel_time = 0.0
                logger.warning("Velocity timeout — zeroing commands")
            return VelCmd(
                (float(self._lin_vel[0, 0]), float(self._lin_vel[0, 1])),
                float(self._ang_vel[0, 0]),
            )

    def poll_commands(self) -> list[StateCommand]:
        """Drain all queued commands."""
        commands: list[StateCommand] = []
        while self._queue:
            commands.append(self._queue.popleft())
        return commands
