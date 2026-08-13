"""Node-owned input provider for Variant B (single rclpy owner).

In Variant B the service node owns *all* ROS2 I/O. Instead of the policy
constructing its own ``Ros2Input`` (Variant A), the node creates this provider,
subscribing on a node it owns, and injects it via the ``"injected"`` input
source (see ``holosoma_inference.inputs.create_input``).

Like ``Ros2Input``, this is a *single* object implementing both
``VelCmdProvider`` and ``StateCommandProvider`` — so when ``velocity_input`` and
``state_input`` are both ``injected`` the base policy shares one provider for
both roles (it reuses the velocity provider as the command provider, which then
must also expose ``poll_commands``). It does not own a node or spin thread —
subscriptions are created on the caller-supplied node, which the service node
spins once. ``start()`` is therefore a no-op.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.ros2 import ROS2_COMMAND_MAP

CMD_VEL_TOPIC = "cmd_vel"
STATE_INPUT_TOPIC = "holosoma/state_input"


class InjectedRos2Input:
    """Combined vel + state provider backed by subscriptions on a shared node.

    Subscribes a TwistStamped topic for velocity and a String topic for discrete
    state commands, both on the caller-supplied node (the service node owns the
    single spin thread). Mirrors ``Ros2Input``'s semantics: velocity clamped to
    [-1, 1] with timeout-zeroing; state strings mapped via ``ROS2_COMMAND_MAP``
    (including ``"switch_mode"`` -> ``SWITCH_MODE`` for the dual-mode swap).
    """

    def __init__(
        self,
        node: Node,
        vel_topic: str = CMD_VEL_TOPIC,
        cmd_topic: str = STATE_INPUT_TOPIC,
        vel_timeout: float = 1.0,
    ):
        from geometry_msgs.msg import TwistStamped
        from std_msgs.msg import String

        self._vel_timeout = vel_timeout
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))
        self._last_vel_time: float = 0.0
        self._lock = threading.Lock()
        self._queue: deque[StateCommand] = deque()

        # Newest-wins velocity (depth=1, BEST_EFFORT) suits a 50 Hz teleop loop;
        # state commands are reliable + queued so none are dropped.
        vel_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(TwistStamped, vel_topic, self._vel_callback, vel_qos)
        node.create_subscription(String, cmd_topic, self._cmd_callback, 10)
        logger.info(f"Injected ROS2 input subscribed to velocity: {vel_topic}, commands: {cmd_topic}")

    def start(self) -> None:
        """No-op: the service node owns the subscriptions + spin thread."""

    # --- velocity ---
    def _vel_callback(self, msg):
        with self._lock:
            self._lin_vel[0, 0] = max(-1.0, min(1.0, msg.twist.linear.x))
            self._lin_vel[0, 1] = max(-1.0, min(1.0, msg.twist.linear.y))
            self._ang_vel[0, 0] = max(-1.0, min(1.0, msg.twist.angular.z))
            self._last_vel_time = time.monotonic()

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

    # --- state commands ---
    def _cmd_callback(self, msg):
        cmd_str = msg.data.strip().lower()
        cmd = ROS2_COMMAND_MAP.get(cmd_str)
        if cmd is not None:
            self._queue.append(cmd)
        else:
            logger.warning(f"ROS2 command: unknown command '{cmd_str}'")

    def poll_commands(self) -> list[StateCommand]:
        commands: list[StateCommand] = []
        while self._queue:
            commands.append(self._queue.popleft())
        return commands
