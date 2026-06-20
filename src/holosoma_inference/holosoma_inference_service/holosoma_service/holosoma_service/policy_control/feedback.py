"""Policy-agnostic feedback publisher.

Publishes the executed-command + heartbeat feedback for *any* holosoma policy,
driven by the policy's per-tick ``_on_command_sent`` hook (so it runs in the
policy control thread, in every state). Mirrors the split-body controller's
feedback topics:

  - ``/holosoma/holosoma_executed_cmd`` (``JointState``): the full-body joint
    command actually sent this tick, at the control rate.
  - ``/holosoma/heartbeat`` (``Heartbeat``): liveness + status at ~5 Hz.

Extracted verbatim (no behavior change) from the feedback half of the original
WBT-only ``holosoma_node`` so the loco path can reuse it.
"""

from __future__ import annotations

import numpy as np
import rclpy
from holosoma_msgs.msg import Heartbeat
from rclpy.node import Node
from sensor_msgs.msg import JointState

EXECUTED_CMD_TOPIC = "/holosoma/holosoma_executed_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
HEARTBEAT_EVERY = 10  # control ticks -> 5 Hz at a 50 Hz control loop


class PolicyFeedbackPublisher(Node):
    """Publishes executed-cmd + heartbeat from a policy's ``_on_command_sent`` hook.

    Policy-agnostic: works for locomotion, WBT, or any ``BasePolicy`` subclass.
    ``dof_names`` is set by the caller after the policy is built (the policy's
    joint names aren't known until then).
    """

    def __init__(self, dof_names: list[str] | None = None):
        super().__init__("holosoma_feedback")
        self.dof_names: list[str] = list(dof_names or [])
        self._cmd_pub = self.create_publisher(JointState, EXECUTED_CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        self._tick = 0

    def on_command_sent(self, policy, cmd_q) -> None:
        """Publish executed command (every tick) and heartbeat (~5 Hz)."""
        cmd = np.asarray(cmd_q, dtype=np.float64).reshape(-1)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if len(self.dof_names) == cmd.shape[0]:
            msg.name = self.dof_names
        msg.position = cmd.tolist()
        self._cmd_pub.publish(msg)

        if self._tick % HEARTBEAT_EVERY == 0:
            hb = Heartbeat()
            hb.header.stamp = msg.header.stamp
            hb.robot_connected = getattr(policy, "interface", None) is not None
            hb.control_mode = 0
            if getattr(policy, "use_policy_action", False):
                hb.status = "running"
            elif getattr(policy, "get_ready_state", False):
                hb.status = "get_ready"
            else:
                hb.status = "stiff_hold"
            self._hb_pub.publish(hb)
        self._tick += 1

    def make_hook(self, policy):
        """Return an ``_on_command_sent``-compatible callable bound to ``policy``.

        The policy calls ``_on_command_sent(cmd_q, robot_state_data)`` each tick;
        we ignore the state and forward ``(policy, cmd_q)`` to ``on_command_sent``.
        """

        def _hook(cmd_q, _state, _p=policy):
            self.on_command_sent(_p, cmd_q)

        return _hook


def spin_in_thread(node: Node) -> None:
    """Spin a node in a daemon thread on its OWN executor.

    ``rclpy.spin()`` uses the process-global executor, so calling it from more
    than one thread (here: feedback + dense bridge, alongside Ros2Input's own
    spin) raises "Executor is already spinning". Each node gets a dedicated
    SingleThreadedExecutor instead, so any number can spin concurrently.
    """
    import threading

    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
