"""Dense tracking-target bridge (WBT-only input seam).

Subscribes ``CmdDense`` and serves it as a WBT ``TargetSource`` (newest-wins;
holds the last frame between policy ticks) via ``get_target()``, which the
policy pulls each control tick.

Extracted verbatim (no behavior change) from the input half of the original
WBT-only ``holosoma_node``. Attached to a policy only when that policy exposes a
``_target_source`` attribute (i.e. a WBT-style policy); locomotion policies have
no tracking target and are left untouched.
"""

from __future__ import annotations

import numpy as np
from holosoma_msgs.msg import CmdDense
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

DENSE_TOPIC = "/holosoma/dense_tracking_command"


class DenseTargetBridge(Node):
    """ROS subscriber that implements the WBT ``TargetSource`` protocol.

    ``get_target(num_dofs, rl_rate_hz, urdf_path)`` returns the newest
    ``(motion_command (1, 2*num_dofs), ref_quat_xyzw (4,))`` received on the
    dense topic, holding the last frame between policy ticks.
    """

    def __init__(self, num_dofs: int, topic: str = DENSE_TOPIC):
        super().__init__("holosoma_target_bridge")
        self._cmd = np.zeros((1, 2 * num_dofs), dtype=np.float32)  # held until first frame
        self._ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CmdDense, topic, self._cb, qos)

    def _cb(self, msg: CmdDense) -> None:
        self._cmd = np.concatenate([msg.q, msg.dq]).astype(np.float32).reshape(1, -1)
        r = msg.root_quat
        self._ref = np.array([r.x, r.y, r.z, r.w], dtype=np.float32)

    def get_target(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        return self._cmd, self._ref
