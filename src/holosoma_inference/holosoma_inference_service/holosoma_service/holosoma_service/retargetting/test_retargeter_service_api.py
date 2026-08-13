"""ROS2-native integration test for the retargeter service API.

Exercises the *contract* external consumers depend on, rather than co-running
two services: spin the real ``RetargeterNode`` and drive it
over actual ROS2 topics, asserting the ``CmdSMPLH -> CmdDense`` surface. This is
the one service node that is a pure topic-in/topic-out transform (no ONNX, no
robot SDK), so the whole pub/sub + serialization + callback path is real.

Two salient cases:

1. Valid ``CmdSMPLH`` in -> one ``CmdDense`` out with the documented shape
   (29-DoF q/dq, finite, normalized root quat).
2. ``valid=False`` in -> no ``CmdDense`` out (the "no usable tracking this
   frame; consumers skip" contract), verified only after the pipeline has been
   shown to work so the negative assertion is meaningful.

Skips cleanly when the ROS2 / IK stack isn't importable (host env); a built
colcon workspace (or a ROS2 CI job) has rclpy + holosoma_msgs + mink and the
``g1-smpl`` retargeter entry point registered.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("geometry_msgs")
pytest.importorskip("holosoma_msgs")
pytest.importorskip("mink")

import rclpy
from geometry_msgs.msg import Pose
from holosoma_msgs.msg import CmdDense, CmdSMPLH
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_service.retargetting import available_retargeters
from holosoma_service.retargetting.retargeter_node import IN_TOPIC, OUT_TOPIC, RetargeterNode
from holosoma_service.retargetting.smpl_retargeter import JOINT_NAMES

# Shipped G1 MJCF the retargeter IK loads, relative to the holosoma repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
_G1_XML = os.path.join(
    _REPO_ROOT, "src", "holosoma_retargeting", "holosoma_retargeting", "models", "g1", "g1_29dof.xml"
)

_EXPECTED_DOF = 29
# Match the node's QoS exactly (BEST_EFFORT, depth 1) or the endpoints won't pair.
_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

pytestmark = [
    pytest.mark.skipif(not Path(_G1_XML).exists(), reason="shipped g1_29dof.xml not present"),
    pytest.mark.skipif(
        "g1-smpl" not in available_retargeters(),
        reason="g1-smpl retargeter entry point not registered (package not installed)",
    ),
]


def _make_smplh(valid: bool = True) -> CmdSMPLH:
    """A non-degenerate SMPL-H frame: identity orientations, joints spread along
    the vertical axis so pelvis->ankle separation (the height-ratio basis) is
    nonzero and the IK gets a well-posed target.

    ``joint_poses`` is populated even when ``valid=False`` so the invalid frame
    differs from the valid one *only* in the ``valid`` flag. The node guard is
    ``if not msg.valid or not msg.joint_poses`` — leaving poses empty would let
    the empty-poses branch suppress output on its own, so the negative test
    could not distinguish the ``valid`` contract from a degenerate payload.
    """
    msg = CmdSMPLH()
    msg.valid = valid
    msg.joint_names = list(JOINT_NAMES)
    poses = []
    n = len(JOINT_NAMES)
    for i in range(n):
        p = Pose()
        p.orientation.w = 1.0  # identity quaternion (xyzw)
        p.position.y = 1.0 - i / n  # descend 1.0 -> ~0 so positions are distinct
        poses.append(p)
    msg.joint_poses = poses
    return msg


class _Harness(Node):
    """External peer: publishes CmdSMPLH, records CmdDense — stands in for a downstream consumer."""

    def __init__(self) -> None:
        super().__init__("retargeter_api_test_harness")
        self.pub = self.create_publisher(CmdSMPLH, IN_TOPIC, _QOS)
        self.received: list[CmdDense] = []
        self.create_subscription(CmdDense, OUT_TOPIC, self.received.append, _QOS)


@pytest.fixture
def pipeline():
    """Real RetargeterNode + harness on one executor. Publishing is best-effort,
    so callers republish each spin until an output arrives (or timeout)."""
    rclpy.init()
    node = RetargeterNode(urdf_path=_G1_XML, rl_rate_hz=50.0, joint_names=[], retargeter="g1-smpl")
    harness = _Harness()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(harness)

    def pump(msg: CmdSMPLH, *, until_received: bool, timeout_s: float = 5.0) -> None:
        """Publish `msg` each spin. If until_received, stop once output arrives;
        otherwise pump for the full timeout (used to assert no output)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            harness.pub.publish(msg)
            executor.spin_once(timeout_sec=0.05)
            if until_received and harness.received:
                return

    try:
        yield harness, pump
    finally:
        executor.shutdown()
        node.destroy_node()
        harness.destroy_node()
        rclpy.shutdown()


def test_valid_smplh_yields_wellformed_dense(pipeline):
    harness, pump = pipeline
    pump(_make_smplh(valid=True), until_received=True)

    assert harness.received, "no CmdDense published for a valid CmdSMPLH within timeout"
    out = harness.received[0]

    assert len(out.q) == _EXPECTED_DOF
    assert len(out.dq) == _EXPECTED_DOF
    assert np.all(np.isfinite(out.q)), "q contains non-finite values"
    assert np.all(np.isfinite(out.dq)), "dq contains non-finite values"

    quat = np.array([out.root_quat.x, out.root_quat.y, out.root_quat.z, out.root_quat.w])
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3), f"root_quat not normalized: {quat}"


def test_invalid_smplh_publishes_nothing(pipeline):
    harness, pump = pipeline
    # First prove the pipeline is live (discovery complete), else the negative
    # assertion below would pass trivially.
    pump(_make_smplh(valid=True), until_received=True)
    assert harness.received, "pipeline never produced output; cannot trust the negative case"

    harness.received.clear()
    pump(_make_smplh(valid=False), until_received=False, timeout_s=1.5)
    assert not harness.received, "CmdDense published for an invalid (valid=False) frame"
