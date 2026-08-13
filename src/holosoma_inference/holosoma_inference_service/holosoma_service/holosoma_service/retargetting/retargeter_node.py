"""RetargeterNode: CmdSMPLH in -> Retargeter (G1SmplRetargeter) -> CmdDense out.

Decouples retargeting (variable-latency mink IK) from the control loop: it
solves on its own subscription rate and publishes a dense per-joint target
in holosoma (URDF/Mujoco 29-DOF) convention. Per-policy adapters subscribe to
CmdDense and feed their policy.
"""

from __future__ import annotations

import numpy as np
import rclpy
import tyro
from holosoma_msgs.msg import CmdDense, CmdSMPLH
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_service.retargetting import create_retargeter
from holosoma_service.retargetting.smpl_retargeter import JOINT_NAMES, NUM_JOINTS, Retargeter

DEFAULT_RETARGETER = "g1-smpl"

IN_TOPIC = "/holosoma/smplh_command"
OUT_TOPIC = "/holosoma/dense_tracking_command"
_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def _to_transforms(msg: CmdSMPLH) -> np.ndarray:
    """CmdSMPLH -> (24, 7) [xyz, qxyzw], canonical order, identity for missing."""
    out = np.zeros((NUM_JOINTS, 7))
    out[:, 6] = 1.0
    for name, pose in zip(msg.joint_names, msg.joint_poses):
        if (i := _IDX.get(name)) is not None:
            p, q = pose.position, pose.orientation
            out[i] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return out


class RetargeterNode(Node):
    def __init__(self, urdf_path: str, rl_rate_hz: float, joint_names: list[str], retargeter: str = DEFAULT_RETARGETER):
        super().__init__("holosoma_retargeter")
        self._rt: Retargeter = create_retargeter(retargeter, urdf_path=urdf_path, dt=1.0 / rl_rate_hz)
        self._joint_names = joint_names
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(CmdDense, OUT_TOPIC, qos)
        self.create_subscription(CmdSMPLH, IN_TOPIC, self._cb, qos)

    def _cb(self, msg: CmdSMPLH) -> None:
        if not msg.valid or not msg.joint_poses:
            return
        q, dq, wxyz = self._rt.retarget(_to_transforms(msg))
        out = CmdDense()
        out.header.stamp = self.get_clock().now().to_msg()
        out.joint_names = self._joint_names
        out.q = np.asarray(q, dtype=np.float32).tolist()
        out.dq = np.asarray(dq, dtype=np.float32).tolist()
        out.root_quat.x, out.root_quat.y, out.root_quat.z, out.root_quat.w = (
            float(wxyz[1]),
            float(wxyz[2]),
            float(wxyz[3]),
            float(wxyz[0]),
        )
        self._pub.publish(out)


def main(args=None) -> None:
    # Under `ros2 run` / a launch file, argv carries ROS args (e.g.
    # "--ros-args -r __node:=..."). tyro would reject those, so strip them
    # before parsing this node's own CLI, and hand the full argv to
    # rclpy.init (which consumes the ROS args).
    import sys

    from rclpy.utilities import remove_ros_args

    ros_argv = sys.argv if args is None else args
    cli_argv = remove_ros_args(args=ros_argv)[1:]  # drop argv[0]

    def run(urdf_path: str, rl_rate_hz: float = 50.0, retargeter: str = DEFAULT_RETARGETER):
        rclpy.init(args=ros_argv)
        node = RetargeterNode(urdf_path, rl_rate_hz, joint_names=[], retargeter=retargeter)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    tyro.cli(run, args=cli_argv)


if __name__ == "__main__":
    main()
