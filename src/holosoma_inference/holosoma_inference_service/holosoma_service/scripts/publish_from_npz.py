#!/usr/bin/env python3
"""Stream a reference-motion NPZ to the WBT policy as a live ``CmdDense`` feed.

This drives ``policy_service_node`` exactly like a teleop source would: instead of
handing the NPZ to the policy via ``--task.ref-motion-path`` (which the injected
live target source ignores), we replay the NPZ frame-by-frame onto the dense
topic so the policy tracks the full trajectory. Useful for sim2sim "watch it
dance" tests without any real input device.

Frame → CmdDense mapping (matches the WBT policy's per-frame motion accessor):
    q        = joint_pos[t, 7:]    # drop root pos(3)+quat(4) → 29 joints
    dq       = joint_vel[t, 6:]    # drop root lin/ang vel(6) → 29 joints
    root_quat= body_quat_w[t, ref] # wxyz in NPZ → xyzw for geometry_msgs

Joint order in the reference-motion NPZs already matches the Unitree SDK / MuJoCo
29-DOF order, so no reordering is needed.

DDS note: the policy's ROS graph runs on a separate ROS_DOMAIN_ID from the
Unitree SDK link to the sim (the SDK hardwires domain 0; rclpy must not collide
with it). Run this publisher on the SAME ROS_DOMAIN_ID as policy_service_node — e.g.
``ROS_DOMAIN_ID=1``.

Usage:
    ROS_DOMAIN_ID=1 python publish_from_npz.py <motion.npz>
    ROS_DOMAIN_ID=1 python publish_from_npz.py <motion.npz> --rate 50 --loop
    ROS_DOMAIN_ID=1 python publish_from_npz.py <motion.npz> --ref-body torso_link
"""

from __future__ import annotations

import argparse

import numpy as np
import rclpy
from holosoma_msgs.msg import CmdDense
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

DENSE_TOPIC = "/holosoma/dense_tracking_command"


def _resolve_ref_body(npz, ref_body: str) -> int:
    """Index into body_pos_w/body_quat_w for the reference body (default pelvis)."""
    if "body_names" not in npz.files:
        return 0
    names = [str(n) for n in npz["body_names"].tolist()]
    if ref_body in names:
        return names.index(ref_body)
    print(f"  ref-body {ref_body!r} not in body_names; falling back to body 0 ({names[0]!r})")
    return 0


class NpzDensePublisher(Node):
    def __init__(self, npz_path: str, topic: str, rate_hz: float, loop: bool, ref_body: str):
        super().__init__("npz_dense_publisher")
        data = np.load(npz_path)

        jp = data["joint_pos"]  # (T, 36) = root_pos(3)+root_quat(4)+joints(29) OR (T, 29)
        jv = data["joint_vel"]  # (T, 35) = root_vel(6)+joints(29)            OR (T, 29)
        self._q = (jp[:, 7:] if jp.shape[1] > 29 else jp).astype(np.float32)
        self._dq = (jv[:, 6:] if jv.shape[1] > 29 else jv).astype(np.float32)

        ref_idx = _resolve_ref_body(data, ref_body)
        # body_quat_w is wxyz; geometry_msgs/Quaternion + the policy want xyzw.
        bq = data["body_quat_w"][:, ref_idx, :].astype(np.float32)  # (T, 4) wxyz
        self._quat_xyzw = bq[:, [1, 2, 3, 0]]

        self._joint_names = [str(n) for n in data["joint_names"].tolist()] if "joint_names" in data.files else []

        self._n = self._q.shape[0]
        self._loop = loop
        self._i = 0

        if rate_hz <= 0:
            fps = float(data["fps"]) if "fps" in data.files else 50.0
            rate_hz = fps
        self._rate_hz = rate_hz

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(CmdDense, topic, qos)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

        print(
            f"Loaded {self._n} frames from {npz_path}\n"
            f"  publishing {topic} at {rate_hz:.1f} Hz, ref_body={ref_body!r} (idx {ref_idx}), loop={loop}"
        )

    def _tick(self) -> None:
        if self._i >= self._n:
            if self._loop:
                self._i = 0
            else:
                print("Reached end of motion — holding last frame. Ctrl-C to stop.")
                self._timer.cancel()
                self._i = self._n - 1  # republish final frame once below
            return

        i = self._i
        msg = CmdDense()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self._joint_names:
            msg.joint_names = self._joint_names
        msg.q = self._q[i].tolist()
        msg.dq = self._dq[i].tolist()
        qx, qy, qz, qw = (float(v) for v in self._quat_xyzw[i])
        msg.root_quat.x, msg.root_quat.y, msg.root_quat.z, msg.root_quat.w = qx, qy, qz, qw
        self._pub.publish(msg)

        if i % 100 == 0:
            print(f"  frame {i}/{self._n}", flush=True)
        self._i += 1


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", help="Path to reference-motion NPZ.")
    parser.add_argument("--topic", default=DENSE_TOPIC, help=f"Dense topic (default: {DENSE_TOPIC}).")
    parser.add_argument("--rate", type=float, default=0.0, help="Publish rate Hz (default: NPZ fps, else 50).")
    parser.add_argument("--loop", action="store_true", help="Loop the motion instead of holding the last frame.")
    parser.add_argument("--ref-body", default="pelvis", help="Body for root_quat (default: pelvis).")
    ns = parser.parse_args()

    rclpy.init(args=args)
    node = NpzDensePublisher(ns.npz, ns.topic, ns.rate, ns.loop, ns.ref_body)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
