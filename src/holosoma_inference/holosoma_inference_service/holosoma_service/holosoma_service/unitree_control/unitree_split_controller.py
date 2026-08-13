"""Split-body Unitree controller (arm_sdk + loco)."""

from __future__ import annotations

import numpy as np
import rclpy
import tyro
from holosoma_msgs.msg import CmdExoskeleton, Heartbeat
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from holosoma_inference.sdk.unitree_high_level import make_mp_arm_client, make_mp_loco_client
from holosoma_inference.utils.rate import RateLimiter

EXOSKELETON_TOPIC = "/holosoma/exoskeleton_command"
CMD_TOPIC = "/holosoma/holosoma_executed_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
CONTROL_RATE_HZ = 50.0
HEARTBEAT_EVERY = 10  # ticks -> 5 Hz at 50 Hz control

# 14-DoF arm command order: [left(7), right(7)], matching CmdExoskeleton and track_dual_arm.
_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINT_NAMES = [f"{side}_{j}" for side in ("left", "right") for j in _ARM_JOINTS]

# Full 29-DoF G1 joint order. The split-body controller only commands the 14
# arm joints; the executed-cmd feedback is zero-padded to 29 so it shares the
# schema with the policy backend's /holosoma/holosoma_executed_cmd.
FULL_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# The 14 arm joints occupy indices 15..28 in FULL_DOF_NAMES.
_ARM_SLICE = slice(15, 29)


class UnitreeSplitControllerNode(Node):
    def __init__(self, arm, loco) -> None:
        super().__init__("unitree_split_controller")
        self._arm = arm
        self._loco = loco
        self._latest: CmdExoskeleton | None = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CmdExoskeleton, EXOSKELETON_TOPIC, self._cb, qos)
        self._cmd_pub = self.create_publisher(JointState, CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        self._rate = RateLimiter(CONTROL_RATE_HZ)

    def run(self) -> None:
        logger.info(f"control loop @ {CONTROL_RATE_HZ:.0f} Hz")
        tick = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)  # drain pending callbacks (newest cmd)
            cmd = self._latest
            if cmd is not None:
                q_target = np.concatenate([cmd.q_left_arm, cmd.q_right_arm])
                if self._arm is not None:
                    self._arm.track_dual_arm(q_target)
                if self._loco is not None:
                    v = cmd.base_velocity
                    self._loco.set_velocity(v.linear.x, v.linear.y, v.angular.z)
                self._publish_joint_command(q_target)
            if tick % HEARTBEAT_EVERY == 0:
                self._publish_heartbeat("running" if cmd is not None else "waiting_for_cmd")
            tick += 1
            self._rate.sleep()

    def _cb(self, msg: CmdExoskeleton) -> None:
        self._latest = msg

    def _publish_joint_command(self, q_target: np.ndarray) -> None:
        # Zero-pad the 14-DoF arm command into the full 29-DoF layout so the
        # topic shares its schema with the policy backend.
        full = np.zeros(len(FULL_DOF_NAMES))
        full[_ARM_SLICE] = q_target
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = FULL_DOF_NAMES, full.tolist()
        self._cmd_pub.publish(msg)

    def _publish_heartbeat(self, status: str) -> None:
        msg = Heartbeat()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.robot_connected = self._arm is not None or self._loco is not None
        msg.control_mode = 0
        msg.status = status
        self._hb_pub.publish(msg)


def main(iface: str = "eth0", no_arms: bool = False, no_loco: bool = False, arm_kp_scale: float = 2.0) -> None:
    arm = loco = None
    if not no_loco:
        logger.info("starting loco client …")
        loco = make_mp_loco_client(iface=iface)
        loco.start()
        loco.set_walk_mode()
    if not no_arms:
        logger.info("starting arm client …")
        arm = make_mp_arm_client(iface=iface, motion_mode=True, arm_kp_scale=arm_kp_scale)
        arm.ctrl_dual_arm_initialization_pose()
        arm.speed_gradual_max()

    rclpy.init()
    controller = UnitreeSplitControllerNode(arm=arm, loco=loco)
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        controller.destroy_node()
        rclpy.shutdown()
        if loco is not None:
            loco.close()  # type: ignore[attr-defined]
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]


def _cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    _cli()
