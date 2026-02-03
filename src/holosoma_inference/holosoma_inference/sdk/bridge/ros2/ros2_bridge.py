import threading
from dataclasses import dataclass

import numpy as np
import rclpy
from far_msgs.msg import PolicyActions, RobotState
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.node import Node
from std_msgs.msg import Header

from holosoma_inference.sdk.bridge.base import BasicSdk2Bridge


@dataclass
class MotorCommand:
    """Motor command structure compatible with base_sim.py"""

    q: float = 0.0  # Position command
    dq: float = 0.0  # Velocity command
    tau: float = 0.0  # Torque command
    kp: float = 0.0  # Position gain
    kd: float = 0.0  # Velocity gain


class LowCmd:
    """Low-level command structure compatible with base_sim.py"""

    def __init__(self, num_motors):
        self.motor_cmd: list[MotorCommand] = [MotorCommand() for _ in range(num_motors)]


class ROS2Bridge(BasicSdk2Bridge):
    def __init__(self, mj_model, mj_data, robot_config, lcm=None):
        # Initialize BasicSdk2Bridge
        super().__init__(mj_model, mj_data, robot_config, lcm)

    def _init_sdk_components(self):
        """Initialize ROS2-specific components."""
        if not rclpy.ok():
            rclpy.init()

        self._ros2_node = Node("mujoco_robot_state_publisher")
        self.robot_state_pub = self._ros2_node.create_publisher(RobotState, "robot_state", 10)

        # Subscribe to PolicyActions messages
        self.policy_actions_sub = self._ros2_node.create_subscription(
            PolicyActions,
            "policy_actions",  # Topic name
            self.low_cmd_handler,
            10,
        )

        # Thread for rclpy.spin()
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self._ros2_node,), daemon=True)
        self.spin_thread.start()

        # Initialize low_cmd with proper structure
        self.low_cmd = LowCmd(self.num_motor)

        # Get robot's control gains
        self.Kps: list[float] = self.robot.MOTOR_KP
        self.Kds: list[float] = self.robot.MOTOR_KD

    def low_cmd_handler(self, msg):
        """Handle PolicyActions messages and convert to low-level commands."""
        if msg is None:
            return

        # Ensure arrays have correct length
        num_motors = min(self.num_motor, len(msg.joint_positions))

        # Update motor commands based on control mode
        for i in range(num_motors):
            motor_cmd = self.low_cmd.motor_cmd[i]

            if msg.control_mode == "position":
                # Position control mode
                motor_cmd.q = msg.joint_positions[i] if i < len(msg.joint_positions) else 0.0
                motor_cmd.dq = 0.0  # No velocity feedforward in position mode
                motor_cmd.tau = 0.0  # No torque feedforward in position mode
                motor_cmd.kp = self.Kps[i]
                motor_cmd.kd = self.Kds[i]
            else:
                raise NotImplementedError

    def publish_low_state(self):
        """Publish low-level state via ROS2."""
        msg = RobotState()

        # Header
        msg.header = Header()
        msg.header.stamp = self._ros2_node.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        # Base pose
        msg.base_pose = Pose()
        msg.base_pose.position = Point(x=self.mj_data.qpos[0], y=self.mj_data.qpos[1], z=self.mj_data.qpos[2])
        msg.base_pose.orientation = Quaternion(
            x=self.mj_data.qpos[4], y=self.mj_data.qpos[5], z=self.mj_data.qpos[6], w=self.mj_data.qpos[3]
        )

        # Base velocities
        msg.base_angular_velocity = Vector3(x=self.mj_data.qvel[3], y=self.mj_data.qvel[4], z=self.mj_data.qvel[5])

        # Projected gravity (gravity vector in body frame)
        # Get quaternion - use the same values as in base_pose.orientation for consistency
        qw = self.mj_data.qpos[3]
        qx = self.mj_data.qpos[4]
        qy = self.mj_data.qpos[5]
        qz = self.mj_data.qpos[6]

        # Compute rotation matrix from quaternion
        R = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

        # Transform gravity vector from world to body frame
        # Use normalized gravity vector [0, 0, -1] to match training data
        gravity_world = np.array([0, 0, -1])
        gravity_body = R.T @ gravity_world

        msg.projected_gravity = Vector3(x=gravity_body[0], y=gravity_body[1], z=gravity_body[2])

        # Joint states
        msg.joint_positions = self.mj_data.qpos[7 : 7 + self.num_motor].tolist()
        msg.joint_velocities = self.mj_data.qvel[6 : 6 + self.num_motor].tolist()

        # Joint torques from actuator forces
        msg.joint_torques = self.mj_data.actuator_force[: self.num_motor].tolist()

        self.robot_state_pub.publish(msg)
