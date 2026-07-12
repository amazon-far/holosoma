"""Unitree SDK bridge implementation.

H1 GO2 traffic uses the official ``unitree_sdk2py`` DDS API (20 physical motor
slots), matching the inference side and Beyondmimic.  Other robot / message
types stay on the C++ ``unitree_interface`` binding.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge – sdk2py for H1 GO2, C++ binding for others."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _init_sdk_components(self):
        robot_type = self.robot.asset.robot_type

        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(
                f"Invalid robot type '{robot_type}'. "
                f"Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}"
            )

        self._use_sdk2py = robot_type == "h1"
        self.joint2motor = self._get_joint2motor_order(robot_type)
        self.motor2joint = self._invert_joint2motor(self.joint2motor)

        if self._use_sdk2py:
            self._init_sdk2py()
        else:
            self._init_binding()

    # ------------------------------------------------------------------
    #  sdk2py path (H1 GO2 – 20 physical motor slots)
    # ------------------------------------------------------------------

    def _init_sdk2py(self):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise ImportError(
                "unitree_sdk2py is required for H1 GO2 bridge communication."
            ) from e

        interface_name = self.bridge_config.interface or "eth0"
        domain_id = getattr(self.bridge_config, "domain_id", 0)
        ChannelFactoryInitialize(domain_id, interface_name)

        self._crc = CRC()
        self._sdk2py_low_cmd = unitree_go_msg_dds__LowCmd_()
        self._sdk2py_low_state = unitree_go_msg_dds__LowState_()
        self._init_go_low_cmd()

        self._low_state_publisher = ChannelPublisher("rt/lowstate", LowStateGo)
        self._low_state_publisher.Init()
        self._low_cmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmdGo)
        self._low_cmd_subscriber.Init(self._low_cmd_handler_sdk2py, 10)

        logger.info(
            "UnitreeSdk2Bridge initialized with unitree_sdk2py: "
            f"H1-GO2 (20 physical motors) on interface: {interface_name}"
        )

    def _init_go_low_cmd(self):
        """Mirrors Beyondmimic's init_cmd_go and inference's _init_go_low_cmd."""
        self._sdk2py_low_cmd.head[0] = 0xFE
        self._sdk2py_low_cmd.head[1] = 0xEF
        self._sdk2py_low_cmd.level_flag = 0xFF
        self._sdk2py_low_cmd.gpio = 0
        pos_stop = 2.146e9
        vel_stop = 16000.0
        weak_motors = set(range(10, 20))
        for motor_id, motor_cmd in enumerate(self._sdk2py_low_cmd.motor_cmd):
            motor_cmd.mode = 0x01 if motor_id in weak_motors else 0x0A
            motor_cmd.q = float(pos_stop)
            motor_cmd.dq = float(vel_stop)
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0
            motor_cmd.tau = 0.0

    def _low_cmd_handler_sdk2py(self, msg):
        self._sdk2py_low_cmd = msg

    # ------------------------------------------------------------------
    #  C++ binding path (G1, H1-2, GO2 quadruped, ...)
    # ------------------------------------------------------------------

    def _init_binding(self):
        from unitree_interface import (
            LowState,
            MessageType,
            MotorCommand,
            RobotType,
            UnitreeInterface,
            WirelessController,
        )

        robot_type = self.robot.asset.robot_type

        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        interface_name = self.bridge_config.interface or "eth0"
        self.interface = UnitreeInterface(
            interface_name,
            robot_type_map[robot_type],
            message_type_map[robot_type],
        )

        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()

    # ------------------------------------------------------------------
    #  Joint ↔ motor mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_joint2motor_order(robot_type):
        """Return motor indices for simulator joint-order arrays.

        H1 uses **physical** motor IDs (20 slots, ID 9 unused), matching
        Beyondmimic's dof_idx and the inference config.  Other robots return
        ``None`` (identity).
        """
        if robot_type == "h1":
            # Physical motor IDs (20 slots 0..19, skip 9)
            return [7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]
        return None

    @staticmethod
    def _invert_joint2motor(joint2motor):
        if joint2motor is None:
            return None
        num_slots = max(joint2motor) + 1
        motor2joint = [-1] * num_slots
        for joint_idx, motor_idx in enumerate(joint2motor):
            motor2joint[motor_idx] = joint_idx
        return motor2joint

    def _joints_to_motors(self, values):
        """Joint-order (19) → physical-motor-order (20). Unused slot 9 = 0.0."""
        if self.joint2motor is None:
            return values
        num_slots = max(self.joint2motor) + 1
        out = [0.0] * num_slots
        for joint_idx, motor_idx in enumerate(self.joint2motor):
            out[motor_idx] = float(values[joint_idx])
        return out

    def _motors_to_joints(self, values):
        """Physical-motor-order (20) → joint-order (19). Skips slot 9."""
        if self.motor2joint is None:
            return values
        out = [0.0] * len(self.joint2motor)
        for motor_idx, joint_idx in enumerate(self.motor2joint):
            if joint_idx >= 0:
                out[joint_idx] = float(values[motor_idx])
        return out

    # ------------------------------------------------------------------
    #  DDS callbacks / handlers
    # ------------------------------------------------------------------

    def low_cmd_handler(self, msg=None):
        """Poll for incoming commands (binding path only)."""
        if self._use_sdk2py:
            return  # sdk2py updates via subscriber callback
        self.low_cmd = self.interface.read_incoming_command()

    # ------------------------------------------------------------------
    #  Publish state
    # ------------------------------------------------------------------

    def publish_low_state(self):
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        if self._use_sdk2py:
            self._publish_low_state_sdk2py(
                positions, velocities, accelerations, actuator_forces,
                quaternion, gyro, acceleration,
            )
        else:
            self.low_state.motor.q = self._joints_to_motors(positions)
            self.low_state.motor.dq = self._joints_to_motors(velocities)
            self.low_state.motor.ddq = self._joints_to_motors(accelerations)
            self.low_state.motor.tau_est = self._joints_to_motors(actuator_forces)

            quat_array = quaternion.detach().cpu().numpy()
            self.low_state.imu.quat = [
                float(quat_array[0]),
                float(quat_array[1]),
                float(quat_array[2]),
                float(quat_array[3]),
            ]
            self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
            self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()
            self.low_state.tick = int(self.sim_time * 1e3)
            self.interface.publish_low_state(self.low_state)

    def _publish_low_state_sdk2py(
        self, positions, velocities, accelerations, actuator_forces,
        quaternion, gyro, acceleration,
    ):
        motor_q = self._joints_to_motors(positions)
        motor_dq = self._joints_to_motors(velocities)
        motor_ddq = self._joints_to_motors(accelerations)
        motor_tau = self._joints_to_motors(actuator_forces)

        for motor_id in range(20):
            m = self._sdk2py_low_state.motor_state[motor_id]
            m.q = float(motor_q[motor_id])
            m.dq = float(motor_dq[motor_id])
            m.ddq = float(motor_ddq[motor_id])
            m.tau_est = float(motor_tau[motor_id])

        quat_array = quaternion.detach().cpu().numpy()
        imu = self._sdk2py_low_state.imu_state
        imu.quaternion[:] = [float(quat_array[0]), float(quat_array[1]),
                             float(quat_array[2]), float(quat_array[3])]
        imu.gyroscope[:] = gyro.detach().cpu().numpy().tolist()
        imu.accelerometer[:] = acceleration.detach().cpu().numpy().tolist()

        self._sdk2py_low_state.tick = int(self.sim_time * 1e3)
        from unitree_sdk2py.utils.crc import CRC
        self._sdk2py_low_state.crc = self._crc.Crc(self._sdk2py_low_state)
        self._low_state_publisher.Write(self._sdk2py_low_state)

    def publish_wireless_controller(self):
        super().publish_wireless_controller()

        if self._use_sdk2py:
            return  # wireless remote is embedded in LowState for sdk2py

        if self.joystick is not None and hasattr(self, "interface"):
            self.interface.publish_wireless_controller(self.wireless_controller)

    # ------------------------------------------------------------------
    #  Compute torques
    # ------------------------------------------------------------------

    def compute_torques(self):
        if self._use_sdk2py:
            return self._compute_torques_sdk2py()

        if not (hasattr(self, "low_cmd") and self.low_cmd):
            return self.torques

        try:
            return self._compute_pd_torques(
                tau_ff=self._motors_to_joints(self.low_cmd.tau_ff),
                kp=self._motors_to_joints(self.low_cmd.kp),
                kd=self._motors_to_joints(self.low_cmd.kd),
                q_target=self._motors_to_joints(self.low_cmd.q_target),
                dq_target=self._motors_to_joints(self.low_cmd.dq_target),
            )
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise

    def _compute_torques_sdk2py(self):
        cmd = self._sdk2py_low_cmd

        # Build 20-element physical arrays from DDS message
        tau_ff_20 = [float(cmd.motor_cmd[i].tau) for i in range(20)]
        kp_20 = [float(cmd.motor_cmd[i].kp) for i in range(20)]
        kd_20 = [float(cmd.motor_cmd[i].kd) for i in range(20)]
        q_target_20 = [float(cmd.motor_cmd[i].q) for i in range(20)]
        dq_target_20 = [float(cmd.motor_cmd[i].dq) for i in range(20)]

        return self._compute_pd_torques(
            tau_ff=self._motors_to_joints(tau_ff_20),
            kp=self._motors_to_joints(kp_20),
            kd=self._motors_to_joints(kd_20),
            q_target=self._motors_to_joints(q_target_20),
            dq_target=self._motors_to_joints(dq_target_20),
        )
