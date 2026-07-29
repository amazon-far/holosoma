"""Unitree robot interface using C++/pybind11 binding."""

from __future__ import annotations

import numpy as np

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots using C++/pybind11 binding."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str: str | None = None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._init_binding()

    def _init_binding(self):
        """Initialize C++/pybind11 binding."""
        try:
            import unitree_interface
        except ImportError as e:
            raise ImportError("unitree_interface python binding not found.") from e

        robot_type_map = {
            "G1": unitree_interface.RobotType.G1,
            "H1": unitree_interface.RobotType.H1,
            "H1_2": unitree_interface.RobotType.H1_2,
            "GO2": unitree_interface.RobotType.GO2,
        }
        message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}

        self.unitree_interface = unitree_interface.create_robot(
            self.interface_str,
            robot_type_map[self.robot_config.robot.upper()],
            message_type_map[self.robot_config.message_type.upper()],
        )
        self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)

        # GO2 SDK motor order differs from joint order
        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array."""
        state = self.unitree_interface.read_low_state()
        base_pos = np.zeros(3)
        quat = np.array(state.imu.quat)
        motor_pos = np.array(state.motor.q)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu.omega)
        motor_vel = np.array(state.motor.dq)

        joint_pos = np.zeros(self.robot_config.num_joints)
        joint_vel = np.zeros(self.robot_config.num_joints)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            joint_pos[j_id] = float(motor_pos[m_id])
            joint_vel[j_id] = float(motor_vel[m_id])

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Send low-level command to robot."""
        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
        cmd_kd = np.zeros(self.robot_config.num_motors) if kd_override is not None else None

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if cmd_kp is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if cmd_kd is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q_target)
        cmd.dq_target = list(cmd_dq_target)
        cmd.tau_ff = list(cmd_tau_target)

        motor_kp = np.array(cmd_kp if cmd_kp is not None else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if cmd_kd is not None else self.robot_config.motor_kd)
        cmd.kp = list(motor_kp * self._kp_level)
        cmd.kd = list(motor_kd * self._kd_level)

        self.unitree_interface.write_low_command(cmd)

    def get_joystick_msg(self):
        """Get wireless controller message."""
        return self.unitree_interface.read_wireless_controller()

    def get_joystick_key(self, wc_msg=None):
        """Get current key from joystick message."""
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    @property
    def kp_level(self):
        """Get proportional gain level."""
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        """Set proportional gain level."""
        self._kp_level = value

    @property
    def kd_level(self):
        """Get derivative gain level."""
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        """Set derivative gain level."""
        self._kd_level = value
