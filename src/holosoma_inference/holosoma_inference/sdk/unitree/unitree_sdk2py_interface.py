"""Unitree robot interface using unitree_sdk2py (official Python SDK)."""

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class UnitreeSdk2pyInterface(BaseInterface):
    """Interface for Unitree robots using unitree_sdk2py (pure Python SDK)."""

    # Robot type to message type mapping
    ROBOT_MESSAGE_TYPE = {
        "g1": "hg",
        "g1_29dof": "hg",
        "h1": "go",
        "h1_2": "hg",
        "h1-2": "hg",
        "go2": "go",
        "go2_12dof": "go",
    }

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._low_state = None
        self._init_sdk2py()

    def _init_sdk2py(self):
        """Initialize unitree_sdk2py components."""
        try:
            from unitree_sdk2py.core.channel import (
                ChannelPublisher,
                ChannelSubscriber,
                ChannelFactoryInitialize,
            )
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise ImportError("unitree_sdk2py not found. Install with: pip install unitree_sdk2py") from e

        # Initialize DDS channel factory
        ChannelFactoryInitialize(self.domain_id, self.interface_str)

        # Determine message type based on robot
        robot_key = self.robot_config.robot.lower().replace("-", "_")
        msg_type = self.ROBOT_MESSAGE_TYPE.get(robot_key)
        if msg_type is None:
            raise ValueError(
                f"Unknown robot type: {self.robot_config.robot}. Supported: {list(self.ROBOT_MESSAGE_TYPE.keys())}"
            )

        # Import appropriate message types based on robot
        if msg_type == "hg":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

            self._low_cmd = unitree_hg_msg_dds__LowCmd_()
            self._LowCmd = LowCmd_
            self._LowState = LowState_
        else:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

            self._low_cmd = unitree_go_msg_dds__LowCmd_()
            self._LowCmd = LowCmd_
            self._LowState = LowState_

            # Initialize Go2-specific command header
            self._low_cmd.head[0] = 0xFE
            self._low_cmd.head[1] = 0xEF
            self._low_cmd.level_flag = 0xFF
            self._low_cmd.gpio = 0

        self._msg_type = msg_type
        self._crc = CRC()

        # Create publisher for low-level commands
        self._lowcmd_publisher = ChannelPublisher("rt/lowcmd", self._LowCmd)
        self._lowcmd_publisher.Init()

        # Create subscriber for low-level state
        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", self._LowState)
        self._lowstate_subscriber.Init(self._low_state_callback, 10)

        # Initialize motor commands
        for i in range(self.robot_config.num_motors):
            self._low_cmd.motor_cmd[i].mode = 0x01  # Enable motor (PMSM mode)
            self._low_cmd.motor_cmd[i].q = 0.0
            self._low_cmd.motor_cmd[i].dq = 0.0
            self._low_cmd.motor_cmd[i].kp = 0.0
            self._low_cmd.motor_cmd[i].kd = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0

        # GO2 SDK motor order differs from joint order
        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

        logger.info(
            f"Initialized unitree_sdk2py interface for {self.robot_config.robot} on interface {self.interface_str}"
        )

    def _low_state_callback(self, msg):
        """Callback for receiving low-level state from robot."""
        self._low_state = msg

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array."""
        if self._low_state is None:
            # Return zeros if no state received yet
            state_size = 3 + 4 + self.robot_config.num_joints * 2 + 3 + 3
            return np.zeros((1, state_size))

        state = self._low_state
        base_pos = np.zeros(3)
        quat = np.array(state.imu_state.quaternion)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu_state.gyroscope)

        # Extract motor positions and velocities
        motor_pos = np.array([state.motor_state[i].q for i in range(self.robot_config.num_motors)])
        motor_vel = np.array([state.motor_state[i].dq for i in range(self.robot_config.num_motors)])

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
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            self._low_cmd.motor_cmd[m_id].q = float(cmd_q[j_id])
            self._low_cmd.motor_cmd[m_id].dq = float(cmd_dq[j_id])
            self._low_cmd.motor_cmd[m_id].tau = float(cmd_tau[j_id])

            if kp_override is not None:
                self._low_cmd.motor_cmd[m_id].kp = float(kp_override[j_id]) * self._kp_level
            else:
                self._low_cmd.motor_cmd[m_id].kp = float(self.robot_config.motor_kp[j_id]) * self._kp_level

            if kd_override is not None:
                self._low_cmd.motor_cmd[m_id].kd = float(kd_override[j_id]) * self._kd_level
            else:
                self._low_cmd.motor_cmd[m_id].kd = float(self.robot_config.motor_kd[j_id]) * self._kd_level

        # Calculate CRC for message integrity
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)

        # Publish command
        self._lowcmd_publisher.Write(self._low_cmd)

    def get_joystick_msg(self):
        """Get wireless controller message."""
        # unitree_sdk2py doesn't have direct wireless controller access
        # You may need to subscribe to a wireless controller topic if needed
        return None

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
