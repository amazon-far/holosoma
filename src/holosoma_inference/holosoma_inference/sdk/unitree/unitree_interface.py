"""Unitree robot interface.

H1 real GO2 traffic uses the official ``unitree_sdk2py`` DDS API so motor
commands are sent in the same 20-slot physical-ID layout as Unitree's examples
and the Beyondmimic deployment code.  Other Unitree paths keep using the
existing C++/pybind11 binding.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class _WirelessMsg(NamedTuple):
    """Small stand-in object matching the fields used by BaseInterface."""

    lx: float
    ly: float
    rx: float
    keys: int


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots.

    H1 real GO2 uses official unitree_sdk2py LowCmd/LowState messages.
    The C++ binding path is retained for sim and other robot/message types.
    """

    # Compact slot 9 maps to DDS physical slot 10 (slot 9 is unused on H1).
    _H1_COMPACT_INSERT_IDX = 9

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._backend = "sdk2py" if self._should_use_sdk2py() else "binding"
        self._low_state = None
        self._last_low_state_time = 0.0
        if self._backend == "sdk2py":
            self._init_sdk2py()
        else:
            self._init_binding()

    def _should_use_sdk2py(self) -> bool:
        """Use official SDK for all H1 GO2 traffic (sim and real).

        Both simulation (lo) and real robot use the same sdk2py path so
        sim results are directly transferable to hardware.
        """
        return (
            self.robot_config.robot.lower() == "h1"
            and self.robot_config.message_type.upper() == "GO2"
        )

    def _init_sdk2py(self):
        """Initialize official unitree_sdk2py DDS channels for H1 GO2."""
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise ImportError("unitree_sdk2py is required for H1 real GO2 communication.") from e

        ChannelFactoryInitialize(self.domain_id, self.interface_str)
        self._crc = CRC()
        self._low_cmd = unitree_go_msg_dds__LowCmd_()
        self._low_state = unitree_go_msg_dds__LowState_()
        self._init_go_low_cmd()

        self._lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self._lowcmd_publisher.Init()
        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self._lowstate_subscriber.Init(self._low_state_handler, 10)

        logger.info(
            "UnitreeInterface initialized with unitree_sdk2py: "
            f"H1-GO2 (20 physical motors) on interface: {self.interface_str}"
        )
        self._wait_for_low_state(timeout_s=2.0)

    def _low_state_handler(self, msg):
        self._low_state = msg
        self._last_low_state_time = time.perf_counter()

    def _wait_for_low_state(self, timeout_s: float):
        deadline = time.perf_counter() + timeout_s
        while self._last_low_state_time == 0.0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        if self._last_low_state_time == 0.0:
            logger.warning("No H1 low_state received yet; get_low_state() will return zeros until DDS data arrives.")

    def _init_go_low_cmd(self):
        """Match Unitree/Beyondmimic GO LowCmd initialization."""
        self._low_cmd.head[0] = 0xFE
        self._low_cmd.head[1] = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio = 0
        pos_stop = self.robot_config.unitree_legged_const.get("PosStopF", 2.146e9)
        vel_stop = self.robot_config.unitree_legged_const.get("VelStopF", 16000.0)
        # H1 GO LowCmd uses a different mode for the weak/parallel motors.
        # This matches Beyondmimic's weak_motor: [10, 11, ..., 19].
        weak_motors = set(range(10, 20))
        for motor_id, motor_cmd in enumerate(self._low_cmd.motor_cmd):
            motor_cmd.mode = 0x01 if motor_id in weak_motors else 0x0A
            motor_cmd.q = float(pos_stop)
            motor_cmd.dq = float(vel_stop)
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0
            motor_cmd.tau = 0.0

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

        # GO2 quadruped SDK motor order differs from joint order. H1 uses
        # GO2 messages with physical motor IDs directly (no compact mapping).
        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    # ------------------------------------------------------------------
    # 19-compact ↔ 20-physical helpers (H1 only)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_to_physical(compact_array: np.ndarray) -> np.ndarray:
        """Expand a 19-element compact array to 20-element physical array.

        The C++ binding skips physical motor ID 9, so compact indices 0..8
        map to physical 0..8, and compact indices 9..18 map to physical 10..19.
        We insert 0.0 at physical index 9 (the unused slot).
        """
        if compact_array.shape[0] != 19:
            return compact_array
        physical = np.zeros(20, dtype=compact_array.dtype)
        physical[:9] = compact_array[:9]          # compact 0..8 → physical 0..8
        physical[9] = 0.0                          # physical 9  (unused)
        physical[10:] = compact_array[9:]          # compact 9..18 → physical 10..19
        return physical

    @staticmethod
    def _physical_to_compact(physical_array: np.ndarray) -> np.ndarray:
        """Compact a 20-element physical array to 19-element compact array.

        Reverse of _compact_to_physical: removes physical index 9 (unused).
        """
        if physical_array.shape[0] != 20:
            return physical_array
        compact = np.zeros(19, dtype=physical_array.dtype)
        compact[:9] = physical_array[:9]           # physical 0..8 → compact 0..8
        compact[9:] = physical_array[10:]          # physical 10..19 → compact 9..18
        return compact

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array.

        Returns:
            np.ndarray with shape (1, 3+4+N+3+3+N) containing:
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        if self._backend == "sdk2py":
            return self._get_low_state_sdk2py()

        state = self.unitree_interface.read_low_state()
        base_pos = np.zeros(3)
        quat = np.array(state.imu.quat)
        # Binding returns 19 compact motor values. Expand to 20 physical.
        motor_pos_compact = np.array(state.motor.q)
        motor_vel_compact = np.array(state.motor.dq)
        motor_pos = self._compact_to_physical(motor_pos_compact)
        motor_vel = self._compact_to_physical(motor_vel_compact)

        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu.omega)

        joint_pos = np.zeros(self.robot_config.num_joints)
        joint_vel = np.zeros(self.robot_config.num_joints)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]  # physical motor ID (0..19, excluding 9)
            joint_pos[j_id] = float(motor_pos[m_id])
            joint_vel[j_id] = float(motor_vel[m_id])

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def _get_low_state_sdk2py(self) -> np.ndarray:
        state = self._low_state
        base_pos = np.zeros(3)
        imu_state = state.imu_state
        quat = np.array(imu_state.quaternion)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(imu_state.gyroscope)

        motor_pos = np.array([motor.q for motor in state.motor_state], dtype=np.float64)
        motor_vel = np.array([motor.dq for motor in state.motor_state], dtype=np.float64)

        joint_pos = np.zeros(self.robot_config.num_joints)
        joint_vel = np.zeros(self.robot_config.num_joints)
        for j_id, m_id in enumerate(self.robot_config.joint2motor):
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
        """Send low-level command to robot.

        All input arrays (cmd_q, cmd_dq, cmd_tau) are in joint order (19 elements).
        KP/KD overrides are also in joint order and get remapped to physical
        motor order via joint2motor.
        """
        if self._backend == "sdk2py":
            self._send_low_command_sdk2py(cmd_q, cmd_dq, cmd_tau, kp_override, kd_override)
            return

        # 20-element physical-motor-order targets
        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
        cmd_kd = np.zeros(self.robot_config.num_motors) if kd_override is not None else None

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]  # physical motor ID
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if cmd_kp is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if cmd_kd is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        # Compact 20-element physical → 19-element compact for the binding
        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(self._physical_to_compact(cmd_q_target))
        cmd.dq_target = list(self._physical_to_compact(cmd_dq_target))
        cmd.tau_ff = list(self._physical_to_compact(cmd_tau_target))

        motor_kp = np.array(cmd_kp if cmd_kp is not None else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if cmd_kd is not None else self.robot_config.motor_kd)
        cmd.kp = list(self._physical_to_compact(motor_kp * self._kp_level))
        cmd.kd = list(self._physical_to_compact(motor_kd * self._kd_level))

        self.unitree_interface.write_low_command(cmd)

    def _send_low_command_sdk2py(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.array(self.robot_config.motor_kp, dtype=np.float64)
        cmd_kd = np.array(self.robot_config.motor_kd, dtype=np.float64)

        for j_id, m_id in enumerate(self.robot_config.joint2motor):
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if kp_override is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if kd_override is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        # Diagnostic: log actual KP values sent (once)
        if not hasattr(self, "_kp_logged"):
            self._kp_logged = True
            msg = ", ".join(f"m{m}:kp={cmd_kp[m]*self._kp_level:.0f}" for m in [0,2,5,7,10,12,16])
            logger.info(f"Real robot KP sent: {msg}  (level={self._kp_level})")

        for motor_id, motor_cmd in enumerate(self._low_cmd.motor_cmd):
            motor_cmd.q = float(cmd_q_target[motor_id])
            motor_cmd.dq = float(cmd_dq_target[motor_id])
            motor_cmd.tau = float(cmd_tau_target[motor_id])
            motor_cmd.kp = float(cmd_kp[motor_id] * self._kp_level)
            motor_cmd.kd = float(cmd_kd[motor_id] * self._kd_level)

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._lowcmd_publisher.Write(self._low_cmd)

    def get_joystick_msg(self):
        """Get wireless controller message."""
        if self._backend == "sdk2py":
            data = getattr(self._low_state, "wireless_remote", None)
            if data is None:
                return None
            import struct

            keys = int(data[2]) | (int(data[3]) << 8)
            lx = struct.unpack("<f", bytes(data[4:8]))[0]
            rx = struct.unpack("<f", bytes(data[8:12]))[0]
            ly = struct.unpack("<f", bytes(data[20:24]))[0]
            return _WirelessMsg(lx=lx, ly=ly, rx=rx, keys=keys)
        return self.unitree_interface.read_wireless_controller()

    def get_joystick_key(self, wc_msg=None):
        """Get current key from joystick message."""
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    def get_raw_motor_state(self) -> dict:
        """Get raw motor/IMU state as a dict for diagnostics."""
        if self._backend == "sdk2py":
            state = self._low_state
            return {
                "q": [float(m.q) for m in state.motor_state],
                "dq": [float(m.dq) for m in state.motor_state],
                "tau_est": [float(m.tau_est) for m in state.motor_state],
                "temperature": [int(m.temperature) for m in state.motor_state],
                "imu_quat": [float(v) for v in state.imu_state.quaternion],
                "imu_omega": [float(v) for v in state.imu_state.gyroscope],
                "imu_accel": [float(v) for v in state.imu_state.accelerometer],
            }

        state = self.unitree_interface.read_low_state()
        return {
            "q": list(state.motor.q),
            "dq": list(state.motor.dq),
            "tau_est": list(state.motor.tau_est),
            "voltage": list(state.motor.voltage),
            "temperature": list(state.motor.temperature),
            "imu_quat": list(state.imu.quat),
            "imu_omega": list(state.imu.omega),
            "imu_accel": list(state.imu.accel),
        }

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
