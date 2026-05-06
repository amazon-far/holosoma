"""Unitree robot interface using C++/pybind11 binding."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface
from holosoma_inference.sdk.dampening import Dampener
from holosoma_inference.sdk.send_log import SendLogger


def _load_joint_limits_from_mjcf(robot_config: RobotConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort MJCF joint-limit lookup for the dampening clip knob.

    The Unitree hardware path has no joint-range info in RobotConfig today,
    so without this helper HOLOSOMA_Q_LIMIT_SCALE is a no-op on real
    robots. Resolve the same 29dof MJCF the retargeter uses and pull the
    limits in dof_names order. Returns (lo, hi) arrays or None if the MJCF
    isn't reachable (host without holosoma_retargeting installed, etc.).
    """
    try:
        import mujoco
    except ImportError:
        return None
    # Look first at HOLOSOMA_MUJOCO_MJCF (same override the sim backend
    # uses), then at robot_config.urdf_path if it's a .xml, then the
    # shipped MJCF via holosoma_retargeting.
    override = os.environ.get("HOLOSOMA_MUJOCO_MJCF")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    urdf = getattr(robot_config, "urdf_path", None)
    if urdf and urdf.endswith(".xml"):
        candidates.append(Path(urdf))
    try:
        import holosoma_retargeting  # type: ignore[import-not-found]

        candidates.append(
            Path(holosoma_retargeting.__file__).parent
            / "models"
            / robot_config.robot
            / f"{robot_config.robot_type}.xml"
        )
    except Exception as exc:
        logger.debug("holosoma_retargeting path probe failed: {}", exc)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            logger.debug("MJCF load failed at {}: {}", path, exc)
            continue
        lo = np.full(robot_config.num_joints, -np.inf)
        hi = np.full(robot_config.num_joints, np.inf)
        any_named = False
        for j_id, name in enumerate(robot_config.dof_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            any_named = True
            if bool(model.jnt_limited[jid]):
                lo[j_id] = float(model.jnt_range[jid][0])
                hi[j_id] = float(model.jnt_range[jid][1])
        if any_named:
            return lo, hi
    return None


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots using C++/pybind11 binding."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        limits = _load_joint_limits_from_mjcf(robot_config)
        if limits is not None:
            self._dampener = Dampener(joint_limits_lo=limits[0], joint_limits_hi=limits[1])
        else:
            self._dampener = Dampener()
        self._send_logger = SendLogger()
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

        # Participant diagnostics: confirm which processes end up on the same
        # DDS domain / interface. Two unitree_interface participants on the
        # same domain on the same NIC will create competing LowCommandWriter
        # endpoints, and the robot firmware latches onto one — the other's
        # write_low_command calls fall on the floor with no visible error.
        import os as _os

        cyclonedds_uri = _os.environ.get("CYCLONEDDS_URI", "<unset>")
        cyclonedds_domain = _os.environ.get("CYCLONEDDS_DOMAIN", "<unset>")
        ros_domain_id = _os.environ.get("ROS_DOMAIN_ID", "<unset>")
        # The Python binding's create_robot() does NOT accept a domain_id;
        # the C++ side hardcodes 0. Log the Python-side arg so callers can
        # see the divergence. Gated on HOLOSOMA_DDS_DIAG (default on) so
        # callers can silence the two participant-diagnostic lines without
        # losing the rest of the boot log.
        _dds_diag = _os.environ.get("HOLOSOMA_DDS_DIAG", "1") not in ("0", "false", "False", "")
        if _dds_diag:
            logger.info(
                "[unitree_interface] pid={} ppid={} iface={} "
                "python_domain_id_arg={} C++_domain_id=0 (hardcoded) "
                "ROS_DOMAIN_ID={} CYCLONEDDS_DOMAIN={} CYCLONEDDS_URI_set={}",
                _os.getpid(),
                _os.getppid(),
                self.interface_str,
                self.domain_id,
                ros_domain_id,
                cyclonedds_domain,
                cyclonedds_uri != "<unset>",
            )

        self.unitree_interface = unitree_interface.create_robot(
            self.interface_str,
            robot_type_map[self.robot_config.robot.upper()],
            message_type_map[self.robot_config.message_type.upper()],
        )
        self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)

        if _dds_diag:
            logger.info(
                "[unitree_interface] pid={} create_robot + set_control_mode(PR) complete",
                _os.getpid(),
            )

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
        # Apply the dampening shim FIRST — in joint-space — before remapping
        # to motor order. Keeps q_slew/q_limit/blend_alpha operating on the
        # same axes as the robot_config joint limits.
        motor_kp_cfg = np.asarray(
            kp_override if kp_override is not None else self.robot_config.motor_kp,
            dtype=np.float64,
        )
        motor_kd_cfg = np.asarray(
            kd_override if kd_override is not None else self.robot_config.motor_kd,
            dtype=np.float64,
        )
        q_d, dq_d, tau_d, kp_d, kd_d = self._dampener.apply(
            cmd_q=np.asarray(cmd_q, dtype=np.float64),
            cmd_dq=np.asarray(cmd_dq, dtype=np.float64),
            cmd_tau=np.asarray(cmd_tau, dtype=np.float64),
            kp=motor_kp_cfg,
            kd=motor_kd_cfg,
            dof_pos_latest=dof_pos_latest,
        )

        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors)
        cmd_kd = np.zeros(self.robot_config.num_motors)

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(q_d[j_id])
            cmd_dq_target[m_id] = float(dq_d[j_id])
            cmd_tau_target[m_id] = float(tau_d[j_id])
            cmd_kp[m_id] = float(kp_d[j_id])
            cmd_kd[m_id] = float(kd_d[j_id])

        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q_target)
        cmd.dq_target = list(cmd_dq_target)
        cmd.tau_ff = list(cmd_tau_target)
        cmd.kp = list(cmd_kp * self._kp_level)
        cmd.kd = list(cmd_kd * self._kd_level)

        self._send_logger.maybe_log(
            q_target=cmd_q_target,
            kp=cmd_kp * self._kp_level,
            kd=cmd_kd * self._kd_level,
            unitree=self.unitree_interface,
        )

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
