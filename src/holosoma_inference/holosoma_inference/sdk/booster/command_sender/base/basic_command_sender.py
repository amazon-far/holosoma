from __future__ import annotations

from abc import ABC, abstractmethod

from holosoma_inference.config.config_types.robot import RobotConfig


def _get_config_value(config: RobotConfig | dict, key: str, default=None):
    """
    Util function to temporarily support getting valeus from both `RobotConfig` and old dict-based configs.

    TODO: Force dataclasses-based access once `run_sim.py` is moved to `holosoma`.
    """

    # Try dataclass attribute access first
    if hasattr(config, key):
        return getattr(config, key)
    # Fall back to hydra
    if isinstance(config, dict):
        return config.get(key, default)
    raise RuntimeError(f"Unsupported type {type(config)}")


class BasicCommandSender(ABC):
    """Abstract base class for command sender implementations."""

    def __init__(self, config: RobotConfig | dict, lcm=None):
        self.lcm = lcm
        self.config = config
        self.sdk_type = _get_config_value(config, "sdk_type", "unitree")
        self.motor_type = _get_config_value(config, "motor_type", "serial")

        # Initialize control gains
        self.kp_level = 1.0
        self.kd_level = 1.0
        self.waist_kp_level = 1.0

        # Initialize weak motor joint index
        self.weak_motor_joint_index = []
        weak_motor_cfg = _get_config_value(self.config, "weak_motor_joint_index")
        if weak_motor_cfg:
            for value in self.robot.WeakMotorJointIndex.values():
                self.weak_motor_joint_index.append(value)

        self.no_action = 0

        # Initialize SDK-specific components
        self._init_sdk_components()

    @abstractmethod
    def _init_sdk_components(self):
        """Initialize SDK-specific components. Must be implemented by subclasses."""

    @abstractmethod
    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        """Send command to robot. Must be implemented by subclasses."""

    def is_weak_motor(self, motor_index):
        """Check if a motor is a weak motor."""
        return motor_index in self.weak_motor_joint_index

    def _set_motor_command(
        self,
        motor_cmd,
        motor_id,
        joint_id,
        cmd_q,
        cmd_dq,
        cmd_tau,
        motor_kp,
        motor_kd,
        kp_override=None,
        kd_override=None,
    ):
        """Set motor command for a specific motor."""
        default_q = _get_config_value(self.config, "default_motor_angles")

        if joint_id == -1 or self.no_action:
            motor_cmd.q = default_q[motor_id]
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0
        else:
            motor_cmd.q = cmd_q[joint_id]
            motor_cmd.dq = cmd_dq[joint_id]
            motor_cmd.tau = cmd_tau[joint_id]

            # The native runner always supplies joint-ordered gains; motor_kp/kd may
            # then be None (no config override), so only read them as a fallback.
            if kp_override is not None:
                kp_value = kp_override[joint_id]
            elif motor_kp is not None:
                kp_value = motor_kp[motor_id]
            else:
                raise ValueError("No KP gain: pass kp_override or set robot motor_kp")
            if kd_override is not None:
                kd_value = kd_override[joint_id]
            elif motor_kd is not None:
                kd_value = motor_kd[motor_id]
            else:
                raise ValueError("No KD gain: pass kd_override or set robot motor_kd")

            motor_cmd.kp = kp_value * self.kp_level
            motor_cmd.kd = kd_value * self.kd_level

    def _fill_motor_commands(self, motor_cmd, cmd_q, cmd_dq, cmd_tau, kp_override=None, kd_override=None):
        """Fill motor commands for all motors."""
        motor2joint = _get_config_value(self.config, "motor2joint")
        num_motors = _get_config_value(self.config, "num_motors")
        motor_kp = _get_config_value(self.config, "motor_kp")
        motor_kd = _get_config_value(self.config, "motor_kd")

        for i in range(num_motors):
            m_id = i
            j_id = motor2joint[i]
            cmd = motor_cmd[m_id]
            self._set_motor_command(
                cmd,
                m_id,
                j_id,
                cmd_q,
                cmd_dq,
                cmd_tau,
                motor_kp,
                motor_kd,
                kp_override=kp_override,
                kd_override=kd_override,
            )
