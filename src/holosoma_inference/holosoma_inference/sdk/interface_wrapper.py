import numpy as np
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.command_sender import create_command_sender
from holosoma_inference.sdk.state_processor import create_state_processor


class InterfaceWrapper:
    """
    Wrapper for robot control supporting multiple backends:
    - sdk2py: uses Python SDK for both unitree and booster robots
    - unitree: uses C++/pybind11 binding for unitree robots only

    Backend selection based on `robot_config.sdk_type`:
      - 'booster': uses sdk2py (booster robots)
      - 'unitree': uses C++/pybind11 binding (unitree robots only)
    Provides a unified interface for get_low_state, send_low_command, and joystick input.
    """

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        self.logger = logger
        self.use_joystick = use_joystick
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.sdk_type = robot_config.sdk_type
        self.backend = None
        self._unitree_motor_order = None  # Optional hardware motor order override

        # Initialize gain levels for binding backend
        self._kp_level = 1.0
        self._kd_level = 1.0

        # Initialize sdk components
        self._init_sdk_components()

    def _init_sdk_components(self):
        """Initialize the appropriate backend based on SDK type."""
        if self.sdk_type == "booster":
            # Use sdk2py Python interface for booster
            self.backend = "sdk2py"
            self.command_sender = create_command_sender(self.robot_config)
            self.state_processor = create_state_processor(self.robot_config)
        elif self.sdk_type == "unitree":
            try:
                import unitree_interface
            except ImportError as e:
                raise ImportError("unitree_interface python binding not found.") from e
            # Use C++/pybind11 binding (unitree only)
            self.backend = "binding"
            # Parse robot type
            robot_type_map = {
                "G1": unitree_interface.RobotType.G1,
                "H1": unitree_interface.RobotType.H1,
                "H1_2": unitree_interface.RobotType.H1_2,
                "GO2": unitree_interface.RobotType.GO2,
            }
            # Parse message type
            message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}
            self.unitree_interface = unitree_interface.create_robot(
                self.interface_str,
                robot_type_map[self.robot_config.robot.upper()],
                message_type_map[self.robot_config.message_type.upper()],
            )
            # Set control mode to PR (Pitch/Roll)
            self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)
            control_mode = self.unitree_interface.get_control_mode()
            print(f"Control mode set to: {'PR' if control_mode == unitree_interface.ControlMode.PR else 'AB'}")

            # GO2 SDK motor order differs from our joint order; override when using the binding.
            if self.robot_config.robot.lower() == "go2":
                # SDK motor order: FR, FL, RR, RL (each hip, thigh, calf)
                self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
        else:
            raise ValueError(f"Unsupported SDK_TYPE: {self.sdk_type}")

        # Setup wireless controller
        if self.use_joystick:
            self._setup_wireless_controller()

    # ============================================================================
    # Robot State and Command Interface
    # ============================================================================

    def get_low_state(self):
        """Get the latest low-level robot state as a numpy array."""
        if self.backend == "sdk2py":
            return self.state_processor.get_robot_state_data()
        if self.backend == "binding":
            return self._convert_binding_state_to_array()
        raise RuntimeError("InterfaceWrapper not initialized correctly.")

    def _convert_binding_state_to_array(self):
        """Convert binding LowState to numpy array format compatible with sdk2py."""
        state = self.unitree_interface.read_low_state()

        # Compose array: [base_pos(3), quat(4), joint_pos(N), base_lin_vel(3), base_ang_vel(3), joint_vel(N), ...]
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
        return np.concatenate(
            [
                base_pos,
                quat,
                joint_pos,
                base_lin_vel,
                base_ang_vel,
                joint_vel,
            ]
        ).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q,
        cmd_dq,
        cmd_tau,
        dof_pos_latest=None,
        kp_override=None,
        kd_override=None,
    ):
        """Send low-level command to the robot using the selected backend."""
        if self.backend == "sdk2py":
            self.command_sender.send_command(
                cmd_q,
                cmd_dq,
                cmd_tau,
                dof_pos_latest,
                kp_override=kp_override,
                kd_override=kd_override,
            )
        elif self.backend == "binding":
            # print(f"cmd_q: {type(cmd_q)}, {len(cmd_q)}, {cmd_q}")
            cmd_q_target = np.zeros(self.robot_config.num_motors)
            cmd_dq_target = np.zeros(self.robot_config.num_motors)
            cmd_tau_target = np.zeros(self.robot_config.num_motors)
            cmd_kp_override = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
            cmd_kd_override = np.zeros(self.robot_config.num_motors) if kd_override is not None else None
            motor_order = self._unitree_motor_order or self.robot_config.joint2motor
            for j_id in range(self.robot_config.num_joints):
                m_id = motor_order[j_id]
                cmd_q_target[m_id] = float(cmd_q[j_id])
                cmd_dq_target[m_id] = float(cmd_dq[j_id])
                cmd_tau_target[m_id] = float(cmd_tau[j_id])
                if cmd_kp_override is not None:
                    cmd_kp_override[m_id] = float(kp_override[j_id])
                if cmd_kd_override is not None:
                    cmd_kd_override[m_id] = float(kd_override[j_id])
            self._send_binding_command(
                cmd_q_target,
                cmd_dq_target,
                cmd_tau_target,
                kp_override=cmd_kp_override,
                kd_override=cmd_kd_override,
            )
        else:
            raise RuntimeError("InterfaceWrapper not initialized correctly.")

    def _send_binding_command(self, cmd_q, cmd_dq, cmd_tau, kp_override=None, kd_override=None):
        """Send command using the C++/pybind11 binding."""
        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q)
        cmd.dq_target = list(cmd_dq)
        cmd.tau_ff = list(cmd_tau)
        motor_kp = np.array(kp_override if kp_override is not None else self.robot_config.motor_kp)
        motor_kd = np.array(kd_override if kd_override is not None else self.robot_config.motor_kd)
        cmd.kp = list(motor_kp * self._kp_level)
        cmd.kd = list(motor_kd * self._kd_level)
        self.unitree_interface.write_low_command(cmd)

    # ============================================================================
    # Wireless Controller / Joystick Interface
    # ============================================================================

    def _setup_wireless_controller(self):
        """Setup wireless controller for joystick input."""
        if self.sdk_type == "unitree":
            # Wireless controller is already initialized in the binding
            pass
        elif self.sdk_type == "booster":
            from holosoma_inference.sdk.command_sender.booster.joystick_message import (
                BoosterJoystickMessage,
            )
            from holosoma_inference.sdk.command_sender.booster.remote_control_service import (
                BoosterRemoteControlService,
            )

            # Booster robots use evdev-based joystick input
            try:
                self.booster_remote_control = BoosterRemoteControlService()
                self.booster_joystick_msg = BoosterJoystickMessage(self.booster_remote_control)
                print(colored("Booster Remote Control Service Initialized", "green"))
            except ImportError as e:
                print(colored(f"Warning: Failed to initialize booster remote control: {e}", "yellow"))
                self.booster_remote_control = None
                self.booster_joystick_msg = None
        else:
            raise NotImplementedError(f"Joystick is not supported for {self.sdk_type} SDK.")
        self._wc_msg = None
        self._key_states = {}
        self._last_key_states = {}
        self._wc_key_map = self._default_wc_key_map()
        print(colored("Wireless Controller Initialized", "green"))

    def _default_wc_key_map(self):
        """Default wireless controller key mapping."""
        return {
            1: "R1",
            2: "L1",
            3: "L1+R1",
            4: "start",
            8: "select",
            10: "L1+select",
            16: "R2",
            32: "L2",
            64: "F1",
            128: "F2",
            256: "A",
            264: "select+A",
            512: "B",
            520: "select+B",
            768: "A+B",
            1024: "X",
            1032: "select+X",
            1280: "A+X",
            1536: "B+X",
            2048: "Y",
            2304: "A+Y",
            2560: "B+Y",
            2056: "select+Y",
            3072: "X+Y",
            4096: "up",
            4097: "R1+up",
            4352: "A+up",
            4608: "B+up",
            4104: "select+up",
            5120: "X+up",
            6144: "Y+up",
            8192: "right",
            8193: "R1+right",
            8448: "A+right",
            9216: "X+right",
            10240: "Y+right",
            8200: "select+right",
            16384: "down",
            16392: "select+down",
            16385: "R1+down",
            16640: "A+down",
            16896: "B+down",
            17408: "X+down",
            18432: "Y+down",
            32768: "left",
            32769: "R1+left",
            32776: "select+left",
            33024: "A+left",
            33792: "X+left",
            34816: "Y+left",
        }

    def wireless_controller_handler(self, msg):
        """Handle the wireless controller message."""
        self._wc_msg = msg

    def get_joystick_msg(self):
        """
        Get the latest joystick/wireless controller message in a unified format.
        Returns an object with .lx, .ly, .rx, .keys, etc. for both backends.
        """
        if self.sdk_type == "unitree":
            return self.unitree_interface.read_wireless_controller()
        if self.sdk_type == "booster":
            return self.booster_joystick_msg if hasattr(self, "booster_joystick_msg") else None
        return None

    def get_joystick_key(self, wc_msg=None):
        """
        Get the current key (cur_key) from the joystick message using the key map.
        """
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    def process_joystick_input(self, lin_vel_command, ang_vel_command, stand_command, upper_body_motion_active):
        """
        Process joystick input and update commands in a unified way.
        Args:
            lin_vel_command: np.ndarray, shape (1, 2)
            ang_vel_command: np.ndarray, shape (1, 1)
            stand_command: np.ndarray, shape (1, 1)
            upper_body_motion_active: bool
        Returns:
            (lin_vel_command, ang_vel_command, key_states): updated values
        """
        wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return lin_vel_command, ang_vel_command, self._key_states
        # Process stick input
        if getattr(wc_msg, "keys", 0) == 0 and not upper_body_motion_active:
            lx = getattr(wc_msg, "lx", 0.0)
            ly = getattr(wc_msg, "ly", 0.0)
            rx = getattr(wc_msg, "rx", 0.0)
            lin_vel_command[0, 1] = -(lx if abs(lx) > 0.1 else 0.0) * stand_command[0, 0]
            lin_vel_command[0, 0] = (ly if abs(ly) > 0.1 else 0.0) * stand_command[0, 0]
            ang_vel_command[0, 0] = -(rx if abs(rx) > 0.1 else 0.0) * stand_command[0, 0]
        # Process button input
        cur_key = self.get_joystick_key(wc_msg)
        self._last_key_states = self._key_states.copy()
        if cur_key:
            self._key_states[cur_key] = True
        else:
            self._key_states = dict.fromkeys(self._wc_key_map.values(), False)

        return lin_vel_command, ang_vel_command, self._key_states

    # ============================================================================
    # Gain Management
    # ============================================================================

    @property
    def kp_level(self):
        """Get or set the proportional gain level."""
        if self.backend == "sdk2py":
            return self.command_sender.kp_level
        if self.backend == "binding":
            return self._kp_level
        return None

    @kp_level.setter
    def kp_level(self, value):
        if self.backend == "sdk2py":
            self.command_sender.kp_level = value
        elif self.backend == "binding":
            self._kp_level = value

    @property
    def kd_level(self):
        """Get or set the derivative gain level."""
        if self.backend == "sdk2py":
            return getattr(self.command_sender, "kd_level", 1.0)
        if self.backend == "binding":
            return self._kd_level
        return None

    @kd_level.setter
    def kd_level(self, value):
        if self.backend == "sdk2py":
            self.command_sender.kd_level = value
        elif self.backend == "binding":
            self._kd_level = value
