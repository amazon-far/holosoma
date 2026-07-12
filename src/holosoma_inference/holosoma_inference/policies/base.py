from __future__ import annotations

import itertools
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import netifaces as ni
import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.inputs import create_input
from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.sdk import create_interface
from holosoma_inference.utils.latency import LatencyTracker
from holosoma_inference.utils.math.quat import quat_rotate_inverse
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.wandb import load_checkpoint

# Maps SWITCH_POLICY_N commands to 0-based policy indices.
STATE_COMMAND_TO_POLICY_INDEX: dict[StateCommand, int] = {
    StateCommand[f"SWITCH_POLICY_{n}"]: n - 1 for n in range(1, 10)
}


class BasePolicy:
    """
    Base policy class for Holosoma deployment on humanoid robots.

    Supports both simulation and real robot deployment with keyboard/joystick controls.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize the base policy with configuration and model."""
        self.config = config
        # Initialize robot config
        self._init_robot_config(self.config.robot)
        # Initialize SDK components
        self._init_sdk_components()
        # Initialize observation config
        self._init_obs_config()
        # Initialize communication components
        self._init_communication_components()
        # Initialize policy components
        self._init_policy_components(
            self.config.task.model_path, self.config.task.policy_action_scale, self.config.task.rl_rate
        )
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()
        # Initialize phase components
        self._init_phase_components()
        # Initialize latency tracking
        self._init_latency_tracking()

    # ============================================================================
    # Initialization Methods
    # ============================================================================

    def _init_robot_config(self, robot_config: RobotConfig):
        """Initialize robot configuration and parameters."""
        self.robot_config = robot_config
        self.num_dofs = self.robot_config.num_joints
        self.default_dof_angles = np.array(self.robot_config.default_dof_angles)
        self.num_upper_dofs = robot_config.num_upper_body_joints

        # Per-robot joint offsets (action-space calibration, does not affect observations)
        offsets_deg = robot_config.joint_offsets_deg
        self.joint_offsets = np.deg2rad(offsets_deg) if offsets_deg is not None else np.zeros(self.num_dofs)

        # Setup dof names and indices
        self._setup_dof_mappings()

    def _setup_dof_mappings(self):
        """Setup DOF names and their corresponding indices."""
        self.dof_names = self.robot_config.dof_names
        # TODO: Remove upper body mentions as it's not used anymore.
        self.upper_dof_names = self.robot_config.dof_names_upper_body
        self.lower_dof_names = self.robot_config.dof_names_lower_body

        # These are used by derived classes, so keep them
        if self.upper_dof_names:
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
        else:
            self.upper_dof_indices = []

        if self.lower_dof_names:
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
        else:
            self.lower_dof_indices = []

    def _init_sdk_components(self):
        """Additional SDK components initialization based on robot type."""
        if hasattr(self, "_shared_hardware_source"):
            self.sdk_type = self._shared_hardware_source.sdk_type
            return
        self.sdk_type = self.robot_config.sdk_type
        if self.sdk_type == "booster":
            from booster_robotics_sdk import ChannelFactory

            ip = ni.ifaddresses(self.config.task.interface)[ni.AF_INET][0]["addr"]
            ChannelFactory.Instance().Init(self.config.task.domain_id, ip)
        else:
            pass  # No channel initialization needed for Unitree binding / other robots

    def _init_obs_config(self):
        """Initialize observation metadata and history buffers."""
        self.obs_config = self.config.observation
        self.obs_scales = self.obs_config.obs_scales
        self.obs_dims = self.obs_config.obs_dims
        self.obs_dict = self.obs_config.obs_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.obs_config.history_length_dict

        # Initialize per-term history buffers using deques
        self._initialize_history_state()

    def _initialize_history_state(self):
        """Create per-term history deques and zero-initialized flattened buffers."""
        self.obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self.obs_terms_sorted: dict[str, list[str]] = {}
        self.obs_buf_dict: dict[str, np.ndarray] = {}

        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = sorted(term_names)
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))

            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _init_communication_components(self):
        """Initialize appropriate robot interface."""
        if hasattr(self, "_shared_hardware_source"):
            self.interface = self._shared_hardware_source.interface
            return
        # The SDK's own wireless-controller path is only needed when an
        # "interface" channel is selected.  The "joystick" channel is read
        # via host-side evdev (UsbJoystickInput) and does not touch the SDK.
        vel = self.config.task.velocity_input
        other = self.config.task.state_input
        need_joystick = "interface" in {vel, other}
        self.interface = create_interface(
            self.robot_config,
            self.config.task.domain_id,
            self.config.task.interface,
            need_joystick,
        )

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Initialize policy-related components."""
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate
        self.model_paths = self._collect_model_paths(model_path)
        self._policy_states: list[dict] = []
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        resolved_paths: list[str] = []

        for path in self.model_paths:
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
            self.setup_policy(local_path)
            self._policy_states.append(self._capture_policy_state())

        self.model_paths = resolved_paths
        self.active_policy_index = 0
        self.active_model_path = None
        self._activate_policy(0, announce=False)

        # Determine KP/KD values: config override > ONNX metadata > error
        self._resolve_control_gains()

    def _collect_model_paths(self, model_path):
        """Normalize model_path into a list of up to nine entries."""
        if isinstance(model_path, (list, tuple)):
            paths = list(model_path)
        elif model_path is not None:
            paths = [model_path]
        else:
            paths = []

        paths = [str(path) for path in paths if path]
        if not paths:
            raise ValueError("At least one model_path must be provided for policy initialization.")
        if len(paths) > 9:
            # Error out instead of warning
            raise ValueError("Received more than nine model paths. Only up to nine model paths are supported.")
        return paths

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path, downloading from W&B if required."""
        if model_path.startswith(("wandb://", "https://")):
            logger.info(f"Downloading checkpoint from W&B: {model_path}")
            checkpoint_path = load_checkpoint(None, model_path)
            resolved_path = str(checkpoint_path)
            logger.info("Checkpoint downloaded to: %s", resolved_path)
            return resolved_path
        return model_path

    def _capture_policy_state(self) -> dict:
        """Capture the current policy state for later reuse."""
        return {
            "onnx_policy_session": self.onnx_policy_session,
            "onnx_input_names": self.onnx_input_names,
            "onnx_output_names": self.onnx_output_names,
            "policy_callable": self.policy,
            "onnx_kp": self.onnx_kp,
            "onnx_kd": self.onnx_kd,
        }

    def _restore_policy_state(self, state: dict):
        """Restore a previously captured policy state."""
        self.onnx_policy_session = state["onnx_policy_session"]
        self.onnx_input_names = state["onnx_input_names"]
        self.onnx_output_names = state["onnx_output_names"]
        self.policy = state["policy_callable"]
        self.onnx_kp = state["onnx_kp"]
        self.onnx_kd = state["onnx_kd"]

    def _activate_policy(self, index: int, announce: bool = True):
        """Activate a preloaded policy."""
        if not (0 <= index < len(self.model_paths)):
            return

        self._restore_policy_state(self._policy_states[index])
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        self.active_policy_index = index
        self.active_model_path = self.model_paths[index]
        self._on_policy_switched(self.active_model_path)

        if announce and len(self.model_paths) > 1 and hasattr(self, "logger"):
            name = Path(self.active_model_path).name
            self.logger.info(colored(f"Switched to policy [{index + 1}]: {name}", "blue"))

    def _on_policy_switched(self, model_path: str):
        """Hook for derived classes to reset state after loading a new policy."""
        _ = model_path

    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.task.desired_base_height
        self.gait_period = self.config.task.gait_period

        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])

        # These are used by derived classes, so keep them
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))

        # Upper body controller
        self.upper_body_controller = None

        # Pre-allocate command arrays for postprocessing
        self.cmd_q = np.zeros(self.num_dofs)
        self.cmd_dq = np.zeros(self.num_dofs)
        self.cmd_tau = np.zeros(self.num_dofs)
        self._last_sent_q: np.ndarray | None = None
        self._target_transition: dict[str, np.ndarray | int | str | None] | None = None

        # Smooth policy-switch interpolation (loco ↔ wbt)
        self._interp_state: str = "idle"  # "idle" | "delaying" | "interping"
        self._interp_start_q: np.ndarray | None = None
        self._interp_target_q: np.ndarray | None = None
        self._interp_step: int = 0
        self._interp_delay_steps: int = 0
        self._interp_total_steps: int = 0
        self._interp_on_delay_done: callable | None = None
        self._interp_on_done: callable | None = None
        self._interp_kp: np.ndarray | None = None
        self._interp_kd: np.ndarray | None = None
        self._interp_override_indices: list[int] | None = None

    def _init_phase_components(self):
        """Initialize phase components."""
        self.use_phase = self.config.task.use_phase
        if self.use_phase:
            self.phase = np.zeros((1, 2))
            self.phase[:, 0] = 0.0  # left foot starts at 0
            self.phase[:, 1] = np.pi  # right foot starts at pi
            self.phase_dt = 2 * np.pi / (self.rl_rate * self.gait_period)

    def _init_latency_tracking(self):
        """Initialize latency tracking components."""
        self.latency_tracker = LatencyTracker(window_size=int(self.rl_rate))

    def _init_input_handlers(self):
        """Initialize input handlers (ROS, joystick, keyboard)."""
        if hasattr(self, "_shared_hardware_source"):
            self.logger = self._shared_hardware_source.logger
            self.rate = self._shared_hardware_source.rate
            self.rl_rate = self._shared_hardware_source.rl_rate
            self.use_joystick = self._shared_hardware_source.use_joystick
            self.use_keyboard = self._shared_hardware_source.use_keyboard
            # Share input providers — one queue, active policy drains it each cycle.
            self._velocity_input: VelCmdProvider = self._shared_hardware_source._velocity_input
            self._command_provider: StateCommandProvider = self._shared_hardware_source._command_provider
            return
        self._init_rate_handler()
        self._init_input_device()

    def _init_rate_handler(self):
        """Initialize rate limiter and logger."""
        self.rl_rate = self.config.task.rl_rate
        self.logger = logger
        self.rate = RateLimiter(self.rl_rate)

    def _init_input_device(self):
        """Initialize input hardware and create input providers.

        Each channel independently selects from InputSource enum values.
        Hardware is initialized based on the union of both channels' requirements,
        then providers are created via factory methods (overridden by subclasses).
        """
        vel = self.config.task.velocity_input
        other = self.config.task.state_input
        sources = {vel, other}

        # Joystick hardware (needed if either channel uses interface or joystick)
        if {"interface", "joystick"} & sources:
            self._init_joystick_handler()
        else:
            self.use_joystick = False

        # use_keyboard is set by KeyboardListener when providers start
        self.use_keyboard = False

        self._create_input_providers()

    def _init_joystick_handler(self):
        """Initialize joystick handler."""
        if sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Windows or Mac.")
            self.logger.warning("Falling back to keyboard for joystick channel")
            self.use_joystick = False
        else:
            self.logger.info("Using joystick")
            self.use_joystick = True

    def _create_input_providers(self):
        """Create and start input providers based on config.

        When both channels use the same source, a single provider is shared
        (important for KeyboardInput which pops from a shared queue).
        """
        self._setup_keyboard_listener()

        self._velocity_input: VelCmdProvider = create_input(self, self.config.task.velocity_input, "velocity")

        if self.config.task.velocity_input == self.config.task.state_input:
            self._command_provider: StateCommandProvider = self._velocity_input
        else:
            self._command_provider: StateCommandProvider = create_input(self, self.config.task.state_input, "command")

        self._velocity_input.start()
        if self._command_provider is not self._velocity_input:
            self._command_provider.start()

    def _setup_keyboard_listener(self):
        """Start the shared keyboard listener if any channel uses keyboard input."""
        if hasattr(self, "_shared_hardware_source"):
            return
        sources = {self.config.task.velocity_input, self.config.task.state_input}
        if "keyboard" not in sources:
            return
        from holosoma_inference.inputs.impl.keyboard import get_keyboard_listener

        listener = get_keyboard_listener()
        active = listener.start()
        self.use_keyboard = active
        if not active:
            self.logger.warning("No TTY — keyboard input disabled")
            self.use_policy_action = True

    # ============================================================================
    # Policy Methods
    # ============================================================================

    def setup_policy(self, model_path):
        """Setup ONNX policy model and extract metadata."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        self.onnx_input_names = input_names
        self.onnx_output_names = output_names

        # Extract metadata from ONNX model (hard fault if fails)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract KP/KD from metadata (will be None if not present)
        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        def policy_act(obs_dict):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy = policy_act

    def _resolve_control_gains(self):
        """Resolve KP/KD values with priority: config override > ONNX metadata > error.

        Creates a new config instance with resolved values if needed.
        """
        # Check if config has explicit KP/KD values
        config_has_kp = hasattr(self.robot_config, "motor_kp") and self.robot_config.motor_kp is not None
        config_has_kd = hasattr(self.robot_config, "motor_kd") and self.robot_config.motor_kd is not None

        if config_has_kp and config_has_kd:
            # Config already has values (override) - nothing to do
            logger.info(colored("Using KP/KD from config (override)", "yellow"))
            kp_values = np.array(self.robot_config.motor_kp)
            kd_values = np.array(self.robot_config.motor_kd)
        elif self.onnx_kp is not None and self.onnx_kd is not None:
            # Use ONNX metadata (default) - create new config with values
            logger.info(colored("Using KP/KD from ONNX metadata", "green"))
            kp_values = self.onnx_kp
            kd_values = self.onnx_kd
            # Create new config instance with ONNX values
            self.robot_config = replace(
                self.robot_config, motor_kp=tuple(kp_values.tolist()), motor_kd=tuple(kd_values.tolist())
            )
            # Update interface's robot_config and propagate to internal SDK components
            self.interface.update_config(self.robot_config)
        else:
            # No values available - error
            raise ValueError(
                "No KP/KD values found. Either provide them in robot config "
                "or ensure ONNX model has metadata attached during training."
            )

        # Validate dimensions
        if len(kp_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KP array length ({len(kp_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )
        if len(kd_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KD array length ({len(kd_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )

    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict

    def _print_observations(self, obs: dict[str, np.ndarray]) -> None:
        """Print observation vector with term naming for debugging.

        Args:
            obs: Dictionary mapping observation group names to their flattened arrays.
        """
        np.set_printoptions(suppress=True, precision=3)
        print("\n========== Observation Vector ==========")
        for group_name, group_obs in obs.items():
            print(f"\n{group_name}:")
            if group_name in self.obs_dict:
                start_idx = 0
                for term_name in self.obs_terms_sorted.get(group_name, []):
                    term_dim = self.obs_dims[term_name]
                    history_len = self.history_length_dict.get(group_name, 1)
                    total_dim = term_dim * history_len
                    term_values = group_obs[0, start_idx : start_idx + total_dim]
                    print(f"  {term_name:20s} (dim={term_dim:2d}, hist={history_len}): {term_values}")
                    start_idx += total_dim

        # Joint table: dof_name | q (deg) | dq | action
        self._print_joint_table(obs)
        print("========================================\n")

    def _print_joint_table(self, obs: dict[str, np.ndarray]) -> None:
        """Print a compact per-joint table: name | q(°) | dq(°/s) | act(°)."""
        # Walk obs_terms_sorted + obs_dims to locate dof_pos / dof_vel slices
        q = dq = None
        for grp, buf in obs.items():
            col = 0
            for term in self.obs_terms_sorted.get(grp, []):
                dim = self.obs_dims[term] * self.history_length_dict.get(grp, 1)
                if q is None and term == "dof_pos":
                    q = buf[0, col : col + dim] / self.obs_scales.get("dof_pos", 1.0)
                if dq is None and term == "dof_vel":
                    dq = buf[0, col : col + dim] / self.obs_scales.get("dof_vel", 1.0)
                col += dim
        act = self.scaled_policy_action[0] if self.scaled_policy_action is not None else None
        d = np.degrees
        w = max(len(n) for n in self.dof_names)
        print(f"\n  {'joint':<{w}}  {'q(°)':>7}  {'dq(°/s)':>8}  {'act(°)':>7}")
        print(f"  {'─' * (w + 29)}")
        for i, name in enumerate(self.dof_names):
            qi = f"{d(q[i]):7.1f}" if q is not None and i < len(q) else "    n/a"
            di = f"{d(dq[i]):8.1f}" if dq is not None and i < len(dq) else "     n/a"
            ai = f"{d(act[i]):7.1f}" if act is not None and i < len(act) else "    n/a"
            print(f"  {name:<{w}}  {qi}  {di}  {ai}")

    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        self.scaled_policy_action = policy_action * self.policy_action_scale
        if self.config.task.debug.force_zero_action:
            self.scaled_policy_action = np.zeros_like(self.scaled_policy_action)

        return self.scaled_policy_action

    # ============================================================================
    # Observation Processing Methods
    # ============================================================================

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state."""
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        if self.config.task.debug.force_zero_angular_velocity:
            current_obs_buffer_dict["base_ang_vel"] = np.zeros((1, 3))
        else:
            current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # Use pre-computed corrected gravity if available from interface, else compute
        # This logic seems very brittle. TODO: Return a dataclass instead of just a numpy array.
        expected_len = (
            7 + self.num_dofs + 6 + self.num_dofs
        )  # base_pos(3) + quat(4) + dof_pos + lin_vel(3) + ang_vel(3) + dof_vel
        if self.config.task.debug.force_upright_imu:
            current_obs_buffer_dict["projected_gravity"] = np.array([[0.0, 0.0, -1.0]])
        elif robot_state_data.shape[1] == expected_len + 3:
            current_obs_buffer_dict["projected_gravity"] = robot_state_data[:, expected_len : expected_len + 3]
        else:
            v = np.array([[0, 0, -1]])
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

        return current_obs_buffer_dict

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary with per-term scaling."""
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self.obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Observation term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self.obs_scales[term]
                grouped_terms[term] = (term_obs * scale).astype(np.float32, copy=False)
            current_obs_dict[group] = grouped_terms
        return current_obs_dict

    def _prepare_group_observations(self, robot_state_data):
        """Return flattened observations per group with history applied per term."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)

        return self._update_obs_history(current_obs_dict)

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Update observation history buffers and return flattened observations per group."""
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self.obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    history = [np.zeros_like(obs)] * missing + history

                # Match training order: time dimension first, then flatten into [history_len * term_dim].
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

        self.obs_buf_dict = {group: value.copy() for group, value in group_outputs.items()}
        return group_outputs

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference."""
        group_outputs = self._prepare_group_observations(robot_state_data)
        if "actor_obs" not in group_outputs:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")
        return {"actor_obs": group_outputs["actor_obs"].astype(np.float32, copy=False)}

    # ============================================================================
    # Control/Command Methods
    # ============================================================================

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def policy_action(self):
        """Execute policy action and send commands to robot."""

        # Snapshot flags to prevent race with mode-switch handler thread
        use_policy = self.use_policy_action
        get_ready = self.get_ready_state

        kp_override = None
        kd_override = None

        # 读取机器人的当前状态数据
        with self.latency_tracker.measure("read_state"):
            robot_state_data = self.interface.get_low_state()

        #进行预处理，计算目标关节位置等
        with self.latency_tracker.measure("preprocessing"):
            # 决定当前阶段的目标关节位置：准备阶段使用插值，非准备阶段使用当前关节位置或手动命令
            #初始化模式下，关节位置从当前状态逐渐过渡到默认站立位置，过渡时间由init_count控制（最多500步）。
            # 非初始化模式下，如果没有使用策略控制，则尝试获取手动命令（如果提供了_get_manual_command方法），否则直接使用当前关节位置作为目标。
            if get_ready:
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
            #手动控制模式下，如果提供了_get_manual_command方法，则调用它来获取目标关节位置和可选的KP/KD覆盖值；
            
            elif not use_policy:
                manual_cmd = self._get_manual_command(robot_state_data)
                if manual_cmd is not None:
                    q_target = manual_cmd["q"]
                    kp_override = manual_cmd.get("kp")
                    kd_override = manual_cmd.get("kd")
                # 如果没有提供，则直接使用当前关节位置作为目标。保持不动
                else:
                    q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
            else:
                # 准备阶段不使用策略，保持不动
                pass

        # 推理阶段：如果使用策略控制且不在准备阶段，则调用rl_inference方法获取策略输出的目标关节位置，并进行缩放。
        if use_policy and not get_ready:
            with self.latency_tracker.measure("inference"):
                #从机器人状态数据中准备好输入给RL模型的观察数据，并调用policy方法进行推理，得到原始的策略动作输出。
                scaled_policy_action = self.rl_inference(robot_state_data)

        # 后处理阶段：如果使用策略控制且不在准备阶段，则将策略输出的目标关节位置与默认关节角度相加，
        # 得到最终的目标关节位置。然后将目标关节位置加上预先定义的关节偏移，得到最终的命令关节位置。
        with self.latency_tracker.measure("postprocessing"):
            if use_policy and not get_ready:
                # 如果策略输出的动作维度不等于关节数量，则根据是否使用上半身控制进行处理：
                # 如果不使用上半身控制，则在动作前面添加适当数量的零；
                # 如果使用上半身控制，则抛出未实现错误。
                if scaled_policy_action.shape[1] != self.num_dofs:
                    if not self.upper_body_controller:
                        scaled_policy_action = np.concatenate(
                            [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action], axis=1
                        )
                    else:
                        raise NotImplementedError("Upper body controller not implemented")
                # 将策略输出的目标关节位置与默认关节角度相加，得到最终的目标关节位置。
                q_target = scaled_policy_action + self.default_dof_angles

            q_target = self._apply_target_transition(q_target, robot_state_data)

            # Smooth policy-switch interpolation (loco ↔ wbt).
            # Only overrides the DOF indices specified by override_indices;
            # the active policy keeps controlling everything else (e.g. legs).
            interp_result = self._step_policy_interp(robot_state_data)
            if interp_result is not None:
                q_interp, ikp, ikd, interp_idxs = interp_result
                if interp_idxs is not None:
                    q_target[0, interp_idxs] = q_interp[0, interp_idxs]
                else:
                    q_target = q_interp  # backward-compat: override all
                if ikp is not None:
                    kp_override = ikp
                if ikd is not None:
                    kd_override = ikd

            # 将目标关节位置加上预先定义的关节偏移，得到最终的命令关节位置。
            self.cmd_q[:] = q_target[0] + self.joint_offsets
            self._last_sent_q = self.cmd_q.copy()

       
        with self.latency_tracker.measure("action_pub"):
            #  调用接口的send_low_command方法，将最终的命令关节位置、速度和力矩发送给机器人。同时，如果提供了KP/KD覆盖值，则将它们传递给接口以覆盖默认的控制增益。
            self.interface.send_low_command(
                self.cmd_q,
                self.cmd_dq,
                self.cmd_tau,
                robot_state_data[0, 7 : 7 + self.num_dofs],
                kp_override=kp_override,
                kd_override=kd_override,
            )

    # ------------------------------------------------------------------
    #  Smooth policy-switch interpolation (loco ↔ wbt)
    # ------------------------------------------------------------------

    def _start_policy_interp(
        self,
        target_q: np.ndarray,
        duration_s: float,
        delay_s: float = 0.0,
        on_delay_done: callable = None,
        on_done: callable = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
        override_indices: list[int] | None = None,
    ):
        """Begin smooth interpolation from current pose to *target_q*.

        Parameters
        ----------
        target_q:
            Desired joint positions at the end of interpolation (1, N) or (N,).
        duration_s:
            Interpolation duration in seconds.
        delay_s:
            Hold-current-pose delay before interpolation begins.
        on_delay_done:
            Called when the delay phase ends (before interpolation starts).
        on_done:
            Called when interpolation completes.
        override_indices:
            DOF indices to override.  When ``None``, all DOFs are overridden
            (backward-compatible).  Pass e.g. ``[10, 11, …, 18]`` to only
            lerp the upper body while the active policy continues controlling
            the legs.
        """
        dl_steps = self._seconds_to_steps(delay_s)
        interp_steps = self._seconds_to_steps(duration_s)

        self._interp_state = "delaying" if dl_steps > 0 else "interping"
        self._interp_target_q = np.asarray(target_q, dtype=np.float64).reshape(1, -1)
        self._interp_start_q = None  # captured when interpolation actually begins
        self._interp_step = 0
        self._interp_delay_steps = dl_steps
        self._interp_total_steps = max(interp_steps, 1)
        self._interp_on_delay_done = on_delay_done
        self._interp_on_done = on_done
        self._interp_kp = kp_override
        self._interp_kd = kd_override
        self._interp_override_indices = override_indices

    def _step_policy_interp(self, robot_state_data: np.ndarray):
        """Advance interpolation state machine.

        Called from ``policy_action()``.  Returns ``(q_target, kp, kd)``
        when interpolation is active, or ``None`` when idle.
        """
        if self._interp_state == "idle":
            return None

        if self._interp_state == "delaying":
            # Hold current pose while we wait
            q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
            self._interp_step += 1
            if self._interp_step >= self._interp_delay_steps:
                self._interp_state = "interping"
                self._interp_step = 0
                if self._interp_on_delay_done is not None:
                    self._interp_on_delay_done()
                    # Callback may have switched policy – respect a cancel
                    if self._interp_state == "idle":
                        return None
            return (q, self._interp_kp, self._interp_kd, self._interp_override_indices)

        if self._interp_state == "interping":
            if self._interp_start_q is None:
                self._interp_start_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()

            alpha = min(self._interp_step / max(self._interp_total_steps, 1), 1.0)
            # ease-in-out: alpha' = alpha² * (3 - 2*alpha)
            alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            q = (1.0 - alpha_smooth) * self._interp_start_q + alpha_smooth * self._interp_target_q
            self._interp_step += 1

            if self._interp_step >= self._interp_total_steps:
                self._interp_state = "idle"
                self._interp_start_q = None
                self._interp_target_q = None
                if self._interp_on_done is not None:
                    self._interp_on_done()

            return (q, self._interp_kp, self._interp_kd, self._interp_override_indices)

        return None

    def _cancel_policy_interp(self):
        """Abort any in-progress policy-level interpolation."""
        self._interp_state = "idle"
        self._interp_start_q = None
        self._interp_target_q = None

    def _get_manual_command(self, robot_state_data):
        """Optional manual command when policy control is disabled."""
        return

    def _seconds_to_steps(self, seconds: float) -> int:
        """Convert a transition duration in seconds to policy control steps."""
        return max(0, int(round(max(0.0, seconds) * self.rl_rate)))

    def _begin_target_transition(
        self,
        name: str,
        steps: int,
        start_q: np.ndarray | None = None,
    ) -> None:
        """Start blending outgoing joint targets into the next target stream."""
        if steps <= 0:
            self._target_transition = None
            return

        start = None if start_q is None else np.asarray(start_q, dtype=np.float64).reshape(1, self.num_dofs).copy()
        self._target_transition = {
            "name": name,
            "step": 0,
            "steps": steps,
            "start_q": start,
        }

    def _apply_target_transition(self, q_target: np.ndarray, robot_state_data: np.ndarray) -> np.ndarray:
        """Blend the current target from the last command/current pose into ``q_target``."""
        transition = self._target_transition
        if transition is None:
            return q_target

        start_q = transition["start_q"]
        if start_q is None:
            if self._last_sent_q is not None:
                start_q = self._last_sent_q.reshape(1, self.num_dofs).copy()
            else:
                start_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
            transition["start_q"] = start_q

        step = int(transition["step"])
        steps = int(transition["steps"])
        alpha = min(1.0, (step + 1) / steps)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        blended = start_q + (q_target - start_q) * alpha

        transition["step"] = step + 1
        if step + 1 >= steps:
            self._target_transition = None

        return blended

    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

    # ============================================================================
    # Velocity Hook
    # ============================================================================

    def _apply_velocity(self, vc: VelCmd) -> None:
        """Apply a velocity command to the policy state.

        Called from the run loop when a provider returns a non-None VelCmd.
        Subclasses can override to add gating (e.g. stand_command in locomotion).
        """
        self.lin_vel_command[0] = vc.lin_vel
        self.ang_vel_command[0, 0] = vc.ang_vel

    # ============================================================================
    # Command Dispatch
    # ============================================================================

    def _dispatch_command(self, cmd):
        """Dispatch a command enum to the appropriate handler.

        Subclasses override this to handle policy-specific commands,
        calling ``super()._dispatch_command(cmd)`` for unhandled ones.
        """
        if cmd == StateCommand.START:
            self._handle_start_policy()
        elif cmd == StateCommand.STOP:
            self._handle_stop_policy()
        elif cmd == StateCommand.INIT:
            self._handle_init_state()
        elif cmd == StateCommand.KILL:
            self.logger.info(colored("Killing program via command", "red"))
            sys.exit(0)
        elif cmd == StateCommand.NEXT_POLICY:
            next_index = (self.active_policy_index + 1) % len(self.model_paths)
            self._activate_policy(next_index)
        elif cmd in STATE_COMMAND_TO_POLICY_INDEX:
            index = STATE_COMMAND_TO_POLICY_INDEX[cmd]
            if index != self.active_policy_index and 0 <= index < len(self.model_paths):
                self._activate_policy(index)
        elif cmd == StateCommand.KP_UP:
            self.interface.kp_level += 0.1
        elif cmd == StateCommand.KP_DOWN:
            self.interface.kp_level -= 0.1
        elif cmd == StateCommand.KP_UP_FINE:
            self.interface.kp_level += 0.01
        elif cmd == StateCommand.KP_DOWN_FINE:
            self.interface.kp_level -= 0.01
        elif cmd == StateCommand.KP_RESET:
            self.interface.kp_level = 1.0

    # ============================================================================
    # Control Action Methods
    # ============================================================================

    def _handle_start_policy(self):
        """Handle start policy action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self.logger.info(colored("Using policy actions", "blue"))
        self.phase = np.array([[0.0, np.pi]])
        self._begin_target_transition(
            "policy_start",
            self._seconds_to_steps(self.config.task.policy_start_transition_s),
        )
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self.logger.info("Actions set to zero")
        self._begin_target_transition(
            "policy_stop",
            self._seconds_to_steps(self.config.task.policy_stop_transition_s),
        )
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1

    def _handle_init_state(self):
        """Handle initialization state."""
        self.get_ready_state = True
        self.init_count = 0
        self.logger.info("Setting to init state")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _print_control_status(self):
        """Print current control status."""
        self.logger.info("------------ Control Status ------------")
        if self.active_model_path:
            total = len(self.model_paths)
            name = Path(self.active_model_path).name
            debug_str = (
                f"Active policy [{self.active_policy_index + 1}/{total}]: {name} Kp level {self.interface.kp_level:.2f}"
            )
            self.logger.info(debug_str)

    # ============================================================================
    # Main Run Method
    # ============================================================================

    def run(self):
        """Main run loop for the policy."""
        try:
            #一直循环执行以下步骤：
            for it in itertools.count():
                self.latency_tracker.start_cycle()
                #读取输入，比如手柄或者键盘输入
                vc = self._velocity_input.poll_velocity()
                if vc is not None:
                    #更新当前的速度命令状态，这可能会影响后续的观察处理或者直接影响控制命令
                    self._apply_velocity(vc)
                #读取控制命令输入，这些命令可能包括切换策略、调整控制增益或者其他预定义的命令
                commands = self._command_provider.poll_commands()
                for cmd in commands:
                    #执行命令分发，调用相应的处理函数来响应每个命令
                    self._dispatch_command(cmd)
                if commands:
                    self._print_control_status()
                if self.use_phase:
                    #更新步态周期时间，这通常涉及根据当前时间和预定义的步态周期计算相位变量，以便在观察处理或者策略推理中使用
                    self.update_phase_time()
                #如果启用了策略控制，并且当前不处于准备状态，则执行策略推理，计算新的动作命令
                self.policy_action()

                self.latency_tracker.end_cycle()

                if it % 50 == 0 and self.use_policy_action:
                    debug_str = f"RL FPS: {self.latency_tracker.get_fps():.2f} | {self.latency_tracker.get_stats_str()}"
                    self.logger.info(debug_str, flush=True)

                self.rate.sleep()

        except KeyboardInterrupt:
            pass
