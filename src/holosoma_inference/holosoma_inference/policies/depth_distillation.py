"""Depth distillation policy for vision-based locomotion over rough terrain (stairs).

Composes two ONNX models trained as a distilled student:

- ``depth_backbone.onnx``: depth image ``(1, [T,] H, W)`` -> depth latent ``(1, D)``
- ``student.onnx``: ``obs`` -> actions, where
  ``obs = [actor_obs | velocity_command_one_hot | depth_latent]``

Depth frames arrive from an external image server through the injected
``"depth"`` sensor (shared memory by default). Unlike the blind policies, the
observation vector is assembled here rather than by ``BasePolicy`` alone,
because the depth latent must be appended *after* the backbone runs.

Two properties of the checkpoints drive this implementation and are both read
from ONNX metadata rather than hard-coded:

1. ``joint_names`` — the training joint order is IsaacLab's breadth-first
   ordering, which differs from the robot's canonical DOF order, so
   observations are permuted into model order and actions back into robot order.
2. ``action_scale`` — the student emits actions at this scale.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_DIRECTION_COMMANDS
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.sensors.depth_heartbeat import STATE_FAILED
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import quat_from_angle_axis, quat_mul, quat_rotate_inverse


class DepthDistillationPolicy(LocomotionPolicy):
    """Vision-based locomotion policy composing a depth backbone with a student MLP.

    Velocity is commanded as a discrete one-hot class (not continuous lin/ang
    velocity), matching how the command was represented during training. The
    ``CMD_CODES`` table maps a direction to its one-hot index per speed mode.
    """

    # Command index per (speed_mode, direction), matching the training command set.
    # speed_mode 0 = low, 1 = high. The command one-hot stays 15-wide (see
    # ``obs_dims["velocity_command"]``); the indices beyond those used here are simply never set.
    CMD_CODES: dict[int, dict[str, int]] = {
        0: {"stand": 0, "forward": 1, "left_45": 2, "left_90": 3, "right_45": 4, "right_90": 5, "back": 11},
        1: {"stand": 0, "forward": 6, "left_45": 7, "left_90": 8, "right_45": 9, "right_90": 10, "back": 11},
    }
    SPEED_MODE_LABELS = ("LOW", "HIGH")

    # Bind w/a/s/d/q/e to absolute heading commands instead of the default
    # velocity accumulator. Read by the input factory when building the keyboard
    # provider: without this, each press only nudges lin/ang velocity by 0.1, so
    # reversing direction takes several presses to cross zero and the policy
    # looks unresponsive.
    KEYBOARD_DIRECTION_KEYS = KEYBOARD_DIRECTION_COMMANDS

    # Direction keys are MOMENTARY: the robot moves while a key is held and returns to stand on
    # release, like a gamepad d-pad and like the reference deployment. Latching (one tap keeps
    # walking until another key is pressed) is easy to run into a wall with, since letting go of the
    # keyboard does not stop the robot. Needs key-up events; the input factory falls back to
    # latching when they are unavailable (no pynput / no DISPLAY).
    KEYBOARD_HOLD_DIRECTIONS = True

    # StateCommand -> CMD_CODES direction name.
    COMMAND_TO_DIRECTION: dict[StateCommand, str] = {
        StateCommand.MOVE_FORWARD: "forward",
        StateCommand.MOVE_BACKWARD: "back",
        StateCommand.MOVE_LEFT_45: "left_45",
        StateCommand.MOVE_RIGHT_45: "right_45",
        StateCommand.MOVE_LEFT_90: "left_90",
        StateCommand.MOVE_RIGHT_90: "right_90",
    }

    # Velocity-vector -> discrete direction, by angle sector. Applied only when a
    # continuous VelCmd arrives (joystick/ROS2); keyboard uses the direct
    # mapping above.
    VELOCITY_DEADZONE = 0.05

    # Blend duration (control ticks) for easing into the stiff-hold pose.
    STIFF_BLEND_TICKS = 100

    def __init__(self, config: InferenceConfig):
        self.speed_mode = 0
        self._stiff_hold_active = True
        self._damping_mode_active = False
        self._stiff_blend_count = 0
        self._stiff_blend_start_q: np.ndarray | None = None

        # Populated by _init_policy_components.
        self.depth_backbone_session: onnxruntime.InferenceSession | None = None
        self.depth_backbone_input_name: str | None = None
        self.depth_backbone_output_name: str | None = None
        self.depth_latent_dim: int | None = None
        self.depth_image_shape: tuple[int, int] | None = None
        self.depth_buffer_len = 1
        self._model_joint_names: list[str] | None = None
        self._real2model_index: list[int] | None = None
        self._model2real_index: list[int] | None = None

        # Discrete velocity command as a one-hot vector. Dimension comes from the
        # observation config because it is part of the student's input contract.
        self.velocity_command_dim = config.observation.obs_dims.get("velocity_command", 0)
        self.velocity_command = np.zeros((1, self.velocity_command_dim), dtype=np.float32)
        self.active_velocity_command_idx = self.CMD_CODES[0]["stand"]

        # Motion recording, armed with the `c` key. The input thread only sets
        # _record_toggle_request; the control loop owns the buffers.
        self._recording = False
        self._record_toggle_request = False
        self._record_qpos: list[np.ndarray] = []
        self._record_vel_cmd: list[np.ndarray] = []

        super().__init__(config)

        self.set_velocity_command(self.CMD_CODES[0]["stand"], announce=False)
        self._init_stiff_hold_params(config)
        self._init_depth_sensor()

        depth_history_len = config.observation.history_length_dict.get("depth_obs", 1)
        self.depth_frame_buffer: deque[np.ndarray] = deque(maxlen=max(depth_history_len, self.depth_buffer_len))

    # ============================================================================
    # Initialization
    # ============================================================================

    def _init_stiff_hold_params(self, config: InferenceConfig):
        """Load the stiff-hold pose and gains used before the policy is started."""
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is None or config.robot.stiff_startup_kd is None:
            raise ValueError(
                "Robot config must specify stiff_startup_kp and stiff_startup_kd for DepthDistillationPolicy"
            )
        self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError(
                f"stiff_startup_pos has {self._stiff_hold_q.shape[1]} entries but robot has {self.num_dofs} DOFs"
            )

    def _init_depth_sensor(self):
        """Resolve the depth sensor: injected by a service, or shared memory by default."""
        camera_config = self.config.camera
        if camera_config is None:
            raise ValueError("DepthDistillationPolicy requires a camera config (--camera or a preset that sets it)")

        self._depth_stale_warned = False

        injected = getattr(self, "_injected_sensors", None) or {}
        if "depth" in injected:
            self._depth_sensor = injected["depth"]
            logger.info("[DepthDistillation] using injected depth sensor")
            return

        from holosoma_inference.sensors.depth_heartbeat import default_heartbeat_endpoint
        from holosoma_inference.sensors.depth_shm import (
            DepthShmSensor,
        )

        props = camera_config.props
        shape = (camera_config.num_cameras, 1, props.resized_height, props.resized_width)
        shm_config = self.config.task.depth_shm
        heartbeat_endpoint = None
        if shm_config.heartbeat:
            heartbeat_endpoint = shm_config.heartbeat_endpoint or default_heartbeat_endpoint(shm_config.name)
        self._depth_sensor = DepthShmSensor(
            shape=shape, name=shm_config.name, required=shm_config.required, heartbeat_endpoint=heartbeat_endpoint
        )
        self._depth_sensor.start()

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Load the depth backbone and the student model.

        Overrides the base single-model path: ``model_path`` here is an ordered
        pair, not a list of switchable policies.
        """
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate

        paths = self._collect_model_paths(model_path)
        if len(paths) != 2:
            raise ValueError(
                f"DepthDistillationPolicy requires exactly 2 model paths "
                f"[depth_backbone.onnx, student.onnx], got {len(paths)}: {paths}"
            )
        self.model_paths = [self._resolve_model_path(str(p)) for p in paths]

        self._load_depth_backbone(self.model_paths[0])
        self._load_student_model(self.model_paths[1])
        self._setup_joint_reordering()
        self._init_waist_joint_indices()

        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))

        # The backbone and student are one composite policy, not two switchable
        # ones. Collapse model_paths to the student so the base class's
        # switch-policy commands see a single slot and can't index past
        # _policy_states (which has exactly one entry).
        self._policy_states = [self._capture_policy_state()]
        self.active_policy_index = 0
        self.active_model_path = self.model_paths[1]
        self.model_paths = [self.model_paths[1]]

        self._resolve_control_gains()

    def _load_depth_backbone(self, model_path: str):
        """Load the depth backbone and infer its input geometry and latent size."""
        self.depth_backbone_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = self.depth_backbone_session.get_inputs()
        outputs = self.depth_backbone_session.get_outputs()
        self.depth_backbone_input_name = inputs[0].name
        self.depth_backbone_output_name = outputs[0].name

        shape = inputs[0].shape
        if len(shape) == 4:
            # (batch, frames, H, W) — multi-frame stack
            self.depth_buffer_len, height, width = shape[1], shape[2], shape[3]
        elif len(shape) == 3:
            # (batch, H, W) — single frame
            self.depth_buffer_len, height, width = 1, shape[1], shape[2]
        else:
            raise ValueError(f"Unexpected depth backbone input shape {shape}; expected 3 or 4 dims")
        self.depth_image_shape = (height, width)
        self.depth_latent_dim = outputs[0].shape[-1]

        props = self.config.camera.props
        if (height, width) != (props.resized_height, props.resized_width):
            raise ValueError(
                f"Depth backbone expects {(height, width)} but camera config produces "
                f"{(props.resized_height, props.resized_width)}. These must match."
            )

        logger.info(
            f"[DepthDistillation] backbone: input={shape} -> latent_dim={self.depth_latent_dim}, "
            f"frames={self.depth_buffer_len}"
        )

    def _load_student_model(self, model_path: str):
        """Load the student model and read gains, joint order and action scale from metadata."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.onnx_input_names = [i.name for i in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.onnx_policy_session.get_outputs()]

        metadata = {p.key: self._decode_metadata_value(p.value) for p in onnx.load(model_path).metadata_props}

        # These checkpoints name the gain keys joint_stiffness/joint_damping;
        # accept the kp/kd spelling too so either export convention works. These
        # only apply when the robot config leaves motor_kp/motor_kd unset —
        # _resolve_control_gains prefers the config, and the distillation robot
        # preset deliberately pins gains there.
        kp = metadata.get("kp", metadata.get("joint_stiffness"))
        kd = metadata.get("kd", metadata.get("joint_damping"))
        self.onnx_kp = np.array(self._as_float_list(kp)) if kp is not None else None
        self.onnx_kd = np.array(self._as_float_list(kd)) if kd is not None else None

        joint_names = metadata.get("joint_names")
        if isinstance(joint_names, str):
            joint_names = [n.strip() for n in joint_names.split(",")]
        self._model_joint_names = joint_names

        if "action_scale" in metadata:
            scale = metadata["action_scale"]
            if isinstance(scale, (int, float)):
                self.policy_action_scale = float(scale)
            else:
                scale_values = np.asarray(self._as_float_list(scale), dtype=np.float32)
                if scale_values.shape != (self.num_dofs,):
                    raise ValueError(
                        f"action_scale metadata must be scalar or have {self.num_dofs} entries, "
                        f"got shape {scale_values.shape}."
                    )
                self.policy_action_scale = scale_values
            logger.info(f"[DepthDistillation] action_scale from metadata: {self.policy_action_scale}")

        # Gains arrive in model joint order; reorder to robot order before use.
        if self.onnx_kp is not None and self._model_joint_names is not None:
            model_to_real = get_index_of_a_in_b(list(self.robot_config.dof_names), self._model_joint_names)
            self.onnx_kp = self.onnx_kp[model_to_real]
            self.onnx_kd = self.onnx_kd[model_to_real]

        def policy_act(input_feed):
            return self.onnx_policy_session.run(self.onnx_output_names, input_feed)

        self.policy = policy_act
        logger.info(f"[DepthDistillation] student: inputs={self.onnx_input_names}")

    @staticmethod
    def _decode_metadata_value(value):
        """Decode one ONNX metadata value, falling back to the raw string.

        Exporters write some entries as JSON and others as bare comma-separated
        text, so a parse failure is expected rather than an error.
        """
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @staticmethod
    def _as_float_list(value):
        """Coerce an ONNX metadata gain entry (list or comma string) to floats."""
        if isinstance(value, str):
            return [float(x) for x in value.split(",")]
        return [float(x) for x in value]

    def _setup_joint_reordering(self):
        """Build permutations between the robot's DOF order and the model's joint order.

        ``_real2model_index`` reorders robot-order observations into model order;
        ``_model2real_index`` reorders model-order actions back to robot order.
        """
        real_joint_names = list(self.robot_config.dof_names)
        if self._model_joint_names is None or list(self._model_joint_names) == real_joint_names:
            self._real2model_index = None
            self._model2real_index = None
            return

        self._real2model_index = get_index_of_a_in_b(self._model_joint_names, real_joint_names)
        self._model2real_index = get_index_of_a_in_b(real_joint_names, self._model_joint_names)
        if isinstance(self.policy_action_scale, np.ndarray):
            self.policy_action_scale = self.policy_action_scale[self._model2real_index]
        logger.info("[DepthDistillation] joint reordering enabled (model order != robot order)")

    def _init_waist_joint_indices(self):
        """Cache the waist chain used to derive the torso (anchor) orientation.

        Training observed ``robot_anchor_projected_gravity`` in the torso frame.
        On hardware the IMU sits in the torso so the base quaternion already is
        the torso's; in MuJoCo the floating base is the pelvis, so the waist
        yaw/roll/pitch rotations must be chained on.
        """
        waist_chain = (
            ("waist_yaw_joint", np.array([0.0, 0.0, 1.0])),
            ("waist_roll_joint", np.array([1.0, 0.0, 0.0])),
            ("waist_pitch_joint", np.array([0.0, 1.0, 0.0])),
        )
        self._waist_joint_info = [
            (self.dof_names.index(name), axis) for name, axis in waist_chain if name in self.dof_names
        ]

    # ============================================================================
    # Observations
    # ============================================================================

    def _get_anchor_quat(self, robot_state_data, base_quat):
        """Return the torso-frame quaternion by chaining waist joints onto the base."""
        if not self._waist_joint_info:
            return base_quat
        raw_dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        anchor_quat = base_quat.copy()
        for idx, axis in self._waist_joint_info:
            anchor_quat = quat_mul(anchor_quat, quat_from_angle_axis(float(raw_dof_pos[0, idx]), axis))
        return anchor_quat

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Build the proprioceptive terms in the model's joint order."""
        obs = super().get_current_obs_buffer_dict(robot_state_data)

        # projected_gravity is observed in the anchor (torso) frame, not the base frame.
        anchor_quat = self._get_anchor_quat(robot_state_data, robot_state_data[:, 3:7])
        obs["projected_gravity"] = quat_rotate_inverse(anchor_quat, np.array([[0.0, 0.0, -1.0]]))

        if self._real2model_index is not None:
            obs["dof_pos"] = obs["dof_pos"][:, self._real2model_index]
            obs["dof_vel"] = obs["dof_vel"][:, self._real2model_index]

        # The declared ``depth_obs`` group documents the camera's contribution to
        # the observation space, but depth reaches the student as a latent rather
        # than raw pixels. Supply a zero placeholder so the generic group
        # machinery has a value for every declared term; the flattened depth_obs
        # buffer it produces is never read.
        props = self.config.camera.props
        obs["cam_depth"] = np.zeros((1, props.resized_height * props.resized_width), dtype=np.float32)

        return obs

    def _get_depth_latent(self) -> np.ndarray:
        """Read the latest depth frame(s) and encode them with the backbone."""
        frames = self._depth_sensor.get_latest()
        # Sensor yields (num_cameras, channels, H, W); this policy uses one camera.
        frame = np.asarray(frames, dtype=np.float32)[0, 0]
        self.depth_frame_buffer.append(frame.copy())

        stack = list(self.depth_frame_buffer)
        if len(stack) < self.depth_buffer_len:
            # Pad by repeating the oldest frame so the stack is full from tick one.
            stack = [stack[0]] * (self.depth_buffer_len - len(stack)) + stack
        stack = stack[-self.depth_buffer_len :]

        depth_input = np.stack(stack, axis=0)[None] if self.depth_buffer_len > 1 else stack[-1][None]
        outputs = self.depth_backbone_session.run(
            [self.depth_backbone_output_name],
            {self.depth_backbone_input_name: depth_input.astype(np.float32)},
        )
        return outputs[0]

    # ============================================================================
    # Motion recording
    # ============================================================================

    def _service_recording(self, robot_state_data):
        """Append one frame when recording is armed.

        Called from the control loop (not the input thread) so buffer mutation
        stays single-threaded; the ``c`` key only sets an intent flag.
        """
        if self._record_toggle_request:
            self._record_toggle_request = False
            if self._recording:
                self._stop_and_save_recording()
            else:
                self._start_recording()
        if not self._recording:
            return

        qpos = np.empty(7 + self.num_dofs, dtype=np.float32)
        # Quaternion first, then root position — this layout is what the clip
        # consumer expects, and deliberately differs from the pos-then-quat
        # order used by MuJoCo-format training clips.
        qpos[0:4] = robot_state_data[0, 3:7]
        qpos[4:7] = robot_state_data[0, 0:3]
        qpos[7 : 7 + self.num_dofs] = robot_state_data[0, 7 : 7 + self.num_dofs]
        self._record_qpos.append(qpos)
        # Copied because set_velocity_command mutates the array in place.
        self._record_vel_cmd.append(self.velocity_command[0].copy())

    def _start_recording(self):
        self._record_qpos = []
        self._record_vel_cmd = []
        self._recording = True
        logger.info(colored("[REC] recording started", "green"))

    def _stop_and_save_recording(self):
        """Write the buffered clip to ``recorded_motion/<dir>/<label>_duration<X.X>s_motion.npz``."""
        self._recording = False
        if not self._record_qpos:
            logger.warning("[REC] nothing recorded — no file written")
            return

        fps = round(self.rl_rate)
        duration = len(self._record_qpos) / fps
        qpos = np.stack(self._record_qpos, axis=0).astype(np.float32)
        vel_cmd = np.stack(self._record_vel_cmd, axis=0).astype(np.float32)

        record_dir = self.config.task.record_dir or "default_run"
        record_label = self.config.task.record_label or "clip"
        out_dir = Path("recorded_motion") / record_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{record_label}_duration{duration:.1f}s_motion.npz"
        if out_path.exists():
            logger.warning(f"[REC] overwriting {out_path}")
        np.savez(out_path, qpos=qpos, vel_cmd=vel_cmd, fps=np.int64(fps))
        logger.info(colored(f"[REC] saved {out_path} ({qpos.shape[0]} frames, {duration:.1f}s)", "green"))

    def _autosave_if_recording(self):
        """Flush an in-flight recording when leaving policy control.

        ``_service_recording`` only runs while the policy drives, so without this
        a recording left running at stop/stiff-hold would be silently lost.
        """
        if self._recording:
            logger.info("[REC] autosaving on exit from policy control")
            self._stop_and_save_recording()

    def prepare_obs_for_rl(self, robot_state_data):
        """Assemble ``[actor_obs | velocity_command | depth_latent]`` for the student."""
        self._service_recording(robot_state_data)
        self._prepare_group_observations(robot_state_data)
        actor_obs = self.obs_buf_dict["actor_obs"]

        parts = [actor_obs]
        if self.velocity_command_dim > 0:
            parts.append(self.velocity_command)
        parts.append(self._get_depth_latent())
        obs = np.concatenate(parts, axis=1).astype(np.float32)

        input_feed = {"obs": obs}
        if "time_step" in self.onnx_input_names:
            input_feed["time_step"] = np.zeros((1, 1), dtype=np.float32)
        return input_feed

    def rl_inference(self, robot_state_data):
        """Run the student and return the action in robot joint order.

        ``last_policy_action`` is kept in *model* order because it feeds back as
        the ``actions`` observation next tick; only the returned command is
        permuted to robot order.
        """
        input_feed = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations({"actor_obs": input_feed["obs"]})

        policy_action = np.clip(self.policy(input_feed)[0], -100, 100)
        self.last_policy_action = policy_action.copy()

        if self._model2real_index is not None:
            policy_action = policy_action[:, self._model2real_index]

        self.scaled_policy_action = policy_action * self.policy_action_scale
        if self.config.task.debug.force_zero_action:
            self.scaled_policy_action = np.zeros_like(self.scaled_policy_action)
        return self.scaled_policy_action

    # ============================================================================
    # Commands
    # ============================================================================

    def set_velocity_command(self, idx: int, announce: bool = True):
        """Set the active one-hot velocity command class."""
        if self.velocity_command_dim == 0:
            return
        self.velocity_command[:] = 0.0
        if 0 <= idx < self.velocity_command_dim:
            self.velocity_command[0, idx] = 1.0
            self.active_velocity_command_idx = idx
            if announce:
                logger.info(colored(f"Velocity command -> index {idx} ({self._command_name(idx)})", "blue"))
        else:
            self.active_velocity_command_idx = self.CMD_CODES[0]["stand"]
            self.velocity_command[0, self.active_velocity_command_idx] = 1.0

    def _command_name(self, idx: int) -> str:
        """Reverse-lookup a command index to its direction name."""
        for name, code in self.CMD_CODES[self.speed_mode].items():
            if code == idx:
                return name
        return "?"

    def _apply_velocity(self, vc: VelCmd) -> None:
        """Quantize a continuous velocity command into a discrete direction class.

        Keyboard/joystick/ROS2 all produce continuous velocities, but this policy
        was trained on discrete direction classes, so the vector is mapped to the
        nearest sector. Sector boundaries mirror the training command set.
        """
        if self.velocity_command_dim == 0:
            return

        linear_x, linear_y = float(vc.lin_vel[0]), float(vc.lin_vel[1])
        angular_z = float(vc.ang_vel)
        codes = self.CMD_CODES[self.speed_mode]

        if math.hypot(linear_x, linear_y) < self.VELOCITY_DEADZONE and abs(angular_z) < self.VELOCITY_DEADZONE:
            cmd_idx = codes["stand"]
        elif abs(linear_x) < self.VELOCITY_DEADZONE and abs(linear_y) < self.VELOCITY_DEADZONE:
            # Pure yaw -> a 90° turn in the commanded direction.
            cmd_idx = codes["left_90"] if angular_z > 0 else codes["right_90"]
        else:
            angle_deg = math.degrees(math.atan2(linear_y, linear_x))
            if -22.5 <= angle_deg <= 22.5:
                cmd_idx = codes["forward"]
            elif 22.5 < angle_deg <= 67.5:
                cmd_idx = codes["left_45"]
            elif 67.5 < angle_deg <= 112.5:
                cmd_idx = codes["left_90"]
            elif -67.5 <= angle_deg < -22.5:
                cmd_idx = codes["right_45"]
            elif -112.5 <= angle_deg < -67.5:
                cmd_idx = codes["right_90"]
            else:
                cmd_idx = codes["back"]

        if cmd_idx != self.active_velocity_command_idx:
            self.set_velocity_command(cmd_idx)

    def _dispatch_command(self, cmd):
        """Handle discrete commands; STAND_TOGGLE toggles the speed mode here."""
        if cmd in self.COMMAND_TO_DIRECTION:
            # Absolute heading: one press selects it outright.
            self.set_velocity_command(self.CMD_CODES[self.speed_mode][self.COMMAND_TO_DIRECTION[cmd]])
        elif cmd == StateCommand.STAND_TOGGLE:
            self.speed_mode = (self.speed_mode + 1) % len(self.SPEED_MODE_LABELS)
            logger.info(colored(f"Speed mode -> {self.SPEED_MODE_LABELS[self.speed_mode]}", "blue"))
        elif cmd == StateCommand.ZERO_VELOCITY:
            self.set_velocity_command(self.CMD_CODES[self.speed_mode]["stand"])
        elif cmd in (StateCommand.WALK, StateCommand.STAND):
            direction = "forward" if cmd == StateCommand.WALK else "stand"
            self.set_velocity_command(self.CMD_CODES[self.speed_mode][direction])
        elif cmd == StateCommand.TOGGLE_RECORDING:
            # Serviced by the control loop, not applied here.
            self._record_toggle_request = True
        else:
            # Skip LocomotionPolicy's continuous-velocity handling entirely.
            BasePolicy._dispatch_command(self, cmd)

    def policy_action(self):
        # Checked before the base class snapshots use_policy_action, so a stalled depth
        # producer stops the policy on this tick instead of after one more stale action.
        self._check_depth_liveness()
        super().policy_action()

    def _check_depth_liveness(self) -> None:
        """Warn when the depth frame stops advancing, and enter stiff hold if it stays stale.

        Keeping the last frame and stopping is safer than substituting zeros: a
        normalized zero is a valid depth reading to the policy, not "no data".
        """
        liveness_fn = getattr(self._depth_sensor, "liveness", None)
        status = liveness_fn() if liveness_fn is not None else None
        if status is None or not status.checked:
            return  # heartbeat disabled, or a producer that does not publish one

        cfg = self.config.task.depth_shm
        failed = status.state == STATE_FAILED
        if failed or status.stale_s > cfg.stale_warn_s:
            if not self._depth_stale_warned:
                logger.warning(
                    f"[DepthDistillation] depth frame stale for {status.stale_s:.2f}s (producer: {status.state})"
                )
                self._depth_stale_warned = True
        elif self._depth_stale_warned:
            logger.info("[DepthDistillation] depth frames advancing again")
            self._depth_stale_warned = False

        if (failed or status.stale_s > cfg.stale_stop_s) and self.use_policy_action:
            logger.error(
                colored(
                    f"Depth producer {status.state}, frame {status.stale_s:.2f}s old: entering stiff hold. "
                    "Press ] to resume once depth recovers.",
                    "red",
                )
            )
            self._handle_init_state()

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._damping_mode_active = False
        logger.info(colored("Depth distillation policy started", "green"))

    def _handle_stop_policy(self):
        """Enter damping mode (Kp=0, Kd>0) rather than the base zero-action stop.

        Deliberately does not call ``super()``: the base implementation sets
        ``interface.no_action = 1``, which forces both Kp and Kd to zero in the
        command sender and would let the robot collapse.
        """
        self._autosave_if_recording()
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = False
        self._damping_mode_active = True
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.depth_frame_buffer.clear()
        self.set_velocity_command(self.CMD_CODES[0]["stand"], announce=False)
        logger.info(colored("Damping mode (Kp=0, Kd>0)", "yellow"))

    def _handle_init_state(self):
        """Re-enter stiff hold, easing back to the startup pose."""
        self._autosave_if_recording()
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._damping_mode_active = False
        self._stiff_blend_count = 0
        self._stiff_blend_start_q = None
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.depth_frame_buffer.clear()
        self.set_velocity_command(self.CMD_CODES[0]["stand"], announce=False)
        logger.info(colored("Entering stiff hold", "yellow"))

    def _get_manual_command(self, robot_state_data):
        """Command applied while the policy is not driving.

        Stiff hold eases pose and gains from the robot's current state to the
        startup pose so enabling stiffness never snaps the joints. Damping mode
        holds Kp=0 with damping so the robot yields but resists free-fall.
        """
        if self._stiff_hold_active:
            if self._stiff_blend_start_q is None:
                self._stiff_blend_start_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()

            t = min(self._stiff_blend_count / self.STIFF_BLEND_TICKS, 1.0)
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
            self._stiff_blend_count += 1
            return {
                "q": (1 - alpha) * self._stiff_blend_start_q + alpha * self._stiff_hold_q,
                "kp": alpha * self._stiff_hold_kp,
                "kd": alpha * self._stiff_hold_kd,
            }

        if self._damping_mode_active:
            return {
                "q": robot_state_data[:, 7 : 7 + self.num_dofs],
                "kp": np.zeros_like(self._stiff_hold_kp),
                "kd": self._stiff_hold_kd,
            }
        return None

    def _capture_policy_state(self) -> dict:
        state = super()._capture_policy_state()
        state["depth_backbone_session"] = self.depth_backbone_session
        return state

    def _restore_policy_state(self, state: dict):
        super()._restore_policy_state(state)
        self.depth_backbone_session = state["depth_backbone_session"]

    def _print_control_status(self):
        """Report the active direction, speed mode and control mode."""
        if self.active_model_path:
            logger.info(f"Active policy: {Path(self.active_model_path).name}")
        mode = (
            "stiff_hold"
            if self._stiff_hold_active
            else "damping"
            if self._damping_mode_active
            else "policy"
            if self.use_policy_action
            else "idle"
        )
        print(
            f"Command: {self._command_name(self.active_velocity_command_idx)} "
            f"(idx={self.active_velocity_command_idx}) | "
            f"Speed: {self.SPEED_MODE_LABELS[self.speed_mode]} | Mode: {mode}"
        )
        print("💡 Keys: w/s/a/d/q/e (direction) | z (stand) | = (speed mode) | ] start | o damp | i stiff")
