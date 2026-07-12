from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.policies.wbt_utils import MotionClockUtil, PinocchioRobot, TimestepUtil
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class WholeBodyTrackingPolicy(BasePolicy):
    #初始化时间步
    def __init__(self, config: InferenceConfig):
        self.config = config

        # initialize motion state
        self.motion_clip_progressing = False
        self._dual_mode_active = False  # set True by DualModePolicy when wrapping
        self.curr_motion_timestep = config.task.motion_start_timestep
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Initialize clock for sim-time synchronization
        clock_sub = ClockSub()
        clock_sub.start()
        clock_util = MotionClockUtil(clock_sub)
        self.timestep_util = TimestepUtil(
            clock=clock_util,
            interval_ms=1000.0 / config.task.rl_rate,
            start_timestep=config.task.motion_start_timestep,
        )

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self._stiff_hold_confirmed = not self._requires_stiff_hold_start_confirmation(config)
        self._stiff_hold_transition_pending = self._stiff_hold_confirmed
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.per_joint_policy_action_scale: np.ndarray | None = None

        super().__init__(config)
        self._configure_action_scales()

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)
        self._stiff_hold_override_q = None

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        if not self._stiff_hold_confirmed:
            logger.warning(
                colored(
                    "Waiting for joystick Start to enter stiff hold. Robot will stay torque-free until confirmed.",
                    "yellow",
                    attrs=["bold"],
                )
            )
        elif hasattr(self, "_shared_hardware_source"):
            logger.info(colored("Skipping stiff hold prompt (secondary policy)", "yellow"))
        else:
            logger.info(colored("Entering stiff hold mode", "green"))

    @staticmethod
    def _requires_stiff_hold_start_confirmation(config: InferenceConfig) -> bool:
        mode = config.task.stiff_hold_confirmation
        if mode == "none":
            return False
        if mode == "start":
            return True
        return config.task.state_input in {"interface", "joystick"}

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)
    #加载onnx模型并提取相关信息
    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw at the configured start timestep
        time_step = np.array([[self.config.task.motion_start_timestep]], dtype=np.float32)

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "per_joint_policy_action_scale": self.per_joint_policy_action_scale.copy()
                if self.per_joint_policy_action_scale is not None
                else None,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        saved = state["per_joint_policy_action_scale"]
        self.per_joint_policy_action_scale = saved.copy() if saved is not None else None
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._configure_action_scales()

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos
    #每一帧获取当前的obs buffer dict，构造policy输入
    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        return current_obs_buffer_dict
    #调用policy进行推理，获取动作，并更新motion timestep
    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the configured start while waiting to trigger the clip.
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        if self.per_joint_policy_action_scale is None:
            self.scaled_policy_action = policy_action * self.policy_action_scale
        else:
            self.scaled_policy_action = policy_action * self.per_joint_policy_action_scale
        # update motion timestep
        self._set_motion_timestep()

        return self.scaled_policy_action

    def _configure_action_scales(self) -> None:
        """Configure action scales, prioritising ONNX metadata over config fallbacks.

        Resolution order:
        1. ONNX metadata ``action_scale`` (scalar or per-joint list)
        2. ``robot.default_per_joint_action_scale`` when
           ``task.action_scales_by_effort_limit_over_p_gain`` is True
        3. Fall back to the scalar ``task.policy_action_scale``
        """
        raw_metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        onnx_action_scale = self._parse_action_scale_metadata(raw_metadata.get("action_scale"))

        if onnx_action_scale is not None:
            scales = onnx_action_scale.astype(np.float32, copy=False).reshape(-1)
        elif self.config.task.action_scales_by_effort_limit_over_p_gain:
            fallback = self.config.robot.default_per_joint_action_scale
            if fallback is None:
                raise ValueError(
                    "task.action_scales_by_effort_limit_over_p_gain=True requires ONNX metadata key "
                    "'action_scale' (scalar or per-joint list) or "
                    "robot.default_per_joint_action_scale."
                )
            scales = np.asarray(fallback, dtype=np.float32).reshape(-1)
            logger.warning("ONNX metadata 'action_scale' missing; using robot.default_per_joint_action_scale.")
        else:
            self.per_joint_policy_action_scale = None
            return

        if scales.size == 1:
            scales = np.full(self.num_dofs, scales.item(), dtype=np.float32)
        elif scales.size != self.num_dofs:
            raise ValueError(f"Action scale must contain 1 or {self.num_dofs} values, got {scales.size}.")

        self.per_joint_policy_action_scale = scales.reshape(1, -1)

    @staticmethod
    def _parse_action_scale_metadata(raw_value: str | None) -> np.ndarray | None:
        """Parse action_scale metadata from JSON-serialized or CSV string formats."""
        if raw_value is None:
            return None

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value

        if isinstance(parsed, (int, float)):
            return np.array([float(parsed)], dtype=np.float32)
        if isinstance(parsed, str):
            values = [float(token.strip()) for token in parsed.split(",") if token.strip()]
            if not values:
                raise ValueError("ONNX metadata action_scale is an empty string.")
            return np.array(values, dtype=np.float32)

        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("ONNX metadata action_scale is empty.")
        return values

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        if not self._stiff_hold_confirmed:
            return {
                "q": robot_state_data[:, 7 : 7 + self.num_dofs].copy(),
                "kp": np.zeros(self.num_dofs, dtype=np.float32),
                "kd": np.zeros(self.num_dofs, dtype=np.float32),
            }
        if self._stiff_hold_transition_pending:
            self._begin_target_transition(
                "stiff_hold",
                self._seconds_to_steps(self.config.task.stiff_hold_transition_s),
                start_q=robot_state_data[:, 7 : 7 + self.num_dofs],
            )
            self._stiff_hold_transition_pending = False
        return {
            "q": (self._stiff_hold_override_q if self._stiff_hold_override_q is not None else self._stiff_hold_q).copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def hold_current_pose(self, robot_state_data):
        """Arm WBT in a non-moving hold target for safe mode switches."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._stiff_hold_confirmed = True
        self._stiff_hold_transition_pending = False
        self._stiff_hold_override_q = robot_state_data[:, 7 : 7 + self.num_dofs].astype(np.float32, copy=True)
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
    #当策略开始时]，捕获机器人和运动的偏航角，并将stiff hold状态设置为False
    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._stiff_hold_override_q = None
        self._stiff_hold_transition_pending = False
        self._stiff_hold_confirmed = True
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _set_motion_timestep(self):
        if self.motion_clip_progressing:
            prev = self.curr_motion_timestep

            if self.use_sim_time:
                self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
            else:
                self.curr_motion_timestep += 1

            if self.curr_motion_timestep != prev:
                self.logger.info(f"Motion timestep: {prev} → {self.curr_motion_timestep}")  # noqa: G004

            # Stop motion clip at configured end timestep (keep policy running at final pose)
            if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
                self.logger.info(colored(f"Reached end timestep {end}", "yellow"))
                self.motion_clip_progressing = False
                self.curr_motion_timestep = end
                if self.config.task.stop_on_motion_end and not self._dual_mode_active:
                    self._handle_stop_policy()
                elif self.config.task.stop_on_motion_end and self._dual_mode_active:
                    # Signal dual_mode to auto-return to loco
                    self.use_policy_action = False
    #当接收到停止策略的命令时o，重置相关状态并设置stiff hold为True
    def _handle_stop_policy(self):
        """Handle stop policy action.

        When running under DualModePolicy the transition back to locomotion
        is managed externally — we only disarm WBT here.
        """
        self.use_policy_action = False
        self.get_ready_state = False
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        if self._dual_mode_active:
            # DualModePolicy._return_to_primary() handles the transition.
            # Just disarm WBT without entering stiff-hold.
            return

        self._stiff_hold_active = True
        self._stiff_hold_confirmed = True
        self._stiff_hold_override_q = None
        self._stiff_hold_transition_pending = False
        self._begin_target_transition(
            "stiff_hold",
            self._seconds_to_steps(self.config.task.policy_stop_transition_s),
        )
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
    #当接收到开始运动剪辑的命令时m，重置时间步并设置相关状态
    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        if not self.use_policy_action:
            self._handle_start_policy()

        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.motion_clip_progressing = True

        if self.config.task.motion_start_timestep > 0 or self.config.task.motion_end_timestep is not None:
            start_str = str(self.config.task.motion_start_timestep)
            end_str = str(self.config.task.motion_end_timestep) if self.config.task.motion_end_timestep else "end"
            self.logger.info(colored(f"Starting motion clip from timestep {start_str} to {end_str}", "blue"))
        else:
            self.logger.info(colored("Starting motion clip", "blue"))

    def _dispatch_command(self, cmd):
        from holosoma_inference.inputs.api.commands import StateCommand

        if not self._stiff_hold_confirmed:
            if cmd == StateCommand.STAND_TOGGLE:
                self._stiff_hold_confirmed = True
                self._stiff_hold_transition_pending = True
                self.logger.info(colored("Joystick Start confirmed: entering stiff hold", "green"))
            elif cmd == StateCommand.KILL:
                super()._dispatch_command(cmd)
            else:
                self.logger.warning("Ignoring command until joystick Start confirms stiff hold.")
            return

        if cmd == StateCommand.START_MOTION_CLIP:
            self._handle_start_motion_clip()
        else:
            super()._dispatch_command(cmd)

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
