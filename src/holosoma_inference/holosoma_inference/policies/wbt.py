import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.split("}")[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        
        # Motion data for future_motion_targets (if needed)
        self.motion_data = None
        self.motion_fps = None

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

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

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

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

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Detect ONNX format: old style (obs, time_step) vs new style (actor_obs, future_motion_targets)
        self.use_new_onnx_format = "actor_obs" in self.onnx_input_names
        self.has_future_motion = "future_motion_targets" in self.onnx_input_names
        
        if self.use_new_onnx_format:
            logger.info("Using new ONNX format with actor_obs input")
            if self.has_future_motion:
                logger.info("ONNX model expects future_motion_targets input")
                # Load motion file for future_motion_targets computation
                self._load_motion_file()
        else:
            logger.info("Using legacy ONNX format with obs/time_step inputs")

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

        # Compute per-joint action scale from training config embedded in ONNX metadata.
        # Training uses: action_scale * effort_limit / kp  (per joint).
        # The inference TaskConfig has a flat policy_action_scale=1.0 which is wrong
        # for models trained with action_scales_by_effort_limit_over_p_gain=True.
        exp_cfg = metadata.get("experiment_config")
        if exp_cfg is not None:
            robot_ctrl = exp_cfg.get("robot", {}).get("control", {})
            if robot_ctrl.get("action_scales_by_effort_limit_over_p_gain", False):
                effort_limits = np.array(exp_cfg["robot"]["dof_effort_limit_list"], dtype=np.float32)
                kp = np.array(metadata["kp"], dtype=np.float32)
                base_scale = float(robot_ctrl.get("action_scale", 0.25))
                self.policy_action_scale = (effort_limits / kp * base_scale).reshape(1, -1)
                logger.info(
                    f"Using per-joint action scale from ONNX metadata "
                    f"(effort/kp*{base_scale}): min={self.policy_action_scale.min():.4f}, "
                    f"max={self.policy_action_scale.max():.4f})"
                )
            elif robot_ctrl.get("action_scales") is not None:
                self.policy_action_scale = np.array(robot_ctrl["action_scales"], dtype=np.float32).reshape(1, -1)
                logger.info(f"Using explicit per-joint action scales from ONNX metadata")
            else:
                base_scale = float(robot_ctrl.get("action_scale", self.policy_action_scale))
                self.policy_action_scale = base_scale
                logger.info(f"Using flat action scale from ONNX metadata: {base_scale}")

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        
        # Get initial command and ref quat xyzw based on ONNX format
        if self.use_new_onnx_format:
            # New format: direct policy inference without motion extraction
            # For initialization, we can't get motion command from new ONNX (it only outputs actions)
            if self.has_future_motion and self.motion_data is not None:
                # Initialize motion_command from first timestep of motion data
                joint_pos_0 = self.motion_data["joint_pos"][0]  # [num_dofs]
                joint_vel_0 = self.motion_data["joint_vel"][0]  # [num_dofs]
                self.motion_command_t = np.concatenate([
                    joint_pos_0[np.newaxis, :],
                    joint_vel_0[np.newaxis, :]
                ], axis=1).astype(np.float32)  # [1, 58]
                # Get ref_quat from motion data (use ref_body_index, typically torso_link)
                ref_body_idx = self.motion_data["ref_body_index"]
                self.ref_quat_xyzw_t = self.motion_data["body_quat_w"][0, ref_body_idx:ref_body_idx+1, :].astype(np.float32)  # [1, 4] xyzw
                logger.info(f"Initialized motion_command from motion data (ref_body_index={ref_body_idx})")
            else:
                # Fallback to dummy values if no motion data
                logger.warning("New ONNX format doesn't support motion extraction - using default initialization")
                self.motion_command_t = np.zeros((1, 58), dtype=np.float32)  # dummy
                self.ref_quat_xyzw_t = np.array([[0, 0, 0, 1]], dtype=np.float32)  # identity quat (xyzw)
        else:
            # Legacy format: can extract motion command from ONNX
            time_step = np.zeros((1, 1), dtype=np.float32)
            input_feed = {"obs": obs, "time_step": time_step}
            outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
            self.ref_quat_xyzw_t = outputs[2]
        
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            if self.use_new_onnx_format:
                # New format: only outputs actions
                output = self.onnx_policy_session.run(["action"], input_feed)
                action = output[0]
                # Slice back to actual batch size if we padded for fixed-batch ONNX
                if hasattr(self, '_onnx_batch_size') and action.shape[0] == self._onnx_batch_size:
                    action = action[:1]
                # Motion command stays the same (can't extract from new ONNX)
                motion_command = self.motion_command_t
                ref_quat_xyzw = self.ref_quat_xyzw_t
            else:
                # Legacy format: outputs actions + motion data
                output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
                action = output[0]
                motion_command = np.concatenate(output[1:3], axis=1)
                ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act
    
    def _load_motion_file(self):
        """Load motion file for future_motion_targets computation."""
        motion_file_path = self.config.task.motion_file_path
        if motion_file_path is None:
            logger.error(
                "ONNX model requires future_motion_targets input but task.motion_file_path is not configured. "
                "Please specify motion_file_path in the task configuration."
            )
            raise ValueError("Missing required config: task.motion_file_path")
        
        # Check if file exists
        motion_file = Path(motion_file_path)
        if not motion_file.exists():
            logger.error(f"Motion file not found: {motion_file_path}")
            raise FileNotFoundError(f"Motion file not found: {motion_file_path}")
        
        logger.info(f"Loading motion file for future_motion_targets: {motion_file_path}")
        
        # Load npz file
        with open(motion_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            self.motion_fps = float(data["fps"])
            
            # Get body and joint names
            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()
            
            # Load motion data (same format as MotionLoader in holosoma)
            # Joint data: first 7 are pelvis [xyz, wxyz] for pos, first 6 for vel, omit them
            joint_pos = data["joint_pos"][:, 7:]  # [time_steps, num_joints]
            joint_vel = data["joint_vel"][:, 6:]  # [time_steps, num_joints]
            
            # Body data
            body_pos_w = data["body_pos_w"]  # [time_steps, num_bodies, 3]
            body_quat_w_wxyz = data["body_quat_w"]  # [time_steps, num_bodies, 4] wxyz
            body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Convert to xyzw
            body_lin_vel_w = data["body_lin_vel_w"]  # [time_steps, num_bodies, 3]
            body_ang_vel_w = data["body_ang_vel_w"]  # [time_steps, num_bodies, 3]
            
            # Get indices for robot joints and bodies
            robot_joint_names = self.config.robot.dof_names
            robot_body_names_to_track = self.config.robot.motion.get("body_names_to_track")
            robot_body_name_ref = self.config.robot.motion.get("body_name_ref", ["torso_link"])
            
            if robot_body_names_to_track is None:
                logger.warning("body_names_to_track not found in robot.motion config - using all bodies")
                robot_body_names_to_track = body_names  # Use all bodies from motion file
            
            joint_indices = [joint_names.index(name) for name in robot_joint_names]
            body_indices = [body_names.index(name) for name in robot_body_names_to_track]
            ref_body_index = body_names.index(robot_body_name_ref[0])

            # Root body index: the pelvis (first body listed in the robot config).
            # In the motion NPZ, body index 0 is often "world" (all zeros) — the
            # actual pelvis sits at a different index.  Match the training code which
            # uses ``_body_indexes[0]`` (= first robot body mapped into the motion
            # file's body list).
            robot_body_name_root = self.config.robot.motion.get("body_name_root", ["pelvis"])
            root_body_index = body_names.index(robot_body_name_root[0])
            logger.info(f"Root body '{robot_body_name_root[0]}' at motion-file index {root_body_index}")

            # Store motion data
            self.motion_data = {
                "joint_pos": joint_pos[:, joint_indices],  # [time_steps, num_dofs]
                "joint_vel": joint_vel[:, joint_indices],  # [time_steps, num_dofs]
                "body_pos_w": body_pos_w,  # [time_steps, all_bodies, 3]
                "body_quat_w": body_quat_w,  # [time_steps, all_bodies, 4] xyzw
                "body_lin_vel_w": body_lin_vel_w,  # [time_steps, all_bodies, 3]
                "body_ang_vel_w": body_ang_vel_w,  # [time_steps, all_bodies, 3]
                "body_indices": body_indices,  # Indices of 14 tracked bodies
                "ref_body_index": ref_body_index,  # Index of reference body (torso_link)
                "root_body_index": root_body_index,  # Index of root body (pelvis)
                "time_step_total": len(joint_pos),
            }
            
            logger.info(f"Motion file loaded: {self.motion_data['time_step_total']} timesteps at {self.motion_fps} fps")

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0

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

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # motion_command and ref_quat - compute from motion data if available (for new ONNX format)
        if self.has_future_motion and self.motion_data is not None:
            # Compute motion_command from current timestep in motion data
            timestep = min(self.motion_timestep, self.motion_data["time_step_total"] - 1)
            joint_pos_target = self.motion_data["joint_pos"][timestep]  # [num_dofs]
            joint_vel_target = self.motion_data["joint_vel"][timestep]  # [num_dofs]
            self.motion_command_t = np.concatenate([
                joint_pos_target[np.newaxis, :],
                joint_vel_target[np.newaxis, :]
            ], axis=1).astype(np.float32)  # [1, 58]
            
            # Also update ref_quat from motion data (use ref_body_index, typically torso_link)
            ref_body_idx = self.motion_data["ref_body_index"]
            self.ref_quat_xyzw_t = self.motion_data["body_quat_w"][timestep, ref_body_idx:ref_body_idx+1, :].astype(np.float32)  # [1, 4] xyzw
        
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

        # future_motion_targets (if needed)
        if self.has_future_motion and self.motion_data is not None:
            current_obs_buffer_dict["future_motion_targets"] = self._compute_future_motion_targets()

        # Diagnostic logging for first few timesteps when motion is active
        if self.motion_clip_progressing and self.motion_timestep < 5:
            base_pos = robot_state_data[0, :3]
            base_quat_wxyz = robot_state_data[0, 3:7]
            base_lin_vel = robot_state_data[0, 7 + self.num_dofs : 7 + self.num_dofs + 3]
            base_ang_vel = current_obs_buffer_dict["base_ang_vel"][0]
            dof_pos_first5 = current_obs_buffer_dict["dof_pos"][0, :5]
            motion_cmd_first5 = current_obs_buffer_dict["motion_command"][0, :5]
            ref_ori = current_obs_buffer_dict["motion_ref_ori_b"][0, :6]
            logger.info(
                f"[OBS t={self.motion_timestep}] "
                f"pos={base_pos} "
                f"quat_wxyz={base_quat_wxyz} "
                f"lin_vel={base_lin_vel} "
                f"ang_vel={base_ang_vel} "
                f"dof_pos[:5]={dof_pos_first5} "
                f"motion_cmd[:5]={motion_cmd_first5} "
                f"ref_ori={ref_ori}"
            )

        return current_obs_buffer_dict
    
    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference (override to include future_motion_targets)."""
        group_outputs = self._prepare_group_observations(robot_state_data)
        if "actor_obs" not in group_outputs:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")
        
        result = {"actor_obs": group_outputs["actor_obs"].astype(np.float32, copy=False)}
        
        # Add future_motion_targets if available
        if "future_motion_targets" in group_outputs:
            result["future_motion_targets"] = group_outputs["future_motion_targets"].astype(np.float32, copy=False)
        
        return result
    
    def _compute_future_motion_targets(
        self,
        future_num_steps: int = 20,
        future_max_steps: int = 95,
        include_key_body_pos: bool = True,  # Set to True to match training config (1560 dim)
    ) -> np.ndarray:
        """Compute future motion targets for motion encoder.
        
        Returns concatenated future motion features (numpy array):
        - root_height: [future_num_steps, 1]
        - roll_pitch: [future_num_steps, 2]
        - base_lin_vel: [future_num_steps, 3]
        - base_yaw_vel: [future_num_steps, 1]
        - dof_pos: [future_num_steps, num_dofs]
        - local_key_body_pos: [future_num_steps, num_tracked_bodies * 3]  (optional)
        
        Returns:
            Array of shape [1, future_num_steps * feature_dim]
        """
        motion = self.motion_data
        current_timestep = self.motion_timestep
        
        # Generate future timestep indices (linearly spaced)
        tar_obs_steps = np.linspace(
            1,  # start
            future_max_steps,  # stop
            future_num_steps,  # num
        ).astype(np.int64)
        
        # Current timestep + future offsets, clamped to valid range
        future_timesteps = current_timestep + tar_obs_steps
        future_timesteps = np.clip(future_timesteps, 0, motion["time_step_total"] - 1).astype(np.int64)
        
        # Get future motion data
        # Root position and quaternion (pelvis — use stored root_body_index, NOT hardcoded 0
        # because body index 0 in the NPZ is often "world" with all-zero data)
        root_idx = motion["root_body_index"]
        root_pos = motion["body_pos_w"][future_timesteps, root_idx]  # [num_steps, 3]
        root_quat = motion["body_quat_w"][future_timesteps, root_idx]  # [num_steps, 4] xyzw
        
        # Root height
        root_height = root_pos[:, 2:3]  # [num_steps, 1]
        
        # Roll and pitch from quaternion (xyzw format)
        def get_euler_xyz_np(quat_xyzw):
            """Get euler angles (roll, pitch, yaw) from quaternion in xyzw format.
            
            Matches holosoma.utils.rotations.get_euler_xyz with w_last=True.
            Returns values in [0, 2*pi] range (same as Isaac Sim).
            """
            x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch = np.where(
                np.abs(sinp) >= 1,
                np.copysign(np.pi / 2, sinp),  # Use 90 degrees if out of range
                np.arcsin(sinp)
            )
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Apply modulo to match Isaac Sim behavior (convert to [0, 2*pi] range)
            two_pi = 2 * np.pi
            return roll % two_pi, pitch % two_pi, yaw % two_pi
        
        roll, pitch, yaw = get_euler_xyz_np(root_quat)
        roll_pitch = np.stack([roll, pitch], axis=-1)  # [num_steps, 2]
        
        # Base linear velocity
        base_lin_vel = motion["body_lin_vel_w"][future_timesteps, root_idx]  # [num_steps, 3]

        # Base angular velocity (yaw component only)
        base_ang_vel = motion["body_ang_vel_w"][future_timesteps, root_idx]  # [num_steps, 3]
        base_yaw_vel = base_ang_vel[:, 2:3]  # [num_steps, 1]
        
        # Joint positions (relative to default)
        dof_pos = motion["joint_pos"][future_timesteps]  # [num_steps, num_dofs]
        default_dof_pos = self.default_dof_angles  # [num_dofs]
        dof_pos_rel = dof_pos - default_dof_pos[np.newaxis, :]  # [num_steps, num_dofs]
        
        # Local key body positions (tracked bodies relative to root) — optional
        if include_key_body_pos:
            tracked_body_indexes = motion["body_indices"]  # [14]
            tracked_body_pos = motion["body_pos_w"][future_timesteps][:, tracked_body_indexes]  # [num_steps, 14, 3]
            local_body_pos = tracked_body_pos - root_pos[:, np.newaxis, :]  # [num_steps, 14, 3]
            
            # Rotate to body frame using inverse of root quaternion
            # Use the same formula as holosoma.utils.rotations.quat_rotate_inverse
            def quat_rotate_inverse_np(q_xyzw, v):
                """Rotate vector v by inverse of quaternion q (xyzw format).
                
                This matches the formula from holosoma.utils.rotations.quat_rotate_inverse:
                a = v * (2.0 * q_w**2 - 1.0)
                b = cross(q_vec, v) * q_w * 2.0
                c = q_vec * dot(q_vec, v) * 2.0
                return a - b + c
                """
                q_vec = q_xyzw[:, :3]  # [N, 3]
                q_w = q_xyzw[:, 3]     # [N]
                
                # a = v * (2.0 * q_w**2 - 1.0)
                a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
                
                # b = cross(q_vec, v) * q_w * 2.0
                b = np.cross(q_vec, v) * (q_w * 2.0)[:, np.newaxis]
                
                # c = q_vec * dot(q_vec, v) * 2.0
                # dot(q_vec, v) for each row
                dot_product = np.sum(q_vec * v, axis=-1)  # [N]
                c = q_vec * (dot_product * 2.0)[:, np.newaxis]
                
                return a - b + c
            
            # Apply rotation for each timestep and body
            local_body_pos_flat = local_body_pos.reshape(-1, 3)  # [num_steps * 14, 3]
            root_quat_repeated = np.repeat(root_quat, len(tracked_body_indexes), axis=0)  # [num_steps * 14, 4]
            local_body_pos_b_flat = quat_rotate_inverse_np(root_quat_repeated, local_body_pos_flat)
            local_key_body_pos = local_body_pos_b_flat.reshape(future_num_steps, -1)  # [num_steps, 14 * 3]
            
            # Concatenate all features
            future_obs = np.concatenate([
                root_height, roll_pitch, base_lin_vel, base_yaw_vel, dof_pos_rel, local_key_body_pos,
            ], axis=-1)
        else:
            # Without key body positions
            future_obs = np.concatenate([
                root_height, roll_pitch, base_lin_vel, base_yaw_vel, dof_pos_rel,
            ], axis=-1)
        
        # Flatten and add batch dimension: [1, num_steps * feature_dim]
        result = future_obs.reshape(1, -1).astype(np.float32)

        # Debug logging (only when save_debug=True, motion is progressing, first 200 timesteps, and not already logged)
        save_debug = getattr(self.config.task, 'save_debug', False)
        if save_debug:
            if not hasattr(self, '_debug_future_motion_log'):
                self._debug_future_motion_log = []
            if not hasattr(self, '_debug_logged_timesteps'):
                self._debug_logged_timesteps = set()
        
        # Only log when motion is actually progressing and we haven't logged this timestep yet
        should_log = (
            save_debug 
            and self.motion_clip_progressing 
            and current_timestep < 200 
            and current_timestep not in getattr(self, '_debug_logged_timesteps', set())
        )
        if should_log:
            log_data = {
                'timestep': int(current_timestep),
                'future_timesteps': future_timesteps.tolist(),
                'root_height': root_height.tolist(),
                'roll_pitch': roll_pitch.tolist(),
                'base_lin_vel': base_lin_vel.tolist(),
                'base_yaw_vel': base_yaw_vel.tolist(),
                'dof_pos_rel_first': dof_pos_rel[0].tolist(),
                'result_shape': result.shape,
                'result_mean': float(result.mean()),
                'result_std': float(result.std()),
                # Add motion_command and ref_quat logging
                'motion_command_first_10': self.motion_command_t[0, :10].tolist() if hasattr(self, 'motion_command_t') else None,
                'motion_command_vel_first_10': self.motion_command_t[0, 29:39].tolist() if hasattr(self, 'motion_command_t') and self.motion_command_t.shape[1] >= 39 else None,
                'ref_quat_xyzw': self.ref_quat_xyzw_t[0].tolist() if hasattr(self, 'ref_quat_xyzw_t') else None,
            }
            if include_key_body_pos:
                log_data['local_key_body_pos_first_body'] = local_key_body_pos[0, :3].tolist()
            self._debug_future_motion_log.append(log_data)
            self._debug_logged_timesteps.add(current_timestep)
        
        return result

    def policy_action(self):
        # Check if motion finished on the previous cycle (set by rl_inference).
        # Must be handled HERE before super().policy_action() checks use_policy_action,
        # because _handle_stop_policy sets use_policy_action=False.
        if getattr(self, '_motion_finished', False):
            self._motion_finished = False
            self.logger.info(
                colored(
                    f"Motion clip finished ({self.motion_data['time_step_total']} frames). "
                    "Switching to stiff hold.",
                    "green",
                )
            )
            self._handle_stop_policy()
        super().policy_action()

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)

        # Prepare input_feed based on ONNX format
        if self.use_new_onnx_format:
            # New format: actor_obs (and optionally future_motion_targets)
            input_feed = {"actor_obs": obs["actor_obs"]}
            if self.has_future_motion:
                if "future_motion_targets" in obs:
                    input_feed["future_motion_targets"] = obs["future_motion_targets"]
                else:
                    logger.error(
                        "ONNX model expects 'future_motion_targets' input but it's not available in observations. "
                        "Please configure observation to include future_motion_targets or use a different ONNX model."
                    )
                    raise ValueError("Missing required input: future_motion_targets")

            # Handle batch size mismatch: ONNX model may have fixed batch dim
            # (e.g. 8192 from training) but inference uses batch=1.
            # Pad inputs to match expected batch size, run, then slice result.
            expected_batch = self.onnx_policy_session.get_inputs()[0].shape[0]
            if isinstance(expected_batch, int) and expected_batch > 1:
                self._onnx_batch_size = expected_batch
                for key in input_feed:
                    actual = input_feed[key].shape[0]
                    if actual < expected_batch:
                        pad = np.zeros(
                            (expected_batch - actual,) + input_feed[key].shape[1:],
                            dtype=input_feed[key].dtype,
                        )
                        input_feed[key] = np.concatenate([input_feed[key], pad], axis=0)
        else:
            # Legacy format: obs and time_step
            input_feed = {"time_step": np.array([[self.motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}

        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # Debug logging for first few policy steps (including standing phase)
        if getattr(self.config.task, 'save_debug', False) and not hasattr(self, '_debug_standing_count'):
            self._debug_standing_count = 0
        if getattr(self.config.task, 'save_debug', False) and not self.motion_clip_progressing and self._debug_standing_count < 10:
            self._debug_standing_count += 1
            logger.info(f"[STANDING step {self._debug_standing_count}] actor_obs shape: {obs['actor_obs'].shape}, mean: {obs['actor_obs'].mean():.6f}, std: {obs['actor_obs'].std():.6f}")
            logger.info(f"[STANDING step {self._debug_standing_count}] actions[:5]: {self.last_policy_action[0, :5]}")
            logger.info(f"[STANDING step {self._debug_standing_count}] policy_action mean: {policy_action.mean():.6f}, std: {policy_action.std():.6f}, first 5: {policy_action[0, :5]}")
            logger.info(f"[STANDING step {self._debug_standing_count}] dof_pos[:5]: {robot_state_data[0, 7:12]}")
            logger.info(f"[STANDING step {self._debug_standing_count}] dof_vel[:5]: {robot_state_data[0, 7+self.num_dofs+6:7+self.num_dofs+11]}")
            logger.info(f"[STANDING step {self._debug_standing_count}] base_ang_vel: {robot_state_data[0, 7+self.num_dofs+3:7+self.num_dofs+6]}")

        # Debug logging for first few timesteps
        if getattr(self.config.task, 'save_debug', False) and self.motion_timestep < 5:
            logger.info(f"[Sim2Sim Timestep {self.motion_timestep}] actor_obs shape: {obs['actor_obs'].shape}, mean: {obs['actor_obs'].mean():.6f}, std: {obs['actor_obs'].std():.6f}")
            logger.info(f"[Sim2Sim Timestep {self.motion_timestep}] motion_command (first 10): {self.motion_command_t[0, :10]}")
            logger.info(f"[Sim2Sim Timestep {self.motion_timestep}] ref_quat_xyzw: {self.ref_quat_xyzw_t[0]}")
            logger.info(f"[Sim2Sim Timestep {self.motion_timestep}] policy_action mean: {policy_action.mean():.6f}, std: {policy_action.std():.6f}, first 10: {policy_action[0, :10]}")

        # Save full policy input/output log for every timestep (when save_debug and motion is active)
        if getattr(self.config.task, 'save_debug', False) and self.motion_clip_progressing:
            if not hasattr(self, '_debug_policy_io_log'):
                self._debug_policy_io_log = []
            entry = {
                'timestep': int(self.motion_timestep),
                # Robot raw state
                'robot_base_pos': robot_state_data[0, :3].tolist(),
                'robot_base_quat_wxyz': robot_state_data[0, 3:7].tolist(),
                'robot_dof_pos': robot_state_data[0, 7:7 + self.num_dofs].tolist(),
                'robot_base_lin_vel': robot_state_data[0, 7 + self.num_dofs:7 + self.num_dofs + 3].tolist(),
                'robot_base_ang_vel': robot_state_data[0, 7 + self.num_dofs + 3:7 + self.num_dofs + 6].tolist(),
                'robot_dof_vel': robot_state_data[0, 7 + self.num_dofs + 6:7 + self.num_dofs + 6 + self.num_dofs].tolist(),
                # Policy inputs
                'actor_obs': obs['actor_obs'][0].tolist(),
                'actor_obs_shape': list(obs['actor_obs'].shape),
                # Motion tracking state
                'motion_command': self.motion_command_t[0].tolist(),
                'ref_quat_xyzw': self.ref_quat_xyzw_t[0].tolist(),
                'robot_yaw_offset_deg': float(np.degrees(self.robot_yaw_offset)),
                'motion_yaw_offset_deg': float(np.degrees(self.motion_yaw_offset)) if hasattr(self, 'motion_yaw_offset') else None,
                # Policy output
                'policy_action': policy_action[0].tolist(),
                'policy_action_mean': float(policy_action.mean()),
                'policy_action_std': float(policy_action.std()),
            }
            if self.has_future_motion and 'future_motion_targets' in obs:
                entry['future_motion_targets'] = obs['future_motion_targets'][0].tolist()
                entry['future_motion_targets_shape'] = list(obs['future_motion_targets'].shape)
            self._debug_policy_io_log.append(entry)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scale

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1

            # Auto-stop when motion clip is finished.
            # Set a flag so the stop happens at the START of the next cycle
            # (calling _handle_stop_policy here would set use_policy_action=False
            # mid-cycle, causing q_target to be unassigned in policy_action).
            if self.motion_data is not None and self.motion_timestep >= self.motion_data["time_step_total"]:
                self._motion_finished = True

        return self.scaled_policy_action

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing.
        # motion_start_timestep is set here (main thread) on the first call
        # after the motion clip starts, to avoid cross-thread clock skew.
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # First call after motion start — anchor now on the main thread.
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = 0
            return
        if self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
            offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        # Save debug logs before stopping (only when save_debug=True)
        if getattr(self.config.task, 'save_debug', False):
            import json
            from pathlib import Path
            if hasattr(self, '_debug_future_motion_log') and len(self._debug_future_motion_log) > 0:
                debug_log_path = Path("debug_future_motion_sim2sim.json")
                with open(debug_log_path, 'w') as f:
                    json.dump(self._debug_future_motion_log, f, indent=2)
                self.logger.info(f"Debug log saved to: {debug_log_path}")
            if hasattr(self, '_debug_policy_io_log') and len(self._debug_policy_io_log) > 0:
                io_log_path = Path("debug_policy_io_sim2sim.json")
                with open(io_log_path, 'w') as f:
                    json.dump(self._debug_policy_io_log, f, indent=2)
                self.logger.info(f"Policy I/O log saved to: {io_log_path} ({len(self._debug_policy_io_log)} timesteps)")

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Do NOT read the clock here — this runs on the keyboard listener thread,
        # while _update_clock runs on the main policy thread.  Reading the clock
        # here creates a multi-second gap because the main thread won't call
        # _update_clock until its next loop iteration.  Instead, let the main
        # thread anchor the clock on its first _update_clock call.
        self.motion_start_timestep = None
        self.motion_timestep = 0
        self._last_clock_reading = None
        # Reset debug logging for new motion run
        if hasattr(self, '_debug_logged_timesteps'):
            self._debug_logged_timesteps.clear()
        if hasattr(self, '_debug_policy_io_log'):
            self._debug_policy_io_log = []
        self.logger.info(colored("Starting motion clip", "blue"))

    def handle_keyboard_button(self, keycode):
        """Add new keyboard button to start and end the motion clips"""
        if keycode == "s":
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to base class
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

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
