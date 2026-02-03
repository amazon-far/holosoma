from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Tuple

import onnx
import torch

from holosoma.config_types.robot import RobotConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.module_utils import get_holosoma_root

def _find_input_dim_from_module(module: torch.nn.Module) -> int:
    """Finds the input dimension by examining a torch module's structure.

    Tries multiple strategies to find the first Linear layer's input features.
    """
    # Strategy 1: PPO-style - actor_module.module (torch.nn.Sequential)
    if hasattr(module, "actor_module") and hasattr(module.actor_module, "module"):
        core_model = module.actor_module.module
        if hasattr(core_model[0], "in_features"):
            return core_model[0].in_features

    # Strategy 2: FastSAC/FastTD3-style - .net attribute
    if hasattr(module, "net") and len(module.net) > 0:
        if hasattr(module.net[0], "in_features"):
            return module.net[0].in_features

    # Strategy 3: Find first Linear layer in module tree
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            return submodule.in_features

    raise ValueError(f"Cannot determine input dimension from module: {type(module)}")


def _extract_actor_model_and_input_dim(actor_wrapper) -> Tuple[torch.nn.Module, int]:
    """Extracts the underlying actor model and input dimension from various actor wrapper types.

    This function handles the complete actor pipeline including observation normalization.

    Parameters
    ----------
    actor_wrapper : object
        Actor wrapper containing the actor model. Can be various types including:
        - PPO actors: ActorWrapper -> PPOActor -> actor_module.module (torch.nn.Sequential)
        - FastSAC actors: ActorWrapper(with obs_normalizer) -> FastSAC Actor (custom module)
        - FastTD3 actors: ActorWrapper(with obs_normalizer) -> FastTD3 Actor (custom module)

    Returns
    -------
    complete_actor_model : torch.nn.Module
        The complete actor model including observation normalization if present.
    input_dimension : int
        The input dimension of the actor model.

    Notes
    -----
    The returned actor_model includes observation normalization if present,
    so it can be called directly with raw observations.
    """
    # If it's already a complete wrapper (from actor_onnx_wrapper), use it directly
    if hasattr(actor_wrapper, "forward") and hasattr(actor_wrapper, "actor"):
        # Use the complete wrapper that includes obs normalization
        complete_model = actor_wrapper
        # Find input dim from the inner actor
        input_dim = _find_input_dim_from_module(actor_wrapper.actor)
        return complete_model, input_dim

    # Otherwise, extract the inner actor and find its input dimension
    inner_actor = getattr(actor_wrapper, "actor", actor_wrapper)

    if not isinstance(inner_actor, torch.nn.Module):
        raise ValueError(
            f"Unsupported actor type: {type(inner_actor)}. Expected torch.nn.Module or wrapper with .actor attribute"
        )

    input_dim = _find_input_dim_from_module(inner_actor)

    # For unwrapped actors, we might need to return the inner model for certain types
    # PPO: Return the core Sequential model, others: return the actor itself
    if hasattr(inner_actor, "actor_module") and hasattr(inner_actor.actor_module, "module"):
        return inner_actor.actor_module.module, input_dim
    return inner_actor, input_dim


def export_policy_as_onnx(wrapper, onnx_file_path: str, example_obs_dict):
    # Ensure parent directory exists
    os.makedirs(Path(onnx_file_path).parent, exist_ok=True)
    example_input_list = example_obs_dict["actor_obs"]

    # --- SUPPRESS LOGS START ---
    # Silence onnxscript and onnx_ir debug/info noise
    import logging

    for logger_name in ["onnxscript", "onnx_ir", "torch.onnx"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # --- SUPPRESS LOGS END ---

    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        onnx_file_path,
        verbose=False,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,
        dynamo=False,
    )


def export_multi_agent_decouple_policy_as_onnx(wrapper, path, exported_policy_name, example_obs_dict, config):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    body_keys = config.robot.get("body_keys", ["lower_body", "upper_body"])
    actor_obs_keys = {}
    for body_key in body_keys:
        actor_obs_keys[body_key] = config.algo.config.module_dict[f"actor_{body_key}"].input_dim

    # Prepare example inputs
    example_input_list = []
    for body_key in body_keys:
        actor_obs = torch.cat([example_obs_dict[value] for value in actor_obs_keys[body_key]], dim=-1)
        example_input_list.append(actor_obs)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        example_input_list,
        path,
        verbose=False,
        input_names=[f"actor_obs_{body_key}" for body_key in body_keys],
        output_names=["action"],
        opset_version=13,
        dynamo=False,
    )


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, motion_command, actor, device):
        super().__init__()
        self.device = device
        # Extract the underlying actor model and input dimension generically
        actor_model, self.input_dim = _extract_actor_model_and_input_dim(actor)
        # Wrap the actor to handle different return signatures
        self._wrapped_actor = self._create_actor_wrapper(actor_model)

        motion = motion_command.motion

        joint_pos = motion.joint_pos
        joint_vel = motion.joint_vel

        body_pos_w = motion.body_pos_w
        body_quat_w = motion.body_quat_w
        ref_body_index = motion_command.ref_body_index
        ref_body_pos_w = body_pos_w[:, ref_body_index, :]
        ref_body_quat_w = body_quat_w[:, ref_body_index, :]  # in xyzw

        self.joint_pos = joint_pos.to("cpu")
        self.joint_vel = joint_vel.to("cpu")
        self.ref_body_pos_w = ref_body_pos_w.to("cpu")
        self.ref_body_quat_w = ref_body_quat_w.to("cpu")

        self.time_step_total = self.joint_pos.shape[0]

    def _create_actor_wrapper(self, actor_model):
        """Creates a wrapper that normalizes actor output to just return actions."""

        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, x):
                output = self.actor(x)
                # Handle different return signatures:
                # - PPO Sequential: returns tensor directly
                # - PPO ActorWrapper: returns tensor directly
                # - FastSAC/FastTD3: returns tuple (action, mean, log_std) or (action, ...)
                # - FastSAC/FastTD3 ActorWrapper: already returns action tensor
                if isinstance(output, tuple):
                    return output[0]  # Return first element (action)
                return output  # Return as-is for tensors

        return ActorWrapper(actor_model)

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self._wrapped_actor(x),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.ref_body_pos_w[time_step_clamped],
            self.ref_body_quat_w[time_step_clamped],
        )

    def export(self, onnx_file_path: str):
        onnx_file_dir = os.path.dirname(onnx_file_path)
        os.makedirs(onnx_file_dir, exist_ok=True)
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            onnx_file_path,
            export_params=True,
            opset_version=13,
            verbose=False,
            input_names=["obs", "time_step"],
            output_names=["actions", "joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"],
            dynamo=False,
        )
        self.to(self.device)

class _OnnxMultiMotionPolicyExporter(torch.nn.Module):
    def __init__(self, motion_command, actor, device, config):
        super().__init__()
        self.device = device
        self.config = config
        self.num_future_steps = config.num_future_steps
        # Extract the underlying actor model and input dimension generically
        actor_model, self.input_dim = _extract_actor_model_and_input_dim(actor)
        # Wrap the actor to handle different return signatures
        self._wrapped_actor = self._create_actor_wrapper(actor_model)

        motion = motion_command.motion

        joint_pos = motion.joint_pos
        joint_vel = motion.joint_vel

        body_pos_w = motion.body_pos_w
        body_quat_w = motion.body_quat_w
        ref_body_index = motion_command.ref_body_index
        ref_body_pos_w = body_pos_w[:, ref_body_index, :]
        ref_body_quat_w = body_quat_w[:, ref_body_index, :]  # in xyzw

        self.joint_pos = joint_pos.to("cpu")
        self.joint_vel = joint_vel.to("cpu")
        self.ref_body_pos_w = ref_body_pos_w.to("cpu")
        self.ref_body_quat_w = ref_body_quat_w.to("cpu")

        self.time_step_total = self.joint_pos.shape[0]

    def _create_actor_wrapper(self, actor_model):
        """Creates a wrapper that normalizes actor output to just return actions."""

        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, x):
                output = self.actor(x)
                # Handle different return signatures:
                # - PPO Sequential: returns tensor directly
                # - PPO ActorWrapper: returns tensor directly
                # - FastSAC/FastTD3: returns tuple (action, mean, log_std) or (action, ...)
                # - FastSAC/FastTD3 ActorWrapper: already returns action tensor
                if isinstance(output, tuple):
                    return output[0]  # Return first element (action)
                return output  # Return as-is for tensors

        return ActorWrapper(actor_model)

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)

        def _get_future_motion_obs(self, time_step_clamped) -> torch.Tensor:

            future_obs_list = []
            
            for step_offset in range(1, self.num_future_steps + 1):
                # Calculate future time steps (clamp to max motion length)
                future_time_steps = torch.clamp(
                    time_step_clamped + step_offset, 
                    max=self.time_step_total - 1
                )
                
                # Get future joint_pos and joint_vel
                future_joint_pos = self.joint_pos[future_time_steps]  # [num_envs, num_dofs]
                future_joint_vel = self.joint_vel[future_time_steps]  # [num_envs, num_dofs]
                
                # Concatenate pos and vel for this step
                future_step_obs = torch.cat([future_joint_pos, future_joint_vel], dim=1)  # [num_envs, num_dofs*2]
                future_obs_list.append(future_step_obs)
            
            # Concatenate all future steps
            future_motion_obs = torch.cat(future_obs_list, dim=1)  # [num_envs, num_future_steps * num_dofs * 2]

            return future_motion_obs.to(self.device)
        
        future_motion_obs = _get_future_motion_obs(self, time_step_clamped)

        return (
            self._wrapped_actor(x),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.ref_body_pos_w[time_step_clamped],
            self.ref_body_quat_w[time_step_clamped],
            future_motion_obs,
        )

    def export(self, onnx_file_path: str):
        onnx_file_dir = os.path.dirname(onnx_file_path)
        os.makedirs(onnx_file_dir, exist_ok=True)
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            onnx_file_path,
            export_params=True,
            opset_version=13,
            verbose=False,
            input_names=["obs", "time_step"],
            output_names=["actions", "joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw", "future_motion_obs"],
            dynamo=False,
        )
        self.to(self.device)

def export_motion_and_policy_as_onnx(
    actor: object,
    motion_command: object,
    onnx_file_path: str,
    device: str,
):
    policy_exporter = _OnnxMotionPolicyExporter(motion_command, actor, device)
    policy_exporter.export(onnx_file_path)

def export_multi_motion_and_policy_as_onnx(
    actor: object,
    motion_command: object,
    onnx_file_path: str,
    device: str,
    config: object | None = None,
):
    if config is None:
        # Fallback: create minimal config if not provided
        raise ValueError("config is required for multi-motion ONNX export")
    policy_exporter = _OnnxMultiMotionPolicyExporter(motion_command, actor, device, config)
    policy_exporter.export(onnx_file_path)


def attach_onnx_metadata(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Attach custom metadata to an ONNX model file.

    Loads the ONNX model, appends metadata key-value pairs (values are serialized as JSON),
    and saves the modified model back to the same file.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file to modify.
    metadata : dict[str, Any]
        Dictionary of metadata to attach. Values are JSON-serialized before storage.
    """
    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = json.dumps(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)


def get_control_gains_from_config(robot_config: RobotConfig) -> tuple[list[float], list[float]]:
    """Extract Kp & Kd gains from env.

    The order of returned lists is determined by `env.robot_config.dof_names`.
    """

    kp_list = []
    kd_list = []
    stiffness_dict = robot_config.control.stiffness
    damping_dict = robot_config.control.damping

    for dof_name in robot_config.dof_names:
        # Map each DOF to its corresponding kp/kd value using substring matching
        # e.g. `left_hip_pitch_joint` from `dof_names` to  `hip_pitch` in `robot_config.control.stiffness`
        matches = [p for p in stiffness_dict if p in dof_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly 1 pattern match for '{dof_name}', got {len(matches)}: {matches}")

        pattern = matches[0]
        kp_list.append(float(stiffness_dict[pattern]))
        kd_list.append(float(damping_dict[pattern]))

    return kp_list, kd_list


def get_command_ranges_from_env(env: BaseTask) -> dict | None:
    """Extract command limits from env command manager."""

    if env.command_manager is not None:
        locomotion_cmd = env.command_manager.get_state("locomotion_command")
        if locomotion_cmd is not None and hasattr(locomotion_cmd, "command_ranges"):
            return locomotion_cmd.command_ranges
    return None


def get_urdf_text_from_robot_config(robot_config: RobotConfig) -> tuple[str, str]:
    """Extract URDF text from the robot config.

    Returns
    -------
    tuple[str, str]
        (urdf_file_path, urdf_str) - Path to URDF file and its contents
    """
    asset_root = robot_config.asset.asset_root
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root())

    asset_file = robot_config.asset.urdf_file
    urdf_file_path = os.path.join(asset_root, asset_file)
    urdf_str = Path(urdf_file_path).read_text(encoding="utf-8")
    return urdf_file_path, urdf_str
