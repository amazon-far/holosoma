from __future__ import annotations

import os

import numpy as np
import torch
import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)
    
    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir = eval_log_dir.resolve()  # absolute path so writes go here even if cwd changes (e.g. Isaac Sim)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    
    # Enable debug logging for future_motion_targets only when save_debug=True.
    # Use the env that has observation_manager (unwrap if wrapped) so the observation term sees the same object.
    save_debug = getattr(tyro_config.training, 'save_debug', False)
    debug_env = getattr(env, 'unwrapped_env', env)
    if not hasattr(debug_env, 'observation_manager'):
        debug_env = env
    if save_debug:
        debug_env._debug_future_motion_log = []
        debug_env._debug_logged_timesteps = set()
        debug_env._debug_log_path = str(eval_log_dir / "debug_future_motion_isaac.json")
        debug_env._debug_log_saved = False
    else:
        debug_env._debug_future_motion_log = None
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    # Add debug logging wrapper if save_debug is enabled
    if save_debug:
        original_rollout_step = algo._rollout_step
        step_count = [0]
        
        def logged_rollout_step(obs_dict):
            if step_count[0] < 5:
                actor_obs = obs_dict.get("actor_obs")
                if actor_obs is not None:
                    logger.info(f"[Isaac Timestep {step_count[0]}] actor_obs shape: {actor_obs.shape}, mean: {actor_obs.mean().item():.6f}, std: {actor_obs.std().item():.6f}")
                step_count[0] += 1
            return original_rollout_step(obs_dict)
        
        algo._rollout_step = logged_rollout_step

    # Custom evaluation loop to collect motion data and run for 1 loop only
    logger.info("Starting evaluation with motion data collection (1 loop only)...")
    
    # Set environment to evaluation mode
    if hasattr(env, 'set_is_evaluating'):
        env.set_is_evaluating()
    
    # Reset environment
    obs_dict = env.reset_all()
    for key in obs_dict:
        obs_dict[key] = obs_dict[key].to(algo.device)
    
    # Get inference policy from algo
    eval_policy = algo.get_inference_policy(device=str(algo.device))
    
    # Initialize data collection lists
    root_trans_list = []
    root_ori_list = []
    dof_pos_list = []
    
    step = 0
    prev_timestep = 0
    motion_completed_once = False
    env_id = 0  # Assuming single environment
    
    # Get simulator reference
    simulator = env.simulator
    
    # Get fps from motion command if available
    fps = 50  # Default fps
    if hasattr(env, 'command_manager'):
        motion_cmd = env.command_manager.get_state("motion_command")
        if motion_cmd is not None and hasattr(motion_cmd, 'motion'):
            # Try to get fps from motion data
            if hasattr(motion_cmd.motion, 'fps'):
                fps = motion_cmd.motion.fps
            elif hasattr(motion_cmd.motion, 'dt'):
                fps = 1.0 / motion_cmd.motion.dt
    
    logger.info(f"Collecting motion data at {fps} fps...")
    
    # Collect initial frame (after reset)
    root_state = simulator.robot_root_states[env_id].detach().cpu().numpy()  # [13]
    root_pos = root_state[:3]  # [x, y, z]
    root_quat = root_state[3:7]  # [qx, qy, qz, qw] - holosoma format
    # Convert to [x, y, z, w] format for visualize.py
    root_ori_xyzw = np.array([root_quat[0], root_quat[1], root_quat[2], root_quat[3]])  # [x, y, z, w]
    dof_pos = simulator.dof_pos[env_id].detach().cpu().numpy()  # [num_dofs]
    
    root_trans_list.append(root_pos)
    root_ori_list.append(root_ori_xyzw)
    dof_pos_list.append(dof_pos)
    
    while True:
        with torch.inference_mode():
            # Get actions from policy
            if hasattr(algo, 'actor_obs_keys'):
                actor_obs = torch.cat([obs_dict[k] for k in algo.actor_obs_keys], dim=1)
                policy_state_dict = {"actor_obs": actor_obs}
                # Add future_motion_targets if using motion encoder
                if hasattr(algo, 'use_motion_encoder') and algo.use_motion_encoder and "future_motion_targets" in obs_dict:
                    policy_state_dict["future_motion_targets"] = obs_dict["future_motion_targets"]
                actions = eval_policy(policy_state_dict)
            else:
                actions = eval_policy(obs_dict)
        
        # Collect motion data BEFORE step (to capture current state)
        root_state = simulator.robot_root_states[env_id].detach().cpu().numpy()  # [13]
        root_pos = root_state[:3]  # [x, y, z]
        root_quat = root_state[3:7]  # [qx, qy, qz, qw] - holosoma format
        # Convert to [x, y, z, w] format for visualize.py
        root_ori_xyzw = np.array([root_quat[0], root_quat[1], root_quat[2], root_quat[3]])  # [x, y, z, w]
        dof_pos = simulator.dof_pos[env_id].detach().cpu().numpy()  # [num_dofs]
        
        root_trans_list.append(root_pos)
        root_ori_list.append(root_ori_xyzw)
        dof_pos_list.append(dof_pos)
        
        # Step the environment
        actor_state_for_env = {"actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state_for_env)
        
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].to(algo.device)
        
        step += 1
        
        if step % 100 == 0:
            # Log motion progress
            if hasattr(env, 'command_manager'):
                motion_cmd = env.command_manager.get_state("motion_command")
                if motion_cmd is not None and hasattr(motion_cmd, 'time_steps') and hasattr(motion_cmd.motion, 'time_step_total'):
                    current_timestep = motion_cmd.time_steps[0].item()
                    total_timesteps = motion_cmd.motion.time_step_total
                    logger.info(f"Step {step} - Motion progress: {current_timestep}/{total_timesteps}")
                else:
                    logger.info(f"Step {step}")
            else:
                logger.info(f"Step {step}")
        
        # Check if motion ended by detecting timestep reset (decrease)
        motion_ended = False
        if hasattr(env, 'command_manager'):
            motion_cmd = env.command_manager.get_state("motion_command")
            if motion_cmd is not None and hasattr(motion_cmd, 'time_steps') and hasattr(motion_cmd.motion, 'time_step_total'):
                current_timestep = motion_cmd.time_steps[0].item()
                total_timesteps = motion_cmd.motion.time_step_total
                
                # Check if motion reached near the end
                if current_timestep >= total_timesteps - 5:
                    motion_completed_once = True
                
                # Detect reset: timestep decreased significantly (motion restarted)
                if motion_completed_once and current_timestep < prev_timestep - 10:
                    motion_ended = True
                    logger.info(f"Motion completed and reset detected at step {step} (timestep went from {prev_timestep} to {current_timestep})")
                
                prev_timestep = current_timestep
        
        # Also check time_out_buf as fallback
        if hasattr(env, 'time_out_buf') and env.time_out_buf.any():
            motion_ended = True
            logger.info(f"time_out_buf triggered at step {step}")

        if motion_ended:
            logger.info(f"Motion ended at step {step}")
            break
        
        # Safety limit (prevent infinite loops)
        if step >= 100000:
            logger.warning(f"Reached safety limit of 100000 steps, stopping")
            break
    
    logger.info(f"Evaluation completed. Total steps: {step}, Collected frames: {len(root_trans_list)}")
    
    # Save motion data as npy file
    if len(root_trans_list) > 0:
        root_trans_array = np.array(root_trans_list)  # [num_frames, 3]
        root_ori_array = np.array(root_ori_list)  # [num_frames, 4]
        dof_pos_array = np.array(dof_pos_list)  # [num_frames, num_dofs]
        
        # Create output directory if needed
        output_dir = os.path.join(checkpoint_dir, "motion_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as npy file in visualize.py format
        output_file = os.path.join(output_dir, "ref_motion.npy")
        motion_data = {
            'root_trans': root_trans_array,
            'root_ori': root_ori_array,
            'dof_pos': dof_pos_array,
            'fps': fps
        }
        np.save(output_file, motion_data)
        logger.info(f"Saved motion data to {output_file}")
        logger.info(f"  - Frames: {len(root_trans_list)}")
        logger.info(f"  - FPS: {fps}")
        logger.info(f"  - Root trans shape: {root_trans_array.shape}")
        logger.info(f"  - Root ori shape: {root_ori_array.shape}")
        logger.info(f"  - DOF pos shape: {dof_pos_array.shape}")

    # Save debug log (only when save_debug=True and not already saved); use same debug_env as set above
    if save_debug and hasattr(debug_env, '_debug_future_motion_log') and debug_env._debug_future_motion_log is not None:
        if not getattr(debug_env, '_debug_log_saved', False):
            import json
            debug_log_path = eval_log_dir / "debug_future_motion_isaac.json"
            log_list = debug_env._debug_future_motion_log
            with open(debug_log_path, 'w') as f:
                json.dump(log_list, f, indent=2)
            if len(log_list) > 0:
                logger.info(f"Debug log saved to: {debug_log_path} ({len(log_list)} entries)")
            else:
                logger.warning(f"Debug log saved to: {debug_log_path} (empty - future_motion_targets may not have been computed during eval)")
        else:
            logger.info(f"Debug log was already auto-saved at timestep 200")

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
