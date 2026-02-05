from __future__ import annotations

import os

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

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

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
