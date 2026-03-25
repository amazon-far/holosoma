"""Standalone success rate evaluation for PhUMA multi-motion checkpoints.

Loads a checkpoint, overrides the split file (e.g. to validation set),
sets up the environment with MultiMotionCommand, and runs the
SuccessRateCallback evaluation loop.

Must be run from the kyungminn_phuma branch which has MultiMotionCommand.

Usage:
    # IsaacSim (default, uses saved simulator from checkpoint):
    python src/holosoma/holosoma/eval_success_rate.py \
        --checkpoint /path/to/model_106000.pt \
        --motion_dir ./g1_npz \
        --split_file ./split/phuma_val.txt

    # Sim-to-sim with MuJoCo (Warp backend, GPU-accelerated):
    python src/holosoma/holosoma/eval_success_rate.py \
        --checkpoint /path/to/model_106000.pt \
        --motion_dir ./g1_npz \
        --split_file ./split/phuma_val.txt \
        --simulator mjwarp
"""

from __future__ import annotations

import argparse

from loguru import logger

from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


def main():
    init_eval_logging()

    parser = argparse.ArgumentParser(description="Evaluate success rate on a motion split")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--motion_dir", default=None, help="Override motion_dir")
    parser.add_argument("--split_file", default=None, help="Override split_file")
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="Number of parallel envs for evaluation (default: 4096)")
    parser.add_argument("--simulator", default=None, choices=["mujoco", "mjwarp"],
                        help="Override simulator for sim-to-sim eval (default: use saved simulator)")
    args = parser.parse_args()

    # Load saved config from checkpoint
    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # Override eval_overrides.num_envs before get_eval_config() since
    # the default is 1 (for single-motion eval), but SuccessRateCallback
    # needs many envs to batch through all motions efficiently.
    import dataclasses
    from holosoma.config_types.experiment import EvalOverridesConfig
    saved_cfg = dataclasses.replace(
        saved_cfg,
        eval_overrides=dataclasses.replace(
            saved_cfg.eval_overrides,
            num_envs=args.num_envs,
            headless=True,
        ),
    )

    # Override simulator if requested (for sim-to-sim eval with MuJoCo)
    if args.simulator is not None:
        from holosoma.config_values import simulator as sim_presets
        sim_cfg = sim_presets.DEFAULTS[args.simulator]
        saved_cfg = dataclasses.replace(saved_cfg, simulator=sim_cfg)
        # Skip IsaacSim-only randomizers (e.g. rigid body material) when using MuJoCo
        saved_cfg = dataclasses.replace(
            saved_cfg,
            randomization=dataclasses.replace(
                saved_cfg.randomization,
                ignore_unsupported=True,
            ),
        )
        logger.info(f"Overriding simulator -> {args.simulator} ({sim_cfg._target_})")

    # Get eval config (applies eval_overrides including our num_envs)
    eval_cfg = saved_cfg.get_eval_config()

    # Override motion_dir and split_file by mutating the params dict in-place.
    # The params dict inside the frozen dataclass is itself mutable.
    motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion_config, dict):
        if args.motion_dir is not None:
            motion_config["motion_dir"] = args.motion_dir
            logger.info(f"Overriding motion_dir -> {args.motion_dir}")
        if args.split_file is not None:
            motion_config["split_file"] = args.split_file
            logger.info(f"Overriding split_file -> {args.split_file}")
    else:
        # It's a dataclass object - need to replace it
        from dataclasses import asdict, fields
        mc_dict = asdict(motion_config)
        if args.motion_dir is not None:
            mc_dict["motion_dir"] = args.motion_dir
            logger.info(f"Overriding motion_dir -> {args.motion_dir}")
        if args.split_file is not None:
            mc_dict["split_file"] = args.split_file
            logger.info(f"Overriding split_file -> {args.split_file}")
        eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = mc_dict

    # Setup simulation environment
    env, device, simulation_app = setup_simulation_environment(eval_cfg)

    # Create log directory
    eval_log_dir = get_experiment_dir(
        eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval_success_rate"
    )
    eval_log_dir = eval_log_dir.resolve()
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval logs to {eval_log_dir}")

    # Load algo and checkpoint
    algo_class = get_class(eval_cfg.algo._target_)
    algo = algo_class(
        device=device,
        env=env,
        config=eval_cfg.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(args.checkpoint))
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Manually create the SuccessRateCallback and inject it, because the saved
    # config stores eval_callbacks as plain dicts (from YAML) which the
    # instantiate() helper doesn't handle (it expects attribute access, not dict keys).
    from holosoma.agents.callbacks.success_rate_callback import SuccessRateCallback
    callback = SuccessRateCallback(config=None, training_loop=algo)
    algo.eval_callbacks = [callback]

    # Run evaluation using PPO's built-in evaluate_policy
    logger.info("Starting success rate evaluation...")
    algo.evaluate_policy(max_eval_steps=None)

    # Print final results from callbacks
    for cb in algo.eval_callbacks:
        if hasattr(cb, "metrics"):
            logger.info("=" * 60)
            logger.info("=== Final Evaluation Results ===")
            logger.info("=" * 60)
            for key, value in sorted(cb.metrics.items()):
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
