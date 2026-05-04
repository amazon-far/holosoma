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
    parser.add_argument("--max_motions", type=int, default=None,
                        help="Override max_motions / pool_size in the saved config so a "
                        "checkpoint trained with pool_size=8192 can be evaluated on a smaller "
                        "motion dir.")
    parser.add_argument("--obj_dir", default=None,
                        help="Override terrain.terrain_term.obj_dir. Required when evaluating "
                        "on a motion dir different from the training one (the multi-tile "
                        "terrain must reference the same set of tiles as the motion pool).")
    parser.add_argument("--simulator", default=None, choices=["mujoco", "mjwarp"],
                        help="Override simulator for sim-to-sim eval (default: use saved simulator)")
    parser.add_argument("--disable_bad_tracking", action="store_true",
                        help="Loosen the BadTracking termination threshold to ~100 m so it "
                        "effectively never fires during eval. Useful when evaluating on "
                        "synthetic motions where commanded ankle/wrist positions don't match a "
                        "natural walking gait — without this, BadTracking fires within a few "
                        "frames, the env resets, and the SR callback's deterministic motion "
                        "binding breaks (every env gets marked failed).")
    parser.add_argument("--obj_tile_gap", type=float, default=None,
                        help="Override terrain.terrain_term.obj_tile_gap (gap in metres "
                        "between adjacent tiles). Increase (e.g. to 5.0) to keep neighbour "
                        "tiles outside the heightmap obs range, so the policy doesn't react "
                        "to the next tile's stairs while it's still finishing the current one.")
    parser.add_argument("--init_root_offset_z", type=float, default=None,
                        help="Override noise_to_initial_pose.init_root_offset[2]. The saved "
                        "config typically has +0.1 m to lift the spawn pose 10 cm above contact "
                        "and avoid foot penetration during reset. Pass 0.0 to spawn flush with "
                        "the floor (foot collision mesh ~5cm below ankle body, so PhysX bounces "
                        "back up — different transient but no sustained penetration).")
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
    # Keep motion_config as its original dataclass type so downstream attribute
    # access (e.g. base_task's bind_terrain_tiles auto-sync that rewrites
    # pool_size to match max_motions) continues to work.
    motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion_config, dict):
        if args.motion_dir is not None:
            motion_config["motion_dir"] = args.motion_dir
            logger.info(f"Overriding motion_dir -> {args.motion_dir}")
        if args.split_file is not None:
            motion_config["split_file"] = args.split_file
            logger.info(f"Overriding split_file -> {args.split_file}")
        if args.max_motions is not None:
            motion_config["max_motions"] = args.max_motions
            motion_config["pool_size"] = args.max_motions
            logger.info(f"Overriding max_motions/pool_size -> {args.max_motions}")
        if args.init_root_offset_z is not None:
            ntp = motion_config["noise_to_initial_pose"]
            old = list(ntp["init_root_offset"]) if isinstance(ntp, dict) else list(ntp.init_root_offset)
            old[2] = args.init_root_offset_z
            if isinstance(ntp, dict):
                ntp["init_root_offset"] = old
            else:
                motion_config["noise_to_initial_pose"] = dataclasses.replace(ntp, init_root_offset=old)
            logger.info(f"Overriding init_root_offset -> {old}")
    else:
        updates = {}
        if args.motion_dir is not None:
            updates["motion_dir"] = args.motion_dir
            logger.info(f"Overriding motion_dir -> {args.motion_dir}")
        if args.split_file is not None:
            updates["split_file"] = args.split_file
            logger.info(f"Overriding split_file -> {args.split_file}")
        if args.max_motions is not None:
            updates["max_motions"] = args.max_motions
            updates["pool_size"] = args.max_motions
            logger.info(f"Overriding max_motions/pool_size -> {args.max_motions}")
        if args.init_root_offset_z is not None:
            old = list(motion_config.noise_to_initial_pose.init_root_offset)
            old[2] = args.init_root_offset_z
            new_ntp = dataclasses.replace(
                motion_config.noise_to_initial_pose, init_root_offset=old
            )
            updates["noise_to_initial_pose"] = new_ntp
            logger.info(f"Overriding init_root_offset -> {old}")
        if updates:
            new_motion_config = dataclasses.replace(motion_config, **updates)
            eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = new_motion_config

    # Multi-tile terrain must reference the same set of meshes as the motion
    # pool when bind_terrain_tiles=True; sync obj_dir / obj_max_tiles so the
    # check in MultiMotionCommand.setup passes.
    if args.obj_dir is not None or args.obj_tile_gap is not None:
        terrain_term = eval_cfg.terrain.terrain_term
        terrain_updates: dict = {}
        if args.obj_dir is not None:
            terrain_updates["obj_dir"] = args.obj_dir
        if args.max_motions is not None:
            terrain_updates["obj_max_tiles"] = args.max_motions
        if args.obj_tile_gap is not None:
            terrain_updates["obj_tile_gap"] = args.obj_tile_gap
        new_terrain_term = dataclasses.replace(terrain_term, **terrain_updates)
        eval_cfg = dataclasses.replace(
            eval_cfg, terrain=dataclasses.replace(eval_cfg.terrain, terrain_term=new_terrain_term)
        )
        msg_parts = []
        if args.obj_dir is not None:
            msg_parts.append(f"obj_dir={args.obj_dir}")
        if args.max_motions is not None:
            msg_parts.append(f"obj_max_tiles={args.max_motions}")
        if args.obj_tile_gap is not None:
            msg_parts.append(f"obj_tile_gap={args.obj_tile_gap}")
        logger.info("Overriding terrain.terrain_term: " + ", ".join(msg_parts))

    # Optionally loosen BadTracking termination so it effectively never fires.
    if args.disable_bad_tracking:
        term_terms = eval_cfg.termination.terms
        if "bad_tracking" in term_terms:
            bt = term_terms["bad_tracking"]
            new_params = dict(bt.params)
            new_params["bad_motion_body_pos_threshold"] = 100.0
            new_params["bad_object_pos_threshold"] = 100.0
            new_params["bad_object_ori_threshold"] = 100.0
            term_terms["bad_tracking"] = dataclasses.replace(bt, params=new_params)
            logger.info("Loosened BadTracking thresholds to 100 m / 100 rad (effectively disabled).")

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
