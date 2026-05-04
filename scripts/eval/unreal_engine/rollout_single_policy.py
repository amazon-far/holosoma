"""Run one deterministic single-policy rollout per motion.

Works for any holosoma algo whose actor follows the standard interface
(``algo.actor.act_inference({"actor_obs": ...})``); used for PPO checkpoints
that don't have a teacher to pair against. Mirrors
``rollout_paired_motions.py`` but only runs Phase A.

Each env_i is bound to motion_i + tile_i (deterministic), the env state is
forced to the motion's frame 0 with no init-pose noise, and the rollout
records per-step ``robot_root_states[:, :3] - env_origins`` so the saved
trajectory lives in the motion's tile-local frame (matching the .npz mesh).

Saves ``policy_<motion_basename>.npy`` per motion.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


def _override_motion_cfg(eval_cfg, motion_dir: str, n_motions: int):
    motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion_config, dict):
        motion_config["motion_dir"] = motion_dir
        motion_config["max_motions"] = n_motions
        motion_config["pool_size"] = n_motions
    else:
        new_motion_config = dataclasses.replace(
            motion_config, motion_dir=motion_dir, max_motions=n_motions, pool_size=n_motions
        )
        eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = new_motion_config

    terrain_term = eval_cfg.terrain.terrain_term
    new_terrain_term = dataclasses.replace(
        terrain_term, obj_dir=motion_dir, obj_max_tiles=n_motions
    )
    return dataclasses.replace(
        eval_cfg, terrain=dataclasses.replace(eval_cfg.terrain, terrain_term=new_terrain_term)
    )


def _setup_deterministic_batch(env, motion_cmd):
    n = env.num_envs
    device = env.simulator.dof_pos.device

    motion_cmd.motion_indices[:] = torch.arange(n, device=device)
    motion_cmd.time_steps[:] = 0

    lib = motion_cmd.motion_library
    mi = motion_cmd.motion_indices
    ts = motion_cmd.time_steps

    root_pos = lib.body_pos_w(mi, ts)[:, 0].clone()
    root_rot = lib.body_quat_w(mi, ts)[:, 0].clone()
    root_lin_vel = lib.body_lin_vel_w(mi, ts)[:, 0].clone()
    root_ang_vel = lib.body_ang_vel_w(mi, ts)[:, 0].clone()
    dof_pos = lib.joint_pos(mi, ts).clone()
    dof_vel = lib.joint_vel(mi, ts).clone()

    tile_grid = getattr(motion_cmd, "_tile_grid", None)
    if tile_grid is not None:
        row_idx = torch.zeros(n, dtype=torch.long, device=device)
        new_origins = tile_grid[row_idx, mi]
        env.simulator.scene.env_origins[:] = new_origins
        root_pos = root_pos + new_origins

    # Mirror render_per_motion.py: lift the spawn pose by
    # ``noise_to_initial_pose.init_root_offset`` (configured at training time
    # to push the robot e.g. 10 cm above contact). Without this, motion
    # frame 0 puts the ankle_roll_link body exactly on the contact plane,
    # and the foot collision mesh — which extends a few cm below ankle
    # body — penetrates the terrain visually for the first few rollout
    # frames before PD control settles.
    init_pose_cfg = getattr(motion_cmd, "init_pose_cfg", None)
    init_root_offset = getattr(init_pose_cfg, "init_root_offset", None) if init_pose_cfg else None
    if init_root_offset is not None:
        offset = torch.tensor(list(init_root_offset), device=device, dtype=root_pos.dtype)
        root_pos = root_pos + offset

    env.simulator.dof_pos[:] = dof_pos
    env.simulator.dof_vel[:] = dof_vel
    env.simulator.robot_root_states[:, :3] = root_pos
    env.simulator.robot_root_states[:, 3:7] = root_rot
    env.simulator.robot_root_states[:, 7:10] = root_lin_vel
    env.simulator.robot_root_states[:, 10:13] = root_ang_vel

    all_ids = torch.arange(n, device=device)
    env.simulator.set_actor_root_state_tensor_robots(all_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(all_ids, env.simulator.dof_state)

    env.episode_length_buf[:] = 0
    env.reset_buf[:] = 0
    env.time_out_buf[:] = 0


def _record(env, num_envs, max_steps, action_fn, motion_lengths, log_obs: bool = False):
    simulator = env.simulator
    obs_dict = env.reset_all()
    motion_cmd = env.command_manager.get_state("motion_command")
    _setup_deterministic_batch(env, motion_cmd)
    obs_dict = env.observation_manager.compute()
    device = simulator.dof_pos.device
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)

    root_pos_buf = np.zeros((max_steps, num_envs, 3), dtype=np.float32)
    root_quat_buf = np.zeros((max_steps, num_envs, 4), dtype=np.float32)
    dof_pos_buf = np.zeros((max_steps, num_envs, simulator.dof_pos.shape[1]), dtype=np.float32)
    per_env_done_step = np.full(num_envs, max_steps, dtype=np.int64)
    env_done_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs_log_lists: dict[str, list[np.ndarray]] | None = None
    if log_obs:
        obs_log_lists = {
            "actor_obs": [],
            "height_map_obs": [],
            "sim_dof_vel": [],
            "sim_pelvis_lin_vel_w": [],
            "sim_pelvis_ang_vel_w": [],
            "ref_pos_w": [],
            "robot_ref_pos_w": [],
            "time_step": [],
        }

    motion_lengths_t = torch.as_tensor(motion_lengths, device=device, dtype=torch.long)
    horizon = min(max_steps, int(motion_lengths_t.max().item()) + 60)

    for step in range(horizon):
        root_state = simulator.robot_root_states[:, :].detach().cpu().numpy()
        env_origins = simulator.scene.env_origins.detach().cpu().numpy()
        root_pos_buf[step] = root_state[:, 0:3] - env_origins
        root_quat_buf[step] = root_state[:, 3:7]
        dof_pos_buf[step] = simulator.dof_pos.detach().cpu().numpy()

        if obs_log_lists is not None:
            obs_log_lists["actor_obs"].append(
                obs_dict["actor_obs"].detach().cpu().numpy().copy()
            )
            if "height_map_obs" in obs_dict:
                obs_log_lists["height_map_obs"].append(
                    obs_dict["height_map_obs"].detach().cpu().numpy().copy()
                )
            obs_log_lists["sim_dof_vel"].append(
                simulator.dof_vel.detach().cpu().numpy().copy()
            )
            obs_log_lists["sim_pelvis_lin_vel_w"].append(
                simulator.robot_root_states[:, 7:10].detach().cpu().numpy().copy()
            )
            obs_log_lists["sim_pelvis_ang_vel_w"].append(
                simulator.robot_root_states[:, 10:13].detach().cpu().numpy().copy()
            )
            obs_log_lists["ref_pos_w"].append(
                motion_cmd.ref_pos_w.detach().cpu().numpy().copy()
            )
            obs_log_lists["robot_ref_pos_w"].append(
                motion_cmd.robot_ref_pos_w.detach().cpu().numpy().copy()
            )
            obs_log_lists["time_step"].append(
                motion_cmd.time_steps.detach().cpu().numpy().copy()
            )

        with torch.inference_mode():
            actions = action_fn(obs_dict)
        obs_dict, _, dones, _ = env.step({"actions": actions})
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(device)

        cur_ts = motion_cmd.time_steps
        motion_finished = cur_ts >= (motion_lengths_t - 1)
        dones_bool = dones.to(torch.bool).view(-1)
        finished = dones_bool | motion_finished
        newly_done = finished & ~env_done_mask
        if newly_done.any():
            for i in newly_done.nonzero(as_tuple=True)[0].tolist():
                per_env_done_step[i] = step + 1
        env_done_mask |= finished
        if env_done_mask.all():
            break

    actual_steps = step + 1
    out = {
        "root_pos": root_pos_buf[:actual_steps],
        "root_quat": root_quat_buf[:actual_steps],
        "dof_pos": dof_pos_buf[:actual_steps],
        "per_env_done_step": per_env_done_step,
    }
    if obs_log_lists is not None:
        out["obs_log"] = {k: np.stack(v, axis=0) for k, v in obs_log_lists.items() if v}
    return out


def main():
    init_eval_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override num_envs (default: number of motions discovered).")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_obs", action="store_true",
                        help="Save per-step actor_obs, sim velocities, and motion ref pos to "
                        "<out_dir>/obs_log_<motion>.npz for diagnostic comparison.")
    parser.add_argument("--device", default=None,
                        help='Sim+model device, e.g. "cuda:0" or "cpu". Default = auto '
                        '("cuda:0" if available else "cpu"). Use "cpu" to coexist with a '
                        'training run that is hogging the GPU — PhysX runs on CPU, slower '
                        'but doesn\'t need GPU memory.')
    args = parser.parse_args()

    motion_dir = str(Path(args.motion_dir).expanduser().resolve())
    # rglob/glob don't follow symlinks; use os.walk(followlinks=True) so the
    # ``filtered_motions/<symlink>/motion.npz`` layout works.
    import os as _os
    motion_files: list[Path] = []
    for root, _dirs, files in _os.walk(motion_dir, followlinks=True):
        for f in files:
            if f.endswith(".npz"):
                motion_files.append(Path(root) / f)
    motion_files.sort()
    if not motion_files:
        raise FileNotFoundError(f"No .npz files under {motion_dir}")
    n_motions = len(motion_files)
    n_envs = args.num_envs or n_motions
    logger.info(f"{n_motions} motions, num_envs={n_envs}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_cfg, saved_wandb_path = load_saved_experiment_config(
        CheckpointConfig(checkpoint=args.checkpoint)
    )
    saved_cfg = dataclasses.replace(
        saved_cfg,
        eval_overrides=dataclasses.replace(saved_cfg.eval_overrides, num_envs=n_envs, headless=True),
    )
    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg = _override_motion_cfg(eval_cfg, motion_dir, n_motions)

    env, device, simulation_app = setup_simulation_environment(eval_cfg, device=args.device)
    log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(),
                                 task_name="rollout_single_policy").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    algo_class = get_class(eval_cfg.algo._target_)
    algo = algo_class(device=device, env=env, config=eval_cfg.algo.config,
                      log_dir=str(log_dir), multi_gpu_cfg=None)
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(args.checkpoint)
    algo.actor.eval()

    actor_obs_keys = getattr(algo, "actor_obs_keys", ["actor_obs"])
    use_height_map = getattr(algo, "use_height_map", False)
    use_motion_encoder = getattr(algo, "use_motion_encoder", False)
    logger.info(f"actor_obs_keys={actor_obs_keys} use_height_map={use_height_map} "
                f"use_motion_encoder={use_motion_encoder}")

    def policy_action(obs_dict):
        actor_obs = torch.cat([obs_dict[k] for k in actor_obs_keys], dim=1)
        state = {"actor_obs": actor_obs}
        if use_height_map and "height_map_obs" in obs_dict:
            state["height_map_obs"] = obs_dict["height_map_obs"]
        if use_motion_encoder and "future_motion_targets" in obs_dict:
            state["future_motion_targets"] = obs_dict["future_motion_targets"]
        return algo.actor.act_inference(state)

    motion_cmd = env.command_manager.get_state("motion_command")
    motion_lengths = motion_cmd.motion_library.time_step_totals.detach().cpu().numpy()
    logger.info(f"Per-motion lengths: {motion_lengths.tolist()}")

    try:
        env.set_is_evaluating()
        try:
            traj = _record(env, n_motions, args.max_steps, policy_action, motion_lengths,
                           log_obs=args.log_obs)
        except Exception:
            import traceback
            logger.error("Rollout crashed:\n" + traceback.format_exc())
            raise

        for env_id, motion_path in enumerate(motion_files):
            motion_n = int(motion_lengths[env_id])
            # Use the parent directory name as the motion's identifier — many
            # videomimic motions share the basename motion_for_holosoma_with_mesh.npz
            # so the parent (e.g. "00_ok_<subdir>") is what disambiguates them.
            base = motion_path.parent.name if motion_path.name == "motion_for_holosoma_with_mesh.npz" else motion_path.stem
            end = min(motion_n, int(traj["per_env_done_step"][env_id]),
                      traj["root_pos"].shape[0])
            end = max(end, 1)
            np.save(out_dir / f"policy_{base}.npy", {
                "root_trans": traj["root_pos"][:end, env_id],
                "root_ori": traj["root_quat"][:end, env_id],
                "dof_pos": traj["dof_pos"][:end, env_id],
                "fps": 50,
            })
            logger.info(f"Saved policy_{base}.npy ({end} frames; motion_len={motion_n})")

            if "obs_log" in traj:
                obs_log = traj["obs_log"]
                np.savez(
                    out_dir / f"obs_log_{base}.npz",
                    **{k: v[:end, env_id] for k, v in obs_log.items()},
                )
                logger.info(f"Saved obs_log_{base}.npz")
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
