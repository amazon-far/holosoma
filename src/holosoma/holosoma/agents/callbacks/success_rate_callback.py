"""Success rate evaluation callback for multi-motion (PhUMA) training.

Evaluates policy by running all motions from initial frame and measuring
how many the robot can track without the reference body (root / pelvis)
deviating more than a threshold from the commanded root trajectory.

Definition:
  - Each motion starts at timestep 0
  - At each step, check if root (pelvis) 3D position error > threshold
  - If so, mark that motion as failed
  - success_rate = 1 - mean(terminate_states)

Also reports per-skill and per-category success rates when a CSV metadata file
is available (e.g. split/phuma_train.csv).
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import torch
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.managers.command.terms.wbt import MotionLoader

# Root-only thresholds: pelvis 3D position error exceeds this = fail.
MOTION_FAR_THRESHOLDS = [0.5, 0.25]


class SuccessRateCallback(RLEvalCallback):
    """Evaluates success rate across all motions in the library.

    Processes all motions in sequential batches of num_envs. Each env is assigned
    a unique motion from the pool, starts at timestep 0, and runs until
    the motion ends or any body deviates > MOTION_FAR_THRESHOLD from reference.
    """

    def __init__(self, config=None, training_loop=None):
        super().__init__(config, training_loop)
        self._num_envs = training_loop.env.num_envs

    def _load_metadata(self):
        """Load per-motion skill/category from CSV if available.

        Looks for a CSV file next to the split_file (same directory, same stem
        but .csv extension). Falls back to checking common locations.
        """
        self._motion_skill = {}   # basename -> skill
        self._motion_category = {}  # basename -> category

        mc = self._motion_command
        split_file = getattr(mc.motion_cfg, "split_file", "")
        if not split_file:
            return

        # Try .csv with same stem as split file
        csv_path = os.path.splitext(split_file)[0] + ".csv"
        if not os.path.isfile(csv_path):
            logger.info(f"SuccessRateCallback: no CSV metadata at {csv_path}")
            return

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chunk = row.get("chunk_file", "").strip()
                skill = row.get("skill", "").strip()
                category = row.get("category", "").strip()
                if chunk:
                    self._motion_skill[chunk] = skill
                    self._motion_category[chunk] = category

        logger.info(
            f"SuccessRateCallback: loaded metadata for {len(self._motion_skill)} motions "
            f"({len(set(self._motion_skill.values()))} skills, "
            f"{len(set(self._motion_category.values()))} categories)"
        )

    def _get_motion_basename(self, npz_path: str) -> str:
        """Extract basename without extension from npz path (e.g. 'Ways_to_Stand_Winded_clip1_chunk_0000')."""
        return os.path.splitext(os.path.basename(npz_path))[0]

    def on_pre_evaluate_policy(self):
        """Set up sequential motion assignment and tracking buffers."""
        env = self.training_loop.env
        mc = env.command_manager.get_state("motion_command")

        if not hasattr(mc, "motion_library"):
            logger.warning("SuccessRateCallback: no motion_library found, skipping eval")
            self._skip = True
            return

        self._skip = False
        self._motion_library = mc.motion_library
        self._motion_command = mc
        self._env = env

        self._num_motions = len(mc.motion_library._all_npz_files)
        self._num_batches = (self._num_motions + self._num_envs - 1) // self._num_envs

        # Track per-motion results for each threshold
        self._terminate_states = {
            thresh: torch.zeros(self._num_motions, dtype=torch.bool, device=self.device)
            for thresh in MOTION_FAR_THRESHOLDS
        }
        self._current_batch = 0
        self._env_done = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        # Save the original pool size and max_T for restoration
        self._orig_pool_size = self._motion_library.pool_size

        # Pool only needs to hold the number of motions actually evaluated per batch,
        # which is min(num_envs, num_motions). Growing to full num_envs wastes GPU memory
        # (e.g. with 4096 envs, 200 motions, padding pool to 4096 allocated ~3 GB extra and
        # OOM'd during eval right after training filled most of GPU memory).
        target_pool = min(self._num_envs, self._num_motions)
        if self._motion_library.pool_size < target_pool:
            self._motion_library._resize_pool(target_pool)

        # Load CSV metadata for per-skill/category reporting
        self._load_metadata()

        # Defer batch setup to on_pre_eval_env_step so it happens AFTER
        # evaluate_policy()'s reset_all() which would otherwise destroy
        # our deterministic motion assignment and robot state initialization.
        self._needs_initial_setup = True

        logger.info(
            f"SuccessRateCallback: evaluating {self._num_motions} motions "
            f"in {self._num_batches} batches of {self._num_envs} "
            f"(thresholds={MOTION_FAR_THRESHOLDS}m)"
        )

    def _setup_batch(self, batch_idx: int):
        """Load a batch of motions, assign to envs, and set robot physical states."""
        start = batch_idx * self._num_envs
        end = min(start + self._num_envs, self._num_motions)
        batch_size = end - start

        mc = self._motion_command
        lib = self._motion_library

        # Load exactly these motions into pool slots [0, batch_size)
        file_indices = list(range(start, end))
        loaders = []
        for fi in file_indices:
            loader = MotionLoader(
                lib._all_npz_files[fi],
                lib._robot_body_names,
                lib._robot_joint_names,
                device="cpu",
            )
            loaders.append(loader)

        # Check if we need to grow tensors
        if loaders:
            new_max_T = max(l.time_step_total for l in loaders)
            if new_max_T > lib._max_T:
                lib._grow_tensors(new_max_T)

        # Fill pool slots [0, batch_size) with these specific motions
        slot_indices = list(range(batch_size))
        lib._fill_slots(slot_indices, loaders)

        # Deterministic assignment: env[i] tracks pool slot i
        mc.motion_indices[:] = 0
        mc.motion_indices[:batch_size] = torch.arange(batch_size, device=self.device)
        mc.time_steps[:] = 0

        # Remember expected motion lengths per env (before env resets can corrupt them)
        self._expected_totals = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self._expected_totals[:batch_size] = lib.time_step_totals[mc.motion_indices[:batch_size]]

        # Track the previous time_steps to detect env resets
        self._prev_time_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)

        # Set robot physical states from motion data at timestep 0
        env = self._env
        mi = mc.motion_indices
        ts = mc.time_steps

        root_pos = lib.body_pos_w(mi, ts)[:, 0].clone()
        root_rot = lib.body_quat_w(mi, ts)[:, 0].clone()
        root_lin_vel = lib.body_lin_vel_w(mi, ts)[:, 0].clone()
        root_ang_vel = lib.body_ang_vel_w(mi, ts)[:, 0].clone()
        dof_pos = lib.joint_pos(mi, ts).clone()
        dof_vel = lib.joint_vel(mi, ts).clone()

        # When motions are bound to terrain tiles, env_origins must be re-bound
        # to match the new motion_indices (otherwise leftover env_origins from
        # the prior random reset point at the wrong tile, and adding them below
        # would put the robot on the wrong staircase).
        # Body pose data is stored in MOTION-LOCAL frame; the simulator expects
        # WORLD-frame poses, so we add env_origins after rebinding.
        # (Mirrors MultiMotionCommand.reset bind_terrain_tiles handling.)
        tile_grid = getattr(mc, "_tile_grid", None)
        if tile_grid is not None:
            n_rows, _, _ = tile_grid.shape
            # Use row 0 deterministically (eval should be reproducible).
            row_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
            col_idx = mc.motion_indices  # (num_envs,) in [0, n_cols)
            new_origins = tile_grid[row_idx, col_idx]  # (num_envs, 3)
            env.simulator.scene.env_origins[:] = new_origins
            root_pos = root_pos + new_origins

        # Apply the training-time init_root_offset (e.g. +10 cm Z lift) so
        # the foot collision mesh doesn't penetrate the contact plane at
        # spawn — otherwise PhysX bounces the robot, deviating ankles /
        # wrists past the BadTracking termination threshold within the
        # first physics step, the env resets to a random motion, and the
        # SR check sees a mismatched state and marks every env as failed.
        # Mirrors render_per_motion.py / rollout_single_policy.py.
        init_pose_cfg = getattr(mc, "init_pose_cfg", None)
        init_root_offset = (
            getattr(init_pose_cfg, "init_root_offset", None) if init_pose_cfg else None
        )
        if init_root_offset is not None:
            offset = torch.tensor(
                list(init_root_offset), device=self.device, dtype=root_pos.dtype
            )
            root_pos = root_pos + offset

        env.simulator.dof_pos[:] = dof_pos
        env.simulator.dof_vel[:] = dof_vel
        env.simulator.robot_root_states[:, :3] = root_pos
        env.simulator.robot_root_states[:, 3:7] = root_rot
        env.simulator.robot_root_states[:, 7:10] = root_lin_vel
        env.simulator.robot_root_states[:, 10:13] = root_ang_vel

        all_env_ids = torch.arange(self._num_envs, device=self.device)
        env.simulator.set_actor_root_state_tensor_robots(all_env_ids, env.simulator.robot_root_states)
        env.simulator.set_dof_state_tensor_robots(all_env_ids, env.simulator.dof_state)

        # Reset env done tracking
        self._env_done[:] = False
        if batch_size < self._num_envs:
            self._env_done[batch_size:] = True

        self._batch_size = batch_size

        # Reset episode counters
        env.episode_length_buf[:] = 0
        env.reset_buf[:] = 0
        env.time_out_buf[:] = 0

    def _check_motion_far(self) -> dict[float, torch.Tensor]:
        """Root-only check: 3D position error of the reference body (pelvis)
        between motion target and the robot exceeds threshold.

        This replaces the ASAP-style ``max over all tracked bodies`` rule.
        Tracking failures on hands / arms / head — common when the policy is
        only meant to follow the locomotion goal — no longer count as full
        motion failures; the success criterion is purely "did the root stay
        within ``threshold`` of the commanded root trajectory?".

        Returns dict of threshold -> (num_envs,) bool tensor.
        """
        mc = self._motion_command
        error = torch.norm(mc.ref_pos_w - mc.robot_ref_pos_w, dim=-1)  # (num_envs,)
        return {thresh: error > thresh for thresh in MOTION_FAR_THRESHOLDS}

    def on_pre_eval_env_step(self, actor_state):
        if self._skip:
            actor_state["stop"] = True
            return actor_state

        # Deferred setup: runs after evaluate_policy()'s reset_all() so our
        # deterministic motion assignment and robot states are not overwritten.
        if getattr(self, "_needs_initial_setup", False):
            self._needs_initial_setup = False
            self._setup_batch(0)

        return actor_state

    def on_post_eval_env_step(self, actor_state):
        if self._skip:
            actor_state["stop"] = True
            return actor_state

        mc = self._motion_command

        # Detect env resets: if time_steps went backwards (or to 0 when prev > 0),
        # the env was reset by the simulator (motion_ends / timeout).
        # Use our saved _expected_totals to determine if this was a natural completion.
        env_was_reset = (mc.time_steps < self._prev_time_steps) & ~self._env_done
        self._prev_time_steps = mc.time_steps.clone()

        # Check if motion naturally completed: prev_time_steps reached near the end
        # (env reset happens when time_steps >= totals - 2, then time_steps gets reset to 0)
        # At this point, the env has already been reset, so we can't check motion_far reliably.
        # Mark reset envs based on whether they had already failed before.
        # If an env was reset and not yet marked as done, it completed its motion successfully
        # (because if it had failed via motion_far, it would already be in _env_done).
        # So: env_was_reset & ~_env_done => completed successfully, just mark as done.
        self._env_done |= env_was_reset

        # For non-reset, non-done envs: check motion_far at all thresholds
        active = ~self._env_done
        if active[:self._batch_size].any():
            motion_far_dict = self._check_motion_far()
            batch_start = self._current_batch * self._num_envs

            for thresh, motion_far in motion_far_dict.items():
                newly_failed = motion_far & active
                for i in range(self._batch_size):
                    if newly_failed[i]:
                        motion_idx = batch_start + i
                        if motion_idx < self._num_motions:
                            self._terminate_states[thresh][motion_idx] = True

            # Use the largest (most lenient) threshold for env_done
            self._env_done |= motion_far_dict[max(MOTION_FAR_THRESHOLDS)]

        # Check if all envs in this batch are done
        if self._env_done[:self._batch_size].all():
            self._current_batch += 1
            if self._current_batch < self._num_batches:
                self._setup_batch(self._current_batch)
            else:
                actor_state["stop"] = True

        return actor_state

    def on_post_evaluate_policy(self):
        if self._skip:
            return

        num_total = self._num_motions
        self.metrics = {"Eval/num_motions": float(num_total)}

        for thresh in MOTION_FAR_THRESHOLDS:
            terminate_np = self._terminate_states[thresh].cpu().numpy()
            success_rate = 1.0 - terminate_np[:num_total].mean()
            num_success = int((~self._terminate_states[thresh][:num_total]).sum().item())
            thresh_key = f"{thresh}m"

            logger.info(
                f"SuccessRateCallback [{thresh}m]: success_rate={success_rate:.4f} "
                f"({num_success}/{num_total})"
            )

            self.metrics[f"Eval/success_rate_{thresh_key}"] = success_rate
            self.metrics[f"Eval/num_success_{thresh_key}"] = float(num_success)

            # Per-skill and per-category success rates
            if self._motion_skill:
                skill_results = defaultdict(list)
                category_results = defaultdict(list)

                lib = self._motion_library
                for i in range(num_total):
                    basename = self._get_motion_basename(lib._all_npz_files[i])
                    failed = bool(terminate_np[i])

                    skill = self._motion_skill.get(basename)
                    if skill:
                        skill_results[skill].append(failed)

                    category = self._motion_category.get(basename)
                    if category:
                        category_results[category].append(failed)

                logger.info(f"  Per-skill success rates [{thresh}m]:")
                for skill in sorted(skill_results.keys()):
                    fails = skill_results[skill]
                    sr = 1.0 - (sum(fails) / len(fails))
                    n = len(fails)
                    self.metrics[f"Eval/skill_{thresh_key}/{skill}"] = sr
                    logger.info(f"    {skill}: {sr:.4f} ({n - sum(fails)}/{n})")

                logger.info(f"  Per-category success rates [{thresh}m]:")
                for cat in sorted(category_results.keys()):
                    fails = category_results[cat]
                    sr = 1.0 - (sum(fails) / len(fails))
                    n = len(fails)
                    self.metrics[f"Eval/category_{thresh_key}/{cat}"] = sr
                    logger.info(f"    {cat}: {sr:.4f} ({n - sum(fails)}/{n})")

        # Save results to text file in the log directory
        log_dir = getattr(self.training_loop, "log_dir", None)
        if log_dir:
            iteration = getattr(self.training_loop, "current_learning_iteration", 0)
            txt_path = os.path.join(str(log_dir), f"success_rate_iter{iteration:05d}.txt")
            try:
                lines = []
                lines.append(f"=== Success Rate Report (iteration {iteration}) ===")
                for thresh in MOTION_FAR_THRESHOLDS:
                    terminate_np = self._terminate_states[thresh].cpu().numpy()
                    sr = 1.0 - terminate_np[:num_total].mean()
                    ns = int((~self._terminate_states[thresh][:num_total]).sum().item())
                    lines.append(f"Overall [{thresh}m]: {sr:.4f} ({ns}/{num_total})")
                lines.append("")

                # Per-motion results at each threshold (full paths for unambiguous matching)
                lib = self._motion_library
                for thresh in MOTION_FAR_THRESHOLDS:
                    terminate_np = self._terminate_states[thresh].cpu().numpy()
                    lines.append(f"--- Per-Motion [{thresh}m] ---")
                    for i in range(num_total):
                        status = "FAIL" if terminate_np[i] else "OK"
                        lines.append(f"  {status}: {lib._all_npz_files[i]}")

                with open(txt_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                logger.info(f"SuccessRateCallback: saved report to {txt_path}")
            except Exception as e:
                logger.warning(f"SuccessRateCallback: failed to save report: {e}")

        # Restore original pool size
        self._motion_library.pool_size = self._orig_pool_size
