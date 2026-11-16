from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from holosoma.envs.policy_task import PolicyTask
from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker, PenaltyCurriculum
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float

if TYPE_CHECKING:
    from holosoma.managers.command.terms.locomotion import LocomotionGait
    from holosoma.managers.randomization.terms.locomotion import PushRandomizerState
    from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion


_LEGACY_CHECKPOINT_REMOVAL_DATE = datetime.date(2027, 3, 1)


class LeggedRobotLocomotionManager(PolicyTask):
    BASE_NUM_ENVS = 4096

    def _get_task_name(self) -> str:
        return "locomotion"

    def _init_buffers(self) -> None:
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()
        self._init_domain_rand_buffers()

    def _init_domain_rand_buffers(self) -> None:
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)

    def _setup_robot_body_indices(self) -> None:
        super()._setup_robot_body_indices()
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]

        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

    def set_is_evaluating(self, command: Any = None) -> None:
        logger.info("Setting Env is evaluating")
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = cast("LocomotionGait", self.command_manager.get_state("locomotion_gait"))
        gait_state.set_eval_mode(True)

    @property
    def base_quat(self) -> torch.Tensor:
        return self.simulator.base_quat

    def _pre_compute_observations_callback(self) -> None:
        self.terrain_manager.update_heights()

    def _reset_robot_states_callback(
        self, env_ids: torch.Tensor, target_states: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        if target_states is not None:
            # Target root positions are relative to each environment's origin.
            root_states = target_states["root_states"].clone()
            root_states[:, :3] += self._get_env_origins()[env_ids]
            super()._reset_robot_states_callback(
                env_ids, {"root_states": root_states, "dof_states": target_states["dof_states"]}
            )
            return
        super()._reset_robot_states_callback(env_ids)
        self._randomize_spawn(env_ids)

    def _reset_buffers_callback(
        self, env_ids: torch.Tensor, target_buf: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        super()._reset_buffers_callback(env_ids)
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0

    ################ Curriculum #################

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if "curriculum" not in state:
            state = self._convert_legacy_checkpoint_state(state)
        super().load_checkpoint_state(state)

    def _term_name(self, term_cls: type) -> str | None:
        return next((name for name, term in self.curriculum_manager.iter_terms() if isinstance(term, term_cls)), None)

    def _convert_legacy_checkpoint_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Map the pre-``curriculum`` checkpoint layout onto curriculum term state."""
        curriculum: dict[str, dict[str, Any]] = {}
        tracker_state = state.get("average_episode_tracker")
        tracker_name = self._term_name(AverageEpisodeLengthTracker)
        if tracker_state is not None and tracker_name is not None:
            curriculum[tracker_name] = tracker_state
        penalty_scale = state.get("reward_penalty_scale")
        penalty_name = self._term_name(PenaltyCurriculum)
        if penalty_scale is not None and penalty_name is not None:
            curriculum[penalty_name] = {"current_scale": float(penalty_scale)}
        if curriculum:
            message = (
                "This checkpoint uses the old env-state layout (top-level 'average_episode_tracker' / "
                "'reward_penalty_scale'), which is deprecated. Re-save it to store curriculum state per term "
                f"under 'curriculum'. Loading the old layout becomes an error after {_LEGACY_CHECKPOINT_REMOVAL_DATE}."
            )
            if datetime.datetime.now(tz=datetime.timezone.utc).date() > _LEGACY_CHECKPOINT_REMOVAL_DATE:
                raise RuntimeError(message)
            # FutureWarning: shown by default (DeprecationWarning is filtered outside __main__).
            warnings.warn(message, FutureWarning, stacklevel=3)
        return {**state, "curriculum": curriculum}

    def _push_robots(self, env_ids: torch.Tensor) -> None:
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = cast("PushRandomizerState | None", self.randomization_manager.get_state("push_randomizer_state"))
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError("Locomotion push velocity vector must have exactly 2 components.")

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        self._max_push_vel = max_vel_tensor.clone()

    ################ ENV CALLBACKS #################

    def _randomize_spawn(self, env_ids: torch.Tensor) -> None:
        """Offset the default spawn on custom-origin terrains and randomize the base velocity."""
        # Apply randomized XY offset if custom_origins
        if self.terrain_manager.get_state("locomotion_terrain").custom_origins:
            # Get spawn config from terrain term
            spawn_cfg = self.terrain_manager.cfg.terrain_term.spawn

            # Generate random XY offsets
            xy_offsets = torch_rand_float(
                -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(self.device)
            )

            if spawn_cfg.query_terrain_height:
                # ACCURATE: Query terrain height at new XY position (slower, for rough terrain)
                current_xy = self.simulator.robot_root_states[env_ids, :2]
                new_xy = current_xy + xy_offsets

                terrain_state = cast("TerrainLocomotion", self.terrain_manager.get_state("locomotion_terrain"))
                terrain_heights = terrain_state.query_terrain_heights(
                    new_xy,
                    use_grid_sampling=spawn_cfg.use_grid_sampling,
                    grid_size=spawn_cfg.grid_size,
                    grid_spacing=spawn_cfg.grid_spacing,
                )
                robot_base_height = self.robot_config.init_state.pos[2]  # Robot base height above ground
                new_z = terrain_heights + robot_base_height

                # Write new XYZ position all at once
                new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                self.simulator.robot_root_states[env_ids, :3] = new_xyz
            else:
                # FAST: Original simple spawning - just apply XY offset, keep original Z (faster, for flat terrain)
                self.simulator.robot_root_states[env_ids, :2] += xy_offsets

        # base velocities
        self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
        )  # [7:10]: lin vel, [10:13]: ang vel
