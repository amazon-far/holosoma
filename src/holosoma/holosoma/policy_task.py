"""Task specialization that owns the policy managers and rollout buffers."""

from __future__ import annotations

from typing import Any

import numpy as np

from holosoma.config_types.env import EnvConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.curriculum import CurriculumManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.reward import RewardManager
from holosoma.managers.termination import TerminationManager
from holosoma.utils.safe_torch_import import torch


class PolicyTask(BaseTask):
    """BaseTask specialization for policy training and evaluation."""

    def __init__(self, tyro_config: EnvConfig) -> None:
        self.is_evaluating = False
        self.dim_obs = tyro_config.robot.policy_obs_dim
        self.dim_critic_obs = tyro_config.robot.critic_obs_dim
        self.dim_actions = tyro_config.robot.actions_dim
        super().__init__(tyro_config)

    def _validate_task(self) -> None:
        # Validate configs: manager workflow requires all manager configs to be provided
        for name in ("observation", "action", "reward", "termination", "command", "curriculum"):
            if getattr(self.config, name) is None:
                raise ValueError(f"{name}_config must be provided for manager-based environments.")

    def _validate_loaded_assets(self) -> None:
        assert self.num_dof == self.dim_actions, (
            f"Number of DOFs ({self.num_dof}) must be equal to number of actions ({self.dim_actions})"
        )

    def _initialize_task_managers(self) -> None:
        config = self.config
        assert config.observation is not None
        assert config.action is not None
        assert config.reward is not None
        assert config.termination is not None

        self.max_episode_length_s = config.simulator.config.sim.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.observation_manager = ObservationManager(config.observation, self, self.device)
        self.action_manager = ActionManager(config.action, self, self.device)
        self.reward_manager = RewardManager(config.reward, self, self.device)
        self.termination_manager = TerminationManager(config.termination, self, self.device)

    def _initialize_post_randomization_task_managers(self) -> None:
        assert self.config.command is not None
        assert self.config.curriculum is not None
        self.command_manager = CommandManager(self.config.command, self, self.device)
        self.curriculum_manager = CurriculumManager(self.config.curriculum, self, self.device)

    def _prepare_manager_fields(self) -> None:
        self.simulator.prepare_manager_fields(
            randomization_manager=self.randomization_manager,
            observation_manager=self.observation_manager,
            reward_manager=self.reward_manager,
        )

    def _setup_task_managers(self) -> None:
        self.action_manager.setup()
        self.command_manager.setup()
        self.curriculum_manager.setup()

    def _init_buffers(self) -> None:
        super()._init_buffers()

        # Record history length from observation manager config
        self.history_length: dict[str, int] = {}
        for group_name, group_cfg in self.observation_manager.cfg.groups.items():
            self.history_length[group_name] = group_cfg.history_length

        self.obs_buf_dict: dict[str, Any] = {}
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras: dict[str, Any] = {}
        self.log_dict: dict[str, Any] = {}
        self._pending_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pending_torque_rfi: tuple[bool, float] = (False, 0.0)

    def get_checkpoint_state(self) -> dict[str, torch.Tensor | float]:
        """Return environment-specific state to persist in checkpoints."""
        return {}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor | float] | None) -> None:
        """Restore environment-specific state from a checkpoint."""
        if not state:
            return

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        """Synchronize curriculum-related state across distributed processes."""
        return

    def set_is_evaluating(self) -> None:
        """
        Called by agent during pre_evaluate_policy
        """
        self.is_evaluating = True

    # ------------------------------------------------------------------
    # Reset hooks

    def _prepare_reset(self, env_ids: torch.Tensor) -> None:
        # Reset observation history BEFORE state changes (must happen first to clear history buffers)
        self.observation_manager.reset(env_ids)
        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self._pending_episode_update_mask[env_ids] = False

    def _reset_task_state(self, env_ids: torch.Tensor) -> None:
        """Populate per-episode extras after a reset."""
        reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})

        self.extras["time_outs"] = self.time_out_buf

    def _reset_task_managers(self, env_ids: torch.Tensor) -> None:
        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        self._pre_compute_observations_callback()

    def reset_all(self) -> dict[str, Any]:
        obs_dict: dict[str, Any] = super().reset_all()
        return obs_dict

    def _finish_reset(self) -> dict[str, Any]:
        # reset_all() does not step physics, so it opens its own observation frame.
        self.observation_manager.new_frame()
        self.obs_buf_dict = self.observation_manager.step()
        self._post_compute_observations_callback()
        return self.obs_buf_dict

    # ------------------------------------------------------------------
    # Control frame

    def _pre_physics_step(self, actor_state: dict[str, Any]) -> None:
        self.action_manager.process_actions(actor_state["actions"])

    def _apply_force_in_physics_step(self) -> None:
        self.action_manager.apply_actions()

    def _after_physics_step(self) -> None:
        self.observation_manager.new_frame()
        self._update_counters_each_step()
        self._pre_compute_observations_callback()

    def _step_managers(self) -> None:
        self.command_manager.step()
        self.curriculum_manager.step()
        super()._step_managers()

    def _get_control_frame_reset_env_ids(self) -> torch.Tensor:
        self._check_termination()
        return self.reset_buf.nonzero(as_tuple=False).flatten()

    def _before_control_frame_reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        self._compute_reward()
        self._update_log_dict()
        if env_ids.numel() == 0:
            # Keep per-frame extras current when no environment resets this frame.
            self._reset_task_state(env_ids)
            return env_ids, {}
        # Only timed-out envs are bootstrapped from their terminal observation (time-outs are a subset of resets).
        timeout_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        if timeout_ids.numel() == 0:
            return timeout_ids, {}
        return timeout_ids, self.observation_manager.step_terminal(timeout_ids)

    def _after_control_frame(
        self, env_ids: torch.Tensor, reset_state: tuple[torch.Tensor, dict[str, Any]]
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        timeout_ids, final_obs_dict = reset_state
        self.obs_buf_dict = self.observation_manager.step()
        if final_obs_dict:
            self._store_final_observations(self._ensure_long_tensor(timeout_ids), final_obs_dict)

        self._post_compute_observations_callback()

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _store_final_observations(self, env_ids: torch.Tensor, final_obs_dict: dict[str, Any]) -> None:
        if not final_obs_dict:
            return
        final_store = self.extras.setdefault("final_observations", {})

        def _store_value(store_parent: dict[str, Any], key: str, value: Any, template: Any) -> None:
            # A concatenate=False group is a dict of per-term tensors; recurse one level so the
            # per-env final-obs copy works for image groups too (env-axis is dim 0 either way).
            if isinstance(value, dict):
                sub = store_parent.setdefault(key, {})
                for sub_key, sub_val in value.items():
                    _store_value(sub, sub_key, sub_val, template[sub_key])
                return
            if key not in store_parent:
                store_parent[key] = torch.zeros_like(template)
            store_parent[key][env_ids] = value

        for obs_key, values in final_obs_dict.items():
            _store_value(final_store, obs_key, values, self.obs_buf_dict[obs_key])

    def _compute_reward(self) -> None:
        self.rew_buf[:] = self.reward_manager.compute(self.dt)
        self.episode_sums = getattr(self.reward_manager, "episode_sums", {})
        self.episode_sums_raw = getattr(self.reward_manager, "episode_sums_raw", {})

    def _check_termination(self) -> None:
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        reset_flags, timeout_flags = self.termination_manager.check()
        self.reset_buf |= reset_flags.to(dtype=self.reset_buf.dtype)
        self.time_out_buf |= timeout_flags
        self.reset_buf |= self.time_out_buf

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _init_counters(self) -> None:
        return

    def _update_counters_each_step(self) -> None:
        return

    def _pre_compute_observations_callback(self) -> None:
        """Hook invoked after physics but before observation terms compute (no-op by default)."""
        return

    def _post_compute_observations_callback(self) -> None:
        """Hook invoked after observation buffers are produced (no-op by default)."""
        return

    def _setup_simulator_control(self) -> None:
        """Hook for pushing controller state back to the simulator/viewer (no-op by default)."""
        return

    def _setup_simulator_next_task(self) -> None:
        """Hook for interactive viewer task selection (no-op by default)."""
        return

    def _update_log_dict(self) -> None:
        """Hook for appending task-specific metrics to `self.log_dict` (no-op by default)."""
        return
