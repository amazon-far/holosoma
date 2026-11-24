"""Task specialization that owns the policy managers and rollout buffers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from holosoma.config_types.env import EnvConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.base_task.task_hooks import TaskHook
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.reward import RewardManager
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import setup_keyboard_listener


class PolicyTask(BaseTask):
    """BaseTask specialization for policy training and evaluation."""

    def __init__(self, tyro_config: EnvConfig) -> None:
        for name in ("observation", "action", "reward", "termination", "command"):
            if getattr(tyro_config, name) is None:
                raise ValueError(f"{name}_config must be provided for manager-based environments.")
        self.is_evaluating = False
        self.dim_obs = tyro_config.robot.policy_obs_dim
        super().__init__(tyro_config)
        if not self.headless:
            setup_keyboard_listener(self)

    def _initialize_task_managers(self) -> None:
        super()._initialize_task_managers()
        config = self.config
        assert config.observation is not None
        assert config.action is not None
        assert config.reward is not None
        assert config.command is not None
        assert config.curriculum is not None

        self.observation_manager = ObservationManager(config.observation, self, self.device)
        self.action_manager = ActionManager(config.action, self, self.device)
        self.reward_manager = RewardManager(config.reward, self, self.device)
        self.command_manager = CommandManager(config.command, self, self.device)

    def _prepare_manager_fields(self) -> None:
        self.simulator.prepare_manager_fields(
            randomization_manager=self.randomization_manager,
            observation_manager=self.observation_manager,
            reward_manager=self.reward_manager,
        )

    def _setup_task_managers(self) -> None:
        super()._setup_task_managers()
        self.action_manager.setup()
        self.command_manager.setup()

    def _init_buffers(self) -> None:
        super()._init_buffers()

        # Record history length from observation manager config
        self.history_length: dict[str, int] = {}
        for group_name, group_cfg in self.observation_manager.cfg.groups.items():
            self.history_length[group_name] = group_cfg.history_length

        self.obs_buf_dict: dict[str, Any] = {}
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.extras: dict[str, Any] = {}
        self.log_dict: dict[str, Any] = {}
        self._pending_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pending_torque_rfi: tuple[bool, float] = (False, 0.0)
        self._metric_sources: list[Any] = []

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return task state to persist in checkpoints; the base policy task has none."""
        return {}

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore task state returned by :meth:`get_checkpoint_state`."""

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
        self._pending_episode_update_mask[env_ids] = True

    def _frame_end_steps(self) -> list[tuple[TaskHook, Callable[[], None]]]:
        base = dict(super()._frame_end_steps())
        return [
            (TaskHook.COUNT_FRAME, base[TaskHook.COUNT_FRAME]),
            (TaskHook.TERMINATION, base[TaskHook.TERMINATION]),
            (TaskHook.OBSERVATION_FRAME, lambda: self.observation_manager.new_frame()),
            (TaskHook.PRE_COMPUTE_OBSERVATIONS, lambda: self._pre_compute_observations_callback()),
            (TaskHook.REWARD, lambda: self._compute_reward()),
            (TaskHook.UPDATE_LOG_DICT, lambda: self._update_log_dict()),
            (TaskHook.TERMINAL_OBSERVATIONS, lambda: self._store_timeout_observations()),
            (TaskHook.RESET, base[TaskHook.RESET]),
            # After resets, so WBT termination checks see the tracking error before the command
            # manager advances the motion clip.
            (TaskHook.COMMAND_STEP, lambda: self.command_manager.step()),
            (TaskHook.RANDOMIZATION_STEP, base[TaskHook.RANDOMIZATION_STEP]),
            (TaskHook.METRICS, lambda: self._collect_metrics()),
            (TaskHook.OBSERVATIONS, lambda: self._compute_observations()),
        ]

    def _reset_steps(self) -> list[tuple[TaskHook, Callable[[torch.Tensor], None]]]:
        base = dict(super()._reset_steps())
        return [
            (TaskHook.PREPARE_RESET, base[TaskHook.PREPARE_RESET]),
            (TaskHook.WRITE_STATE, base[TaskHook.WRITE_STATE]),
            (TaskHook.COUNT_EPISODE, base[TaskHook.COUNT_EPISODE]),
            (TaskHook.REWARD_RESET, lambda env_ids: self._reset_reward(env_ids)),
            (TaskHook.RANDOMIZATION_RESET, base[TaskHook.RANDOMIZATION_RESET]),
            (TaskHook.TERMINATION_RESET, base[TaskHook.TERMINATION_RESET]),
            (TaskHook.ACTION_RESET, lambda env_ids: self.action_manager.reset(env_ids)),
            (TaskHook.COMMAND_RESET, lambda env_ids: self.command_manager.reset(env_ids)),
            (TaskHook.MANAGERS_RESET, base[TaskHook.MANAGERS_RESET]),
            (TaskHook.REFRESH, base[TaskHook.REFRESH]),
            (TaskHook.AFTER_RESET_SYNC, base[TaskHook.AFTER_RESET_SYNC]),
        ]

    def _reset_reward(self, env_ids: torch.Tensor) -> None:
        reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})

        self.extras["time_outs"] = self.time_out_buf

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        self._pre_compute_observations_callback()

    def reset_all(self) -> dict[str, Any]:
        super().reset_all()
        return self.compute_reset_observations()

    def compute_reset_observations(self) -> dict[str, Any]:
        """Compute observations of the current state after a reset, without advancing physics."""
        self.observation_manager.new_frame()
        self.obs_buf_dict = self.observation_manager.step()
        return self.obs_buf_dict

    # ------------------------------------------------------------------
    # Control frame

    def step(self, actor_state: dict[str, Any] | None = None) -> Any:
        """Apply ``actor_state["actions"]`` and advance one control frame."""
        if actor_state is None:
            raise TypeError("PolicyTask.step() requires an actor_state with 'actions'.")
        self._pre_physics_step(actor_state)
        return super().step()

    def _pre_physics_step(self, actor_state: dict[str, Any]) -> None:
        self.action_manager.process_actions(actor_state["actions"])

    def _apply_force_in_physics_step(self) -> None:
        self.action_manager.apply_actions()

    def _store_timeout_observations(self) -> None:
        # Time-outs are a subset of resets; only they are bootstrapped from a terminal observation.
        timeout_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        if timeout_ids.numel() > 0:
            self._store_final_observations(timeout_ids, self.observation_manager.step_terminal(timeout_ids))

    def _post_physics_step(self) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        super()._post_physics_step()
        self.extras["to_log"] = self.log_dict
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _compute_observations(self) -> None:
        self.obs_buf_dict = self.observation_manager.step()

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

    def _collect_metrics(self) -> None:
        """Copy the metrics every manager's class terms publish into ``log_dict``.

        A term publishes by keeping a ``metrics`` mapping of tensors, refreshed wherever it computes them;
        it is read once per frame, by the ``METRICS`` step after the managers step and before
        observations. Keys are logged as ``Env/<key>``, averaged over the logging window.

        Raises
        ------
        ValueError
            If two terms publish the same key.
        """
        publishers: dict[str, str] = {}
        for manager in self._metric_managers():
            for name, term in manager.iter_terms():
                for key, value in getattr(term, "metrics", {}).items():
                    if key in publishers:
                        raise ValueError(f"Metric '{key}' is published by both '{publishers[key]}' and '{name}'.")
                    publishers[key] = name
                    self.log_dict[key] = value

    def _metric_managers(self) -> list[Any]:
        """Managers whose class terms may publish metrics."""
        return [
            self.action_manager,
            self.command_manager,
            self.reward_manager,
            self.termination_manager,
            self.randomization_manager,
            self.terrain_manager,
            *self._metric_sources,
        ]

    def add_metric_source(self, source: Any) -> None:
        """Also collect the metrics of ``source``'s terms, which it yields from ``iter_terms()``."""
        self._metric_sources.append(source)

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _pre_compute_observations_callback(self) -> None:
        """Hook invoked after physics but before observation terms compute (no-op by default)."""
        return

    def _update_log_dict(self) -> None:
        """Hook invoked after rewards to append per-frame metrics to ``log_dict`` (no-op by default)."""
