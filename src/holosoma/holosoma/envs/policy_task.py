"""Task specialization that owns the policy managers and rollout buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.curriculum import CurriculumManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.reward import RewardManager
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import setup_keyboard_listener

if TYPE_CHECKING:
    from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker


class PolicyTask(BaseTask):
    """BaseTask specialization for policy training and evaluation."""

    def __init__(self, tyro_config: EnvConfig) -> None:
        for name in ("observation", "action", "reward", "termination", "command", "curriculum"):
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
        self.curriculum_manager = CurriculumManager(config.curriculum, self, self.device)

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
        self.curriculum_manager.setup()

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

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return environment-specific state to persist in checkpoints."""
        state: dict[str, Any] = {}
        curriculum_state = self.curriculum_manager.state_dict()
        if curriculum_state:
            state["curriculum"] = curriculum_state
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore environment-specific state from a checkpoint."""
        if not state:
            return
        curriculum_state = state.get("curriculum")
        if not curriculum_state:
            return
        self.curriculum_manager.load_state_dict(curriculum_state)
        for name in curriculum_state:
            term = self.curriculum_manager.get_term(name)
            if term is not None:
                term.after_checkpoint_load()

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        """Broadcast rank 0's curriculum state to every rank."""
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        # Every rank builds the same terms from the same config, so all ranks agree on skipping. A term
        # whose state is empty depending on runtime data would deadlock here.
        state = self.curriculum_manager.state_dict()
        if not state:
            return
        if not getattr(self, "_logged_curriculum_sync", False):
            logger.info(f"Multi-GPU curriculum synchronization enabled for terms: {sorted(state)}")
            self._logged_curriculum_sync = True
        payload = [state]
        torch.distributed.broadcast_object_list(payload, src=0, device=torch.device(device))
        if torch.distributed.get_rank() != 0:
            self.curriculum_manager.load_state_dict(payload[0])

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

    def _reset_task_managers(self, env_ids: torch.Tensor) -> None:
        reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})

        self.extras["time_outs"] = self.time_out_buf

        super()._reset_task_managers(env_ids)
        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)

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

    def _evaluate_frame(self, env_ids: torch.Tensor) -> None:
        self.observation_manager.new_frame()
        self._pre_compute_observations_callback()
        self._compute_reward()
        self._update_log_dict()
        # Time-outs are a subset of resets; only they are bootstrapped from a terminal observation.
        timeout_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        if timeout_ids.numel() > 0:
            self._store_final_observations(timeout_ids, self.observation_manager.step_terminal(timeout_ids))

    def _post_physics_step(self) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        super()._post_physics_step()
        self.obs_buf_dict = self.observation_manager.step()
        self.extras["to_log"] = self.log_dict
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _step_managers(self) -> None:
        # After resets, so WBT termination checks see the tracking error before the command
        # manager advances the motion clip.
        self.command_manager.step()
        self.curriculum_manager.step()
        super()._step_managers()

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

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _pre_compute_observations_callback(self) -> None:
        """Hook invoked after physics but before observation terms compute (no-op by default)."""
        return

    def _update_log_dict(self) -> None:
        """Append per-frame metrics to ``log_dict``: the average episode length when it is tracked."""
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is not None:
            avg = cast("AverageEpisodeLengthTracker", tracker).get_average_value()
            self.log_dict["average_episode_length"] = torch.tensor(avg, dtype=torch.float32)

    def _get_average_episode_tracker(self) -> AverageEpisodeLengthTracker:
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return cast("AverageEpisodeLengthTracker", tracker)
