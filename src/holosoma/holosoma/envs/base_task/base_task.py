from __future__ import annotations

from pathlib import Path

import numpy as np
from holosoma.config_types.env import EnvConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.curriculum import CurriculumManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.randomization import RandomizationManager
from holosoma.managers.reset_events.manager import ResetEventManager
from holosoma.managers.reward import RewardManager
from holosoma.managers.termination import TerminationManager
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import to_torch


# Base class for RL tasks built around the manager-based architecture.
class BaseTask:
    def __init__(
        self,
        tyro_config: EnvConfig,
        *,
        device: str,
    ):
        """Initialize task with manager-based observation, action, and reward systems.

        Parameters
        ----------
        tyro_config: EnvConfig
            Environment configuration
        device: str
            Device to run on
        """
        self._manager_domain_rand_cfg = None
        self.is_evaluating = False

        observation_config = tyro_config.observation
        simulator_config = tyro_config.simulator
        terrain_config = tyro_config.terrain
        robot_config = tyro_config.robot
        action_config = tyro_config.action
        reward_config = tyro_config.reward
        termination_config = tyro_config.termination
        randomization_config = tyro_config.randomization
        command_config = tyro_config.command
        curriculum_config = tyro_config.curriculum
        training_config = tyro_config.training

        self.training_config = training_config
        self.robot_config = robot_config

        # Validate configs: manager workflow requires all manager configs to be provided
        if observation_config is None:
            raise ValueError("observation_config must be provided for manager-based environments.")
        if action_config is None:
            raise ValueError("action_config must be provided for manager-based environments.")
        if reward_config is None:
            raise ValueError("reward_config must be provided for manager-based environments.")
        if termination_config is None:
            raise ValueError("termination_config must be provided for manager-based environments.")
        if randomization_config is None:
            raise ValueError("randomization_config must be provided for manager-based environments.")
        if command_config is None:
            raise ValueError("command_config must be provided for manager-based environments.")
        if curriculum_config is None:
            raise ValueError("curriculum_config must be provided for manager-based environments.")
        if terrain_config is None:
            raise ValueError("terrain_config must be provided for manager-based environments.")

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Use pre-computed experiment directory if provided (from train_agent.py),
        # otherwise compute one (for replay, eval, etc.)
        if tyro_config.experiment_dir is not None:
            experiment_dir = Path(tyro_config.experiment_dir)
        else:
            from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: PLC0415

            timestamp = get_timestamp()
            experiment_dir = get_experiment_dir(
                tyro_config.logger, tyro_config.training, timestamp, task_name=self._get_task_name()
            )

        SimulatorClass = get_class(simulator_config._target_)
        full_sim_config = FullSimConfig(
            simulator=simulator_config.config,
            robot=robot_config,
            training=training_config,
            logger=tyro_config.logger,
            experiment_dir=str(experiment_dir),
        )

        self.num_envs = training_config.num_envs
        self.dim_obs = robot_config.policy_obs_dim
        self.dim_critic_obs = robot_config.critic_obs_dim
        self.dim_actions = robot_config.actions_dim
        self.device = device

        self.terrain_manager = TerrainManager(terrain_config, self, device)
        self.simulator: BaseSimulator = SimulatorClass(
            tyro_config=full_sim_config, terrain_manager=self.terrain_manager, device=device
        )

        self.headless = self.training_config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.sim_dt = self.simulator.sim_dt

        self.dt = simulator_config.config.sim.control_decimation * self.sim_dt
        self.max_episode_length_s = simulator_config.config.sim.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.simulator.setup_terrain()
        # create envs, sim and viewer
        self._load_assets()

        # For IsaacGym manager-based environments: Initialize randomization manager BEFORE creating envs
        # so it can be applied during env creation (before prepare_sim).
        # For IsaacSim: The manager will be initialized later (scene is already created in __init__)
        is_isaacgym_manager = hasattr(self.simulator, "gym")
        if is_isaacgym_manager:
            self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
            if self.randomization_manager is not None:
                self.simulator.set_startup_randomization_callback(self.randomization_manager.setup)

        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()
        self.simulator.prepare_sim()

        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if not self.headless:
            self.debug_viz = False
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer

        # Initialize remaining managers
        self.observation_manager = ObservationManager(observation_config, self, self.device)
        self.action_manager = ActionManager(action_config, self, self.device)
        self.reward_manager = RewardManager(reward_config, self, self.device)
        self.termination_manager = TerminationManager(termination_config, self, self.device)
        # For IsaacSim, initialize randomization_manager now
        if not is_isaacgym_manager:
            self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
        self.command_manager = CommandManager(command_config, self, self.device)
        self.curriculum_manager = CurriculumManager(curriculum_config, self, self.device)

        self._init_buffers()

        # Prepare fields required by managers (BEFORE setup calls)
        # This scans decorator metadata and expands model fields for per-environment randomization
        self.simulator.prepare_manager_fields(
            randomization_manager=self.randomization_manager,
            observation_manager=self.observation_manager,
            reward_manager=self.reward_manager,
        )

        # Call setup for managers that need it
        if self.randomization_manager is not None and not is_isaacgym_manager:
            self.randomization_manager.setup()
        if self.action_manager is not None:
            self.action_manager.setup()
        if self.command_manager is not None:
            self.command_manager.setup()
        if self.curriculum_manager is not None:
            self.curriculum_manager.setup()
        if self.terrain_manager is not None:
            self.terrain_manager.setup()

        # Initialize reset manager from simulator config
        self.reset_manager = ResetEventManager(
            self.simulator.simulator_config.reset_manager, self.simulator, self.device
        )

        if not self.headless:
            self.viewer = self.simulator.viewer

    def _init_buffers(self):
        # Record history length from observation manager config
        self.history_length = {}
        for group_name, group_cfg in self.observation_manager.cfg.groups.items():
            self.history_length[group_name] = group_cfg.history_length

        self.obs_buf_dict = {}

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.log_dict = {}
        self._pending_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pending_torque_rfi: tuple[bool, float] = (False, 0.0)

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()

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

    def reset_all(self):
        """Reset all robots"""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(env_ids)

        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self.simulator.set_dof_state_tensor_robots(env_ids, self.simulator.dof_state)

        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Reset some environments and handle video recording callbacks."""

        # Call episode end for environments that are being reset
        for env_id in env_ids:
            if hasattr(self.simulator, "on_episode_end"):
                self.simulator.on_episode_end(env_id.item())

        # Reset observation history BEFORE state changes (must happen first to clear history buffers)
        self.observation_manager.reset(env_ids)

        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self._pending_episode_update_mask[env_ids] = False

        # Call the actual reset implementation (to be overridden by subclasses)
        self._reset_envs_idx_impl(env_ids, target_states, target_buf)

        # Reset all managers AFTER state changes
        if self.randomization_manager is not None:
            self.randomization_manager.reset(env_ids)

        if self.action_manager is not None:
            self.action_manager.reset(env_ids)

        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        if self.curriculum_manager is not None:
            self.curriculum_manager.reset(env_ids)

        if self.termination_manager is not None:
            self.termination_manager.reset(env_ids)

        # Call manager-based reset events
        self.reset_manager.reset_scene(env_ids)

        # Call episode start for environments that have been reset
        for env_id in env_ids:
            if hasattr(self.simulator, "on_episode_start"):
                self.simulator.on_episode_start(env_id.item())

    def _reset_envs_idx_impl(self, env_ids, target_states=None, target_buf=None):
        """Template implementation of environment reset.

        Subclasses can override the helper hooks below to customize the reset behaviour.

        Args
        ----
        env_ids:
            Environments to reset.
        target_states:
            Optional dictionary containing desired DOF/root states.
        target_buf:
            Optional dictionary of buffered tensors to restore (e.g., for replay).
        """
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)
        self._reset_robot_states_callback(env_ids, target_states)
        self._fill_extras(env_ids)

    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions

    @property
    def domain_rand_cfg(self):
        """Return the active domain randomization configuration."""
        return self._manager_domain_rand_cfg

    ###########################################################################
    def _load_assets(self):
        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
        )

        # check dimensions
        assert self.num_dof == self.dim_actions, (
            f"Number of DOFs ({self.num_dof}) must be equal to number of actions ({self.dim_actions})"
        )

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = (
            self.robot_config.init_state.pos
            + self.robot_config.init_state.rot
            + self.robot_config.init_state.lin_vel
            + self.robot_config.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        self.simulator.create_envs(self.num_envs, self._get_env_origins(), self.base_init_state)

    def _setup_robot_body_indices(self):
        """Hook for subclasses to prepare body index caches (default no-op)."""

    def set_is_evaluating(self) -> None:
        """
        Called by agent during pre_evaluate_policy
        """
        self.is_evaluating = True

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _get_task_name(self) -> str:
        """Return a task identifier for logging/experiment directory naming."""
        training_task_name = getattr(self.training_config, "task_name", None)
        if isinstance(training_task_name, str) and training_task_name:
            return training_task_name
        return self.__class__.__name__.lower()

    def _get_env_origins(self):
        """Return environment origins used when creating simulator environments."""
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state is None or not hasattr(terrain_state, "env_origins"):
            raise RuntimeError("Terrain manager state 'locomotion_terrain' must provide env_origins.")
        return terrain_state.env_origins

    # ------------------------------------------------------------------
    # Reset hooks

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        """Reset environment-specific buffers prior to manager resets.

        Default implementation is a no-op. Override in subclasses to zero custom tensors
        or restore from a buffered state.
        """

    def _reset_tasks_callback(self, env_ids):
        """Hook for subclasses to extend reset-time logic."""

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        """Reset simulator DOF/root states for the specified environments.

        Subclasses must implement this to place robots back into their initial configuration.
        """
        raise NotImplementedError("Subclasses must implement `_reset_robot_states_callback` to reset simulator states.")

    def _fill_extras(self, env_ids):
        """Populate per-episode extras after a reset."""
        if self.reward_manager is None:
            return

        reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})

        self.extras["time_outs"] = self.time_out_buf

    ###########################################################################
    # Simulation loop helpers

    def step(self, actor_state):
        """Apply actions, advance the simulation, and return rollout buffers."""
        actions = actor_state["actions"]
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        if self.action_manager is not None:
            self.action_manager.process_actions(actions)

    def _physics_step(self):
        self.render()
        for _ in range(self.simulator.simulator_config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        if self.action_manager is not None:
            self.action_manager.apply_actions()

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        self._update_counters_each_step()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()  # needs to be called before reset_envs_idx
        self._check_termination()
        self._compute_reward()
        self._update_log_dict()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        final_obs_dict = {}
        if env_ids.numel() > 0:
            final_obs_dict = self._compute_final_observations()

        self.reset_envs_idx(env_ids)

        refresh_env_ids = self._ensure_long_tensor(self._get_envs_to_refresh())
        if refresh_env_ids.numel() > 0:
            self._refresh_envs_after_reset(refresh_env_ids)

        self._compute_observations()

        if env_ids.numel() > 0 and final_obs_dict:
            env_ids_long = self._ensure_long_tensor(env_ids)
            self._store_final_observations(env_ids_long, final_obs_dict)

        self._post_compute_observations_callback()
        self._clip_observations()

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()

    def _ensure_long_tensor(self, tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(tensor_like, device=self.device, dtype=torch.long)

    def _get_envs_to_refresh(self):
        return torch.empty(0, device=self.device, dtype=torch.long)

    def _refresh_envs_after_reset(self, env_ids):
        """Hook for subclasses to synchronise simulator state after resets."""
        return

    def _store_final_observations(self, env_ids, final_obs_dict):
        if not final_obs_dict:
            return
        final_store = self.extras.setdefault("final_observations", {})
        for obs_key, values in final_obs_dict.items():
            if obs_key not in final_store:
                final_store[obs_key] = torch.zeros_like(self.obs_buf_dict[obs_key])
            final_store[obs_key][env_ids] = values[env_ids]

    def _clip_observations(self):
        clip_limit = self.observation_manager.cfg.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_limit, clip_limit)

    def _compute_reward(self):
        self.rew_buf[:] = self.reward_manager.compute(self.dt)
        self.episode_sums = getattr(self.reward_manager, "episode_sums", {})
        self.episode_sums_raw = getattr(self.reward_manager, "episode_sums_raw", {})

    def _compute_observations(self):
        self.obs_buf_dict = self.observation_manager.compute()

    def _compute_final_observations(self):
        return self.observation_manager.compute(modify_history=False)

    def _update_tasks_callback(self):
        self.command_manager.step()
        self.curriculum_manager.step()
        self.randomization_manager.step()

    def _init_counters(self):
        return

    def _update_counters_each_step(self):
        return

    def _check_termination(self):
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        if self.termination_manager is None:
            return

        reset_flags, timeout_flags = self.termination_manager.check()
        self.reset_buf |= reset_flags.to(dtype=self.reset_buf.dtype)
        self.time_out_buf |= timeout_flags
        self.reset_buf |= self.time_out_buf

    def _pre_compute_observations_callback(self):
        """Hook invoked after physics but before observation terms compute (no-op by default)."""
        return

    def _post_compute_observations_callback(self):
        """Hook invoked after observation buffers are produced (no-op by default)."""
        return

    def _setup_simulator_control(self):
        """Hook for pushing controller state back to the simulator/viewer (no-op by default)."""
        return

    def _setup_simulator_next_task(self):
        """Hook for interactive viewer task selection (no-op by default)."""
        return

    def _update_log_dict(self):
        """Hook for appending task-specific metrics to `self.log_dict` (no-op by default)."""
        return
