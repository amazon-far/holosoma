from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.managers.randomization import RandomizationManager
from holosoma.managers.reset_events.manager import ResetEventManager
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator, _warn_deprecated
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.session import get_local_rank, get_process_seed, resolve_simulation_device
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.common import seeding
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import to_torch


# Base class for manager-based simulation tasks.
class BaseTask:
    def __init__(self, tyro_config: EnvConfig) -> None:
        self._closed = False
        try:
            self._initialize_base_task(tyro_config)
        except BaseException:
            simulator = getattr(self, "simulator", None)
            if simulator is not None:
                try:
                    simulator.close()
                except Exception:
                    logger.exception("Simulator cleanup failed while preserving the task setup error")
            raise

    def run(self) -> None:
        """Run this task's outer execution loop."""
        raise NotImplementedError(f"{type(self).__name__} must implement run().")

    def close(self) -> None:
        """End active episodes and close the simulator exactly once."""
        if self._closed:
            return
        self._closed = True

        primary_error: BaseException | None = None
        try:
            self._end_episodes(self._episode_active.nonzero(as_tuple=False).flatten())
        except BaseException as exc:
            primary_error = exc

        try:
            self.simulator.close()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                logger.opt(exception=exc).error("Simulator cleanup also failed")

        if primary_error is not None:
            raise primary_error

    def _initialize_base_task(self, tyro_config: EnvConfig) -> None:
        """Initialize the simulator and BaseTask-owned managers.

        Parameters
        ----------
        tyro_config: EnvConfig
            Environment configuration
        """
        self.config = tyro_config
        self._manager_domain_rand_cfg = None

        simulator_config = tyro_config.simulator
        terrain_config = tyro_config.terrain
        robot_config = tyro_config.robot
        training_config = tyro_config.training
        randomization_config = tyro_config.randomization or RandomizationManagerCfg()

        self.training_config = training_config
        self.robot_config = robot_config

        seeding(get_process_seed(tyro_config), torch_deterministic=training_config.torch_deterministic)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Compute experiment directory from logger config
        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(
            tyro_config.logger, tyro_config.training, timestamp, task_name=self._get_task_name()
        )

        SimulatorClass = get_class(simulator_config._target_)
        full_sim_config = FullSimConfig(
            simulator=simulator_config.config,
            robot=robot_config,
            scene=tyro_config.scene,
            sensors=tyro_config.sensors,
            plugin=tyro_config.plugin,
            training=training_config,
            logger=tyro_config.logger,
            experiment_dir=str(experiment_dir),
        )

        self.num_envs = training_config.num_envs
        self.device = resolve_simulation_device(tyro_config, get_local_rank())

        # Episodes each environment has started, incremented on reset before the reset DR runs. With
        # dr_base_seed it keys every DR draw as f(seed, term, env, episode), so async per-env resets
        # reproduce. Set before any manager is constructed so setup-stage DR already sees them.
        self.dr_base_seed = training_config.seed
        self.episode_count = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")

        self.terrain_manager = TerrainManager(terrain_config, self, self.device)
        self.simulator: BaseSimulator = SimulatorClass(
            tyro_config=full_sim_config, terrain_manager=self.terrain_manager, device=self.device
        )
        self.hooks = self.simulator.hooks

        self._validate_task()
        self.headless = training_config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.sim_dt = self.simulator.sim_dt

        self.dt = simulator_config.config.sim.control_decimation_steps * self.sim_dt
        self._rendered_this_substep = False

        self.simulator.setup_terrain()
        # create envs, sim and viewer
        self._load_assets()

        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()
        self.simulator.prepare_sim()
        self._initial_robot_root_states = self.simulator.robot_root_states[:].clone()
        self._initial_robot_dof_states = torch.stack(
            (self.simulator.dof_pos.clone(), self.simulator.dof_vel.clone()),
            dim=-1,
        )

        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if not self.headless:
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer

        # Plugins may depend on backend/provider resources created above. Their CLOSE hooks are
        # registered last and therefore run first during reverse-order teardown.
        self.simulator.install_plugins()

        # Randomization is built between the task's managers because some terms bind to one another
        # during construction.
        self._initialize_task_managers()
        self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
        self._initialize_post_randomization_task_managers()

        self._episode_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._init_buffers()

        # Prepare fields required by managers (BEFORE setup calls)
        # This scans decorator metadata and expands model fields for per-environment randomization
        self._prepare_manager_fields()

        # Call setup for managers that need it
        self._setup_task_managers()

        # Initialize reset manager from simulator config
        self.reset_manager = ResetEventManager(
            self.simulator.simulator_config.reset_manager, self.simulator, self.device
        )

        self._add_pacing_hooks()

    def _add_pacing_hooks(self) -> None:
        """Draw the viewer every ``viewer_dt`` and pace physics to ``time_scale``.

        Registered after every other hook, so pacing runs last in each substep.
        """
        if self.viewer is not None:
            every = max(1, int(self.config.viewer_dt / self.sim_dt))
            self.hooks.add(Phase.POST_STEP, self._render_viewer, name="task.viewer", every=every)
        if self.config.time_scale > 0:
            rate_limiter = RateLimiter(self.config.time_scale / self.sim_dt)
            self.hooks.add(Phase.POST_STEP, rate_limiter.sleep, name="task.rate_limit")

    def _render_viewer(self) -> None:
        if not self._rendered_this_substep:
            self.render()

    @property
    def dr_episode_count(self) -> torch.Tensor:
        """Deprecated alias of :attr:`episode_count`."""
        _warn_deprecated("dr_episode_count", "use episode_count")
        return self.episode_count

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def reset_all(self) -> Any:
        """Deprecated: reset every environment with :meth:`reset_envs_idx`."""
        _warn_deprecated("reset_all", "call reset_envs_idx(), then compute_reset_observations() on a PolicyTask")
        self.reset_envs_idx()

    def _end_episodes(self, env_ids: torch.Tensor) -> None:
        self._episode_active[env_ids] = False
        for env_id in env_ids.tolist():
            self.hooks.emit(Phase.EPISODE_END, env_id)

    def _start_episodes(self, env_ids: torch.Tensor) -> None:
        self._episode_active[env_ids] = True
        for env_id in env_ids.tolist():
            self.hooks.emit(Phase.EPISODE_START, env_id)

    def reset_envs_idx(
        self,
        env_ids: torch.Tensor | None = None,
        target_states: dict[str, Any] | None = None,
        target_buf: dict[str, Any] | None = None,
    ) -> None:
        """Reset selected environments, or all of them when ``env_ids`` is None, without advancing physics."""
        if env_ids is None:
            # Back-compat with reset_all(): in-progress episodes are recorded as zero-length, which
            # restarts episode-length statistics. Not recording them as finished would be more correct.
            self.episode_length_buf.zero_()
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = self._ensure_long_tensor(env_ids)
        if env_ids.numel() == 0:
            # Managers' reset hooks run every control frame, resets or not; there is no state to apply.
            self._reset_task_state(env_ids)
            self._reset_task_managers(env_ids)
            return

        # Initial reset starts an episode without inventing a preceding episode end.
        self._end_episodes(env_ids[self._episode_active[env_ids]])

        self._prepare_reset(env_ids)

        # Call the actual reset implementation (to be overridden by subclasses)
        self._reset_envs_idx_impl(env_ids, target_states, target_buf)

        # Reset all managers AFTER state changes
        # Advance the per-env episode counter BEFORE reset DR draws, so each env's k-th reset is
        # keyed by episode == k (independent of when, or alongside whom, it reset).
        self.episode_count[env_ids.to("cpu")] += 1
        self._reset_task_managers(env_ids)

        # Call manager-based reset events
        self.reset_manager.reset_scene(env_ids)

        self._refresh_envs_after_reset(env_ids)
        self._after_reset_synchronization(env_ids)

        self._start_episodes(env_ids)

    def _reset_envs_idx_impl(
        self,
        env_ids: torch.Tensor,
        target_states: dict[str, Any] | None = None,
        target_buf: dict[str, Any] | None = None,
    ) -> None:
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
        self._reset_objects_callback(env_ids)
        self._reset_task_state(env_ids)

    def render(self, sync_frame_time: bool = True) -> None:
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions

    @property
    def domain_rand_cfg(self) -> Any:
        """Return the active domain randomization configuration."""
        return self._manager_domain_rand_cfg

    ###########################################################################
    def _load_assets(self) -> None:
        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
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

    def _create_envs(self) -> None:
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        self.simulator.create_envs(self.num_envs, self._get_env_origins(), self.base_init_state)  # type: ignore[call-arg]

    def _setup_robot_body_indices(self) -> None:
        """Hook for subclasses to prepare body index caches (default no-op)."""

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _validate_task(self) -> None:
        """Validate subclass-specific requirements after the simulator is constructed."""

    def _initialize_task_managers(self) -> None:
        """Construct managers owned by the task specialization."""

    def _initialize_post_randomization_task_managers(self) -> None:
        """Finish task-manager construction after randomization is available."""

    def _prepare_manager_fields(self) -> None:
        """Prepare simulator fields required by BaseTask-owned managers."""
        self.simulator.prepare_manager_fields(randomization_manager=self.randomization_manager)

    def _setup_task_managers(self) -> None:
        """Set up the task's managers; specializations extend it to set up their own."""
        self.randomization_manager.setup()
        self.terrain_manager.setup()

    def _get_task_name(self) -> str:
        """Return a task identifier for logging/experiment directory naming."""
        training_task_name = getattr(self.training_config, "task_name", None)
        if isinstance(training_task_name, str) and training_task_name:
            return training_task_name
        return self.__class__.__name__.lower()

    def _get_env_origins(self) -> torch.Tensor:
        """Return environment origins used when creating simulator environments."""
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state is None or not hasattr(terrain_state, "env_origins"):  # type: ignore[redundant-expr]
            raise RuntimeError("Terrain manager state 'locomotion_terrain' must provide env_origins.")
        return terrain_state.env_origins

    # ------------------------------------------------------------------
    # Reset hooks

    def _prepare_reset(self, env_ids: torch.Tensor) -> None:
        """Prepare task-specific state before reset callbacks mutate simulator state."""

    def _reset_buffers_callback(self, env_ids: torch.Tensor, target_buf: dict[str, Any] | None = None) -> None:
        """Reset environment-specific buffers prior to manager resets.

        The default clears BaseTask-owned episode state. Subclasses may extend this to
        zero custom tensors or restore from a buffered state.
        """
        self.episode_length_buf[env_ids] = 0

    def _reset_tasks_callback(self, env_ids: torch.Tensor) -> None:
        """Hook for subclasses to extend reset-time logic."""

    def _reset_robot_states_callback(self, env_ids: torch.Tensor, target_states: dict[str, Any] | None = None) -> None:
        """Reset simulator DOF/root states for the specified environments.

        The default restores the state captured after simulator preparation, or the
        ``root_states``/``dof_states`` in ``target_states`` when given, one row per entry of
        ``env_ids``. Subclasses may override this to add task-specific state sampling.
        """
        if target_states is None:
            root_states = self._initial_robot_root_states[env_ids]
            dof_states = self._initial_robot_dof_states[env_ids]
        else:
            root_states = target_states["root_states"]
            dof_states = target_states["dof_states"]
        self.simulator.robot_root_states[env_ids] = root_states
        self.simulator.dof_pos[env_ids] = dof_states[..., 0]
        self.simulator.dof_vel[env_ids] = dof_states[..., 1]

    def _reset_objects_callback(self, env_ids: torch.Tensor) -> None:
        """Reset registered rigid objects to their initial pose and velocity.

        Default implementation iterates the simulator's registered rigid objects and
        writes each back to its configured initial pose (env_origins already applied) and
        initial velocity, via the unified actor API, so objects do not accumulate drift
        across episodes.

        A subclass that manages objects itself (e.g. an object-tracking task that resets
        objects from motion data) may override this. No-op for robot-only scenes.
        """
        object_names = self.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
        if not object_names:
            return

        init_poses = self.simulator.get_actor_initial_poses(object_names, env_ids)  # [n*len(env_ids), 7]
        init_vels = self.simulator.get_actor_initial_velocities(object_names, env_ids)  # type: ignore[attr-defined]  # [n*len(env_ids), 6]
        states = torch.zeros(init_poses.shape[0], 13, device=self.device)
        states[:, :7] = init_poses  # pose
        states[:, 7:] = init_vels  # configured initial velocity
        self.simulator.set_actor_states(object_names, env_ids, states)

    def _reset_task_state(self, env_ids: torch.Tensor) -> None:
        """Reset task-specific state after simulator state callbacks."""

    def _reset_task_managers(self, env_ids: torch.Tensor) -> None:
        """Reset the task's managers; specializations extend it to reset their own."""
        self.randomization_manager.reset(env_ids)

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        """Update task state after reset writes have been synchronized."""

    ###########################################################################
    # Simulation loop helpers

    def step(self) -> Any:
        """Advance one control frame."""
        self._physics_step()
        return self._post_physics_step()

    def _physics_step(self) -> None:
        decimation = self.config.simulator.config.sim.control_decimation_steps
        self.hooks.emit(Phase.FRAME_BEGIN)
        for substep in range(decimation):
            self._apply_force_in_physics_step()
            self.hooks.emit(Phase.PRE_STEP)
            self.simulator.simulate_at_each_physics_step()
            # Offscreen outputs are read at FRAME_END, so they draw on the frame's last substep.
            self._rendered_this_substep = substep == decimation - 1 and self.simulator.requires_step_render()
            if self._rendered_this_substep:
                self.simulator.render()
            self.hooks.emit(Phase.POST_STEP)

    def _apply_force_in_physics_step(self) -> None:
        """Apply task-specific controls during each physics substep."""

    def _post_physics_step(self) -> Any:
        # Cameras render and egress consumers publish here, as FRAME_END plugins (the cameras'
        # render_sensors is registered before any consumer, so buffers are fresh on read).
        self.hooks.emit(Phase.FRAME_END)
        self.episode_length_buf += 1
        self._after_physics_step()

        env_ids = self._get_control_frame_reset_env_ids()
        reset_state = self._before_control_frame_reset(env_ids)
        self.reset_envs_idx(env_ids)

        self._step_managers()

        return self._after_control_frame(env_ids, reset_state)

    def _after_physics_step(self) -> None:
        """Update task-specific state after the frame's substeps and FRAME_END."""

    def _step_managers(self) -> None:
        """Step managers once per control frame, after resets."""
        self.randomization_manager.step()

    def _get_control_frame_reset_env_ids(self) -> torch.Tensor:
        """Return environments whose active episodes should end this control frame."""
        return torch.empty(0, device=self.device, dtype=torch.long)

    def _before_control_frame_reset(self, env_ids: torch.Tensor) -> Any:
        """Capture task state immediately before selected environments reset."""
        return None

    def _after_control_frame(self, env_ids: torch.Tensor, reset_state: Any) -> Any:
        """Finish task-specific work after control-frame resets."""
        return None

    def _ensure_long_tensor(self, tensor_like: Any) -> torch.Tensor:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(tensor_like, device=self.device, dtype=torch.long)

    def _refresh_envs_after_reset(self, env_ids: torch.Tensor) -> None:
        """Apply reset and randomization state writes before observations read them."""
        self.simulator.forward_kinematics()

    def _refresh_sim_tensors(self) -> None:
        """Deprecated no-op: simulator state tensors are current at every read."""
        _warn_deprecated("_refresh_sim_tensors", "drop the call; state is current at every read")
