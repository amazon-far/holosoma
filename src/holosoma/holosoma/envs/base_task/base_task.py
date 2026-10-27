from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.envs.base_task.hooks import HookRegistry, Phase, frame_base_rates
from holosoma.managers.randomization import RandomizationManager
from holosoma.managers.reset_events.manager import ResetEventManager
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
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
        self._shutdown_requested = False
        try:
            self._initialize_base_task(tyro_config)
        except BaseException:
            try:
                self._close_plugins_and_simulator()
            except Exception:
                logger.exception("Task cleanup failed while preserving the task setup error")
            raise

    def run(self) -> None:
        """Run this task's outer execution loop."""
        raise NotImplementedError(f"{type(self).__name__} must implement run().")

    @property
    def shutdown_requested(self) -> bool:
        """Whether something has asked the task's outer run loop to stop."""
        return self._shutdown_requested

    def request_shutdown(self, reason: str = "") -> None:
        """Ask the task's outer run loop to stop after the current control frame."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info(f"Task shutdown requested{f': {reason}' if reason else ''}")

    def close(self) -> None:
        """End active episodes, then close plugins and the simulator exactly once."""
        if self._closed:
            return
        self._closed = True

        primary_error: BaseException | None = None
        try:
            self._end_episodes(self._episode_active.nonzero(as_tuple=False).flatten())
        except BaseException as exc:
            primary_error = exc

        try:
            self._close_plugins_and_simulator()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                logger.opt(exception=exc).error("Plugin or simulator cleanup also failed")

        if primary_error is not None:
            raise primary_error

    def _close_plugins_and_simulator(self) -> None:
        """Emit ``CLOSE`` to plugins, then close the simulator they read from.

        A plugin teardown failure leaves the simulator open, so a consumer that could not stop its
        workers never outlives the resources they use.
        """
        hooks = getattr(self, "hooks", None)
        if hooks is not None:
            hooks.emit(Phase.CLOSE)
        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            simulator.close()

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
            training=training_config,
            logger=tyro_config.logger,
            experiment_dir=str(experiment_dir),
        )

        self.num_envs = training_config.num_envs
        self.device = resolve_simulation_device(tyro_config, get_local_rank())

        # Domain-randomization keying state (read by the randomization manager to bind TermSamplers).
        # base_seed makes every DR draw value = f(seed, term, env, episode); the per-env episode
        # counter (incremented on reset, BEFORE the reset DR runs) keys async per-env resets. Set here
        # (before any manager is constructed) so setup-stage DR already sees them.
        self.dr_base_seed = training_config.seed
        self.dr_episode_count = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")

        sim_config = simulator_config.config.sim
        self.hooks = HookRegistry(
            base_rates=frame_base_rates(float(sim_config.fps), sim_config.control_decimation_steps)
        )
        self.terrain_manager = TerrainManager(terrain_config, self, self.device)
        self.simulator: BaseSimulator = SimulatorClass(
            tyro_config=full_sim_config, terrain_manager=self.terrain_manager, device=self.device
        )

        self._validate_task()
        self.headless = training_config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.sim_dt = self.simulator.sim_dt

        self.dt = simulator_config.config.sim.control_decimation_steps * self.sim_dt
        time_scale = tyro_config.time_scale
        self._rate_limiter = RateLimiter(time_scale / self.sim_dt) if time_scale > 0 else None
        self._next_viewer_time = 0.0
        self._physics_step_count = 0

        self.simulator.setup_terrain()
        # create envs, sim and viewer
        self._load_assets()
        self._validate_loaded_assets()

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
            self.debug_viz = False
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer

        # Randomization is built between the task's managers because some terms bind to one another
        # during construction.
        self._initialize_task_managers()
        self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
        self._initialize_post_randomization_task_managers()

        # Episode state survives _init_buffers(), which some tasks re-run on reset_all().
        self._episode_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._init_buffers()

        # Prepare fields required by managers (BEFORE setup calls)
        # This scans decorator metadata and expands model fields for per-environment randomization
        self._prepare_manager_fields()

        # Call setup for managers that need it
        self.randomization_manager.setup()
        self._setup_task_managers()
        self.terrain_manager.setup()

        # Initialize reset manager from simulator config
        self.reset_manager = ResetEventManager(
            self.simulator.simulator_config.reset_manager, self.simulator, self.device
        )

        if not self.headless:
            self.viewer = self.simulator.viewer

        self._install_plugins()

    def _install_plugins(self) -> None:
        """Construct each configured plugin as ``cls(cfg, task)`` once the task is fully built."""
        self.installed_plugins: dict[str, Any] = {}
        for key, plugin_config in self.config.plugin.items():
            self.installed_plugins[key] = plugin_config.get_cls()(plugin_config, self)
        if self.installed_plugins:
            logger.info(f"Installed plugins: {sorted(self.installed_plugins)}")

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _refresh_sim_tensors(self) -> None:
        self.simulator.refresh_sim_tensors()

    def reset_all(self) -> Any:
        """Reset every environment without advancing physics."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(env_ids)
        return self._finish_reset()

    def _finish_reset(self) -> Any:
        """Finish reset work owned by a task specialization."""
        return None

    def _end_episodes(self, env_ids: torch.Tensor) -> None:
        self._episode_active[env_ids] = False
        for env_id in env_ids.tolist():
            self.simulator.on_episode_end(env_id)
            self.hooks.emit(Phase.EPISODE_END, env_id)

    def _start_episodes(self, env_ids: torch.Tensor) -> None:
        self._episode_active[env_ids] = True
        for env_id in env_ids.tolist():
            self.simulator.on_episode_start(env_id)
            self.hooks.emit(Phase.EPISODE_START, env_id)

    def reset_envs_idx(
        self,
        env_ids: torch.Tensor,
        target_states: dict[str, Any] | None = None,
        target_buf: dict[str, Any] | None = None,
    ) -> None:
        """Reset selected environments without advancing physics."""
        env_ids = self._ensure_long_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # Initial reset starts an episode without inventing a preceding episode end.
        self._end_episodes(env_ids[self._episode_active[env_ids]])

        self._prepare_reset(env_ids)

        # Call the actual reset implementation (to be overridden by subclasses)
        self._reset_envs_idx_impl(env_ids, target_states, target_buf)

        # Reset all managers AFTER state changes
        # Advance the per-env episode counter BEFORE reset DR draws, so each env's k-th reset is
        # keyed by episode == k (independent of when, or alongside whom, it reset).
        self.dr_episode_count[env_ids.to("cpu")] += 1
        self.randomization_manager.reset(env_ids)
        self._reset_task_managers(env_ids)

        # Call manager-based reset events
        self.reset_manager.reset_scene(env_ids)

        task_refresh_env_ids = self._ensure_long_tensor(self._get_envs_to_refresh())
        self._refresh_envs_after_reset(torch.unique(torch.cat((env_ids, task_refresh_env_ids))))
        self.simulator.synchronize_after_reset()
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

    def _validate_loaded_assets(self) -> None:
        """Validate task-specific requirements after simulator assets load."""

    def _initialize_task_managers(self) -> None:
        """Construct managers owned by the task specialization."""

    def _initialize_post_randomization_task_managers(self) -> None:
        """Finish task-manager construction after randomization is available."""

    def _prepare_manager_fields(self) -> None:
        """Prepare simulator fields required by BaseTask-owned managers."""
        self.simulator.prepare_manager_fields(randomization_manager=self.randomization_manager)

    def _setup_task_managers(self) -> None:
        """Set up managers owned by the task specialization."""

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
        self.simulator.set_robot_state(env_ids, root_states=root_states, dof_states=dof_states)

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
        """Reset managers owned by the task specialization."""

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        """Update task state after reset writes have been synchronized."""

    ###########################################################################
    # Simulation loop helpers

    def step(self, actor_state: dict[str, Any] | None = None) -> Any:
        """Advance one control frame using an extensible per-step state dictionary."""
        if actor_state is None:
            actor_state = {}
        self._pre_physics_step(actor_state)
        self._physics_step()
        return self._post_physics_step()

    def _pre_physics_step(self, actor_state: dict[str, Any]) -> None:
        """Process task-specific state before advancing physics."""

    def _physics_step(self) -> None:
        sim_config = self.config.simulator.config.sim
        viewer_dt = self.config.viewer_dt

        for substep in range(sim_config.control_decimation_steps):
            physics_time = self._physics_step_count * self.sim_dt
            next_viewer_time = self._next_viewer_time
            viewer_due = physics_time + 1e-12 >= next_viewer_time
            rendered_this_step = False

            self.simulator.sync_sim_state()
            if substep == 0:
                self.simulator.begin_frame()
                self.hooks.emit(Phase.FRAME_BEGIN)
            self._apply_force_in_physics_step()
            self.simulator.before_substep()
            self.hooks.emit(Phase.PRE_STEP)
            self.simulator.simulate_at_each_physics_step()
            # Count the step just taken, so with render_interval == decimation the render lands on
            # the frame's last substep and FRAME_END consumers see the image for the published state.
            render_due = (self._physics_step_count + 1) % sim_config.render_interval_steps == 0
            if self.simulator.requires_step_render() and render_due:
                self.simulator.render()
                rendered_this_step = True
            self.simulator.after_substep()
            self.hooks.emit(Phase.POST_STEP)

            if viewer_due:
                if self.viewer is not None and not rendered_this_step:
                    self.simulator.sync_sim_state()
                    self.render()
                elapsed_intervals = int((physics_time - next_viewer_time) / viewer_dt) + 1
                self._next_viewer_time = next_viewer_time + elapsed_intervals * viewer_dt

            self._physics_step_count += 1
            if self._rate_limiter is not None:
                self._rate_limiter.sleep()

    def _apply_force_in_physics_step(self) -> None:
        """Apply task-specific controls during each physics substep."""

    def _post_physics_step(self) -> Any:
        self._refresh_sim_tensors()
        # Cameras render before FRAME_END, so egress plugins read fresh buffers.
        self.simulator.end_frame()
        self.hooks.emit(Phase.FRAME_END)
        self.episode_length_buf += 1
        self._after_physics_step()

        env_ids = self._get_control_frame_reset_env_ids()
        reset_state = self._before_control_frame_reset(env_ids)
        self.reset_envs_idx(env_ids)

        # Managers step AFTER reset so that WBT termination checks see the actual tracking
        # error before the command manager advances the motion clip.
        self._step_managers()

        return self._after_control_frame(env_ids, reset_state)

    def _after_physics_step(self) -> None:
        """Update task-specific state after simulator tensors are refreshed."""

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

    def _get_envs_to_refresh(self) -> torch.Tensor:
        return torch.empty(0, device=self.device, dtype=torch.long)

    def _refresh_envs_after_reset(self, env_ids: torch.Tensor) -> None:
        """Flush reset and randomization state through backend-neutral robot writers."""
        self.simulator.set_dof_state_tensor_robots(env_ids)
        self.simulator.set_actor_root_state_tensor_robots(env_ids)
