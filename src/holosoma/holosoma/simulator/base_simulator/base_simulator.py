from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.simulator import SimulatorInitConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.object_registry import ObjectRegistry, ObjectType
from holosoma.simulator.shared.scene_types import SceneInterface
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.experiment_paths import get_video_dir
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

if TYPE_CHECKING:
    from holosoma.simulator.shared.camera_controller import CameraController
    from holosoma.simulator.shared.camera_sensor import SensorManager
    from holosoma.simulator.shared.simulator_bridge import SimulatorBridge
    from holosoma.simulator.shared.video_recorder import VideoRecorderInterface
    from holosoma.simulator.shared.virtual_gantry import VirtualGantry


class BaseSimulator:
    """Base class for robotic simulation environments.

    Provides a unified framework for simulation setup, environment creation, and control
    over robotic assets and simulation properties. Includes unified state accessors
    for cross-simulator compatibility.

    The unified interface abstracts differences between IsaacGym's flat tensor approach
    and IsaacSim's object-based tensors, providing consistent APIs for object state
    management across both simulators.

    Actor Concept
    -------------
    An "actor" in this simulation framework refers to any named physical entity
    that can be manipulated in the simulation. This includes:

    - **Robots**: Articulated bodies with controllable joints (e.g., "humanoid_robot")
    - **Objects**: Rigid bodies like tools, furniture, or obstacles (e.g., "hammer_tool", "workbench_table")
    - **Scene Elements**: Environmental objects loaded from scene files (e.g., "wall_001", "floor_plane")

    All actors have:
    - Unique string identifiers for name-based access
    - 6DOF pose (position + orientation) relative to their asset-defined origin
    - Physical properties (mass, collision geometry, material properties)
    - State information including velocities and accelerations

    **Important**: The pose reference point (origin) is **asset-dependent** and defined
    by the asset file (URDF, USD, etc.), not by the simulator framework. For example:
    - Robot origins might be at the base link center, pelvis, or foot contact point
    - Object origins could be at geometric center, bottom corner, or functional point

    Note that different simulation frameworks use varying terminology for actors e.g., objects,
    models, bodies, entities.

    This abstraction allows the same API to work with any physical entity in the simulation,
    whether it's a complex robot or a simple rigid body object, while respecting each
    asset's coordinate frame conventions.

    Important Implementation Notes
    ------------------------------
    **Actor Index Ordering**: Actor indices returned by get_actor_indices() may have
    different ordering between simulator implementations. Always get indices dynamically
    to maintain maximum simulator-agnostic code. Never assume or hardcode specific
    index values in task/environment code.

    **State Synchronization**:
    - IsaacGym: State changes applied immediately
    - IsaacSim: State changes can be deferred for batching (see write_state_updates)

    Examples
    --------
    **Unified Interface (Recommended)**:
    >>> # Works identically across both simulators
    >>> states = sim.get_actor_states(["obj0", "table"], env_ids)
    >>> sim.set_actor_states(["obj0", "table"], env_ids, new_states)

    **Direct Tensor-Like Access (Advanced)**:
    >>> # Always get indices dynamically on or after setup - don't hardcode values.
    >>> indices = sim.get_actor_indices(["robot", "obj0", "obj1"], env_ids)
    >>>
    >>> states = sim.all_root_states[indices]          # gather [len(indices), 13]
    >>> sim.all_root_states[indices] = new_states      # scatter
    >>> sim.write_state_updates()                      # Sync changes

    ``all_root_states`` is a duck-typed proxy (NOT a ``torch.Tensor``) spanning all actors
    (robot, scene, and individual objects), routing ``[indices]`` reads/writes through
    ``get/set_actor_states_by_index``. Supported surface: ``[indices]`` and
    ``[indices, column_slice]`` read/write, ``.shape``, ``.device``, ``.dtype``, ``.clone()``.
    Tensor-only methods (``.view``/``.reshape``/``.to``) and ``isinstance(_, torch.Tensor)`` are
    not supported.

    Passing it wholesale to ``set_actor_root_state_tensor`` writes the robot only (that method
    detects the proxy by identity). Use ``set_actor_states`` to write objects.

    **Anti-Pattern (Don't Do This)**:
    >>> # Never assume specific index values
    >>> indices = torch.tensor([0, 1, 2])  # WRONG - hardcoded indices
    >>> # Always get indices dynamically to remain simulator-agnostic
    >>> indices = sim.get_actor_indices(["robot", "obj0"], env_ids)

    **Batched Updates (Performance)**:
    >>> # Batch multiple updates for efficiency in IsaacSim
    >>> sim.set_actor_states(["obj0"], env_ids, states1, write_updates=False)
    >>> sim.set_actor_states(["obj1"], env_ids, states2, write_updates=False)
    >>> sim.write_state_updates()  # Single batch sync
    """

    training_config: TrainingConfig
    simulator_config: SimulatorInitConfig
    scene_config: SceneConfig
    robot_config: RobotConfig

    sim_dt: float
    viewer: object
    robot_root_states: torch.Tensor
    base_quat: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    contact_forces: torch.Tensor
    contact_forces_history: torch.Tensor

    # Robot properties, populated by each backend during setup (see _setup_robot_props_*).
    num_dof: int
    num_bodies: int
    dof_names: list[str]
    body_names: list[str]
    scene: SceneInterface
    # Unified all-actors (robot + objects) root-state view: a duck-typed UnifiedRootStatesView,
    # not a torch.Tensor. Index with get_actor_indices results for any actor; supports [indices]
    # read/write, .shape/.device/.dtype/.clone(), not Tensor-only methods (.view/.reshape/.to).
    all_root_states: torch.Tensor  # type: ignore[assignment]

    height_samples: None | torch.Tensor

    num_envs: int

    debug_viz_enabled: bool = False

    _rigid_body_pos: torch.Tensor
    _rigid_body_rot: torch.Tensor
    _rigid_body_vel: torch.Tensor
    _rigid_body_ang_vel: torch.Tensor

    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str):
        """
        Initializes the base simulator with configuration settings and simulation device.

        Parameters
        ----------
        tyro_config: FullSimConfig
            Tyro configuration for the simulation.
        terrain_manager: TerrainManager
            Terrain manager for the simulation.
        device: str
            Device type for simulation ('cpu' or 'cuda').

        """
        self.terrain_manager = terrain_manager
        self.training_config = tyro_config.training
        self.simulator_config = tyro_config.simulator
        self.scene_config = tyro_config.scene
        self.sensor_config = tyro_config.sensors
        self.robot_config = tyro_config.robot
        self.video_config = tyro_config.logger.video
        self.sim_device = device
        self.headless = False
        self.debug_viz_enabled = self.simulator_config.debug_viz
        self.object_registry = ObjectRegistry(device)
        self.hooks = HookRegistry(base_rates=self._hook_base_rates())
        # Register camera rendering FIRST (before the plugins below), so on FRAME_END cameras render
        # before any camera-consumer plugin (egress/viz/video) reads. No-ops until a backend builds
        # sensor_manager during setup; harmless when no cameras are configured.
        self.hooks.add(Phase.FRAME_END, self.render_sensors, name="sensors.render")
        # Build plugins from tyro_config.plugin and keep the instances alive (key -> plugin).
        # Constructing each here registers its hooks on self.hooks, so they fire on later
        # emit(). The `none` preset disables a slot.
        self.installed_plugins: dict[str, Any] = {
            key: cfg.get_cls()(cfg, self) for key, cfg in tyro_config.plugin.items()
        }
        if self.installed_plugins:
            logger.info(f"Installed plugins: {sorted(self.installed_plugins)}")
        self._closed = False

        # Virtual gantry system
        self.virtual_gantry: VirtualGantry | None = None

        # Camera controller for viewer tracking
        self.camera_controller: CameraController | None = None

        # Video recording system
        self.video_recorder: VideoRecorderInterface | None = None

        # Bridge system
        self.bridge: SimulatorBridge | None = None

        # Mounted-camera sensors, populated by each backend during its own sensor setup (None until
        # then, and when no cameras are configured). The render_sensors hook above no-ops until it exists.
        self.sensor_manager: SensorManager | None = None

        # To be overridden by subclasses
        self.height_samples = None

        if tyro_config.experiment_dir is not None and self.video_config.save_dir is None:
            self.video_config = dataclasses.replace(
                self.video_config, save_dir=str(get_video_dir(Path(tyro_config.experiment_dir)))
            )

        self.logger_cfg = tyro_config.logger

        # force video recording when headless_recording=true for backwards compatibility
        video_recording_enabled = self.logger_cfg.headless_recording or self.video_config.enabled
        self.video_config = dataclasses.replace(self.video_config, enabled=video_recording_enabled)

    # ----- Configuration Setup Methods -----

    def set_headless(self, headless):
        """
        Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without graphical display.
        """
        self.headless = headless

    def set_startup_randomization_callback(self, callback):
        """Sets a callback to be invoked during environment startup for domain randomization.

        This method allows the environment to inject domain randomization logic during
        environment creation. The callback behavior and timing may vary by simulator.

        Note: This is currently only used by IsaacGym. Other simulators may implement
        this as a no-op if they don't support startup randomization callbacks.

        Args:
            callback: A callable to be invoked during environment startup.
        """
        # Default no-op implementation

    def setup(self):
        """
        Initializes the simulator parameters and environment. This method should be implemented
        by subclasses to set specific simulator configurations.
        """
        raise NotImplementedError("The 'setup' method must be implemented in subclasses.")

    def prepare_manager_fields(self, **managers) -> None:
        """Scan managers for field requirements and prepare them.

        Parameters
        ----------
        **managers : Any
            Manager instances to scan for field requirements.
        """
        # default is no-op

    # ----- Terrain Setup Methods -----

    def setup_terrain(self):
        """
        Configures the terrain based on specified mesh type.

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'trimesh').
        """
        raise NotImplementedError("The 'setup_terrain' method must be implemented in subclasses.")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self):
        """
        Loads the robot assets into the simulation environment.
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names
        """
        raise NotImplementedError("The 'load_assets' method must be implemented in subclasses.")

    def get_supported_scene_formats(self) -> list[str]:
        """Returns the scene formats supported by this simulator in order of preference.

        This method should be implemented by subclasses to specify which scene file
        formats they can load and in what order of preference they should be tried.

        Returns
        -------
        list[str]
            List of supported scene formats in preference order (e.g., ['urdf', 'usd'])

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError(
            f"Simulator {type(self).__name__} must implement get_supported_scene_formats() method"
        )

    # ----- Environment Creation Methods -----

    def create_envs(self, num_envs, env_origins, base_init_state, env_config):
        """
        Creates and initializes environments with specified configurations.

        Args:
            num_envs (int): Number of environments to create.
            env_origins (list): List of origin positions for each environment.
            base_init_state (array): Initial state of the base.
            env_config (dict): Configuration for each environment.
        """
        raise NotImplementedError("The 'create_envs' method must be implemented in subclasses.")

    # ----- Scene-asset registration (shared across backends) -----

    def _register_scene_assets(self) -> None:
        """Register the robot and every scene asset in the ObjectRegistry.

        Shared across backends: each supplies its spawned data via ``_collect_spawned_actors``;
        this sorts the actors by name, prepends the robot, splits the rest into SCENE (static) /
        INDIVIDUAL (free) bands (each numbered from 0), sizes the registry ranges from the
        per-type counts, and registers each entry.
        """
        robot_pose, items = self._collect_spawned_actors()

        # Sort scene actors by name so the registry slot (position_in_type) each one lands
        # at is backend-independent. Robot stays first (it's prepended after the sort and
        # is not part of `items`).
        items = sorted(items, key=lambda item: item[0])

        # The robot's name/type/position are fixed and it carries no scene-configured velocity,
        # so it's the same constant entry for every backend (only its pose differs). The rest
        # split into the SCENE (static) / INDIVIDUAL (free) bands; pose/velocity pass through.
        entries = [("robot", ObjectType.ROBOT, 0, robot_pose, None)]
        free_pos = scene_pos = 0
        for name, is_static, pose, velocity in items:
            if is_static:
                entries.append((name, ObjectType.SCENE, scene_pos, pose, velocity))
                scene_pos += 1
            else:
                entries.append((name, ObjectType.INDIVIDUAL, free_pos, pose, velocity))
                free_pos += 1

        self.object_registry.setup_ranges(
            self.num_envs,
            robot_count=1,
            scene_count=scene_pos,
            individual_count=free_pos,
        )

        for name, object_type, position, pose, velocity in entries:
            self.object_registry.register_object(name, object_type, position, pose, velocity)
        self.object_registry.finalize_registration()

    def _collect_spawned_actors(self):
        """Return ``(robot_pose, items)`` describing what this backend spawned.

        - ``robot_pose``: the robot's ``[num_envs, 7]`` initial pose, in the backend's own
          world/local convention.
        - ``items``: one ``(name, is_static, pose, velocity)`` per spawned scene actor, free
          and static bodies alike, including the 1->N bodies a scene file expands into. ``pose``
          is ``[num_envs, 7]``; ``velocity`` is ``[num_envs, 6]`` or ``None`` for zero. Order is
          irrelevant (``_register_scene_assets`` sorts by name), so collect them however is
          natural for the backend.

        ``_register_scene_assets`` turns this into the registry: it sorts, prepends the robot, and
        classifies ``items`` into type bands identically for every backend, so the subclass only
        describes its actors, not how they map to the registry.
        """
        raise NotImplementedError("Backends with scene assets must implement _collect_spawned_actors().")

    # ----- Property Retrieval Methods -----

    def get_simulator_type(self) -> SimulatorType:
        """Get the simulator type for this simulator instance.

        Returns:
            SimulatorType: The type of simulator (ISAACGYM, ISAACSIM, MUJOCO)
        """
        return get_simulator_type()

    def get_dof_limits_properties(self):
        """
        Retrieves the DOF (degrees of freedom) limits and properties.

        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        raise NotImplementedError("The 'get_dof_limits_properties' method must be implemented in subclasses.")

    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        raise NotImplementedError("The 'find_rigid_body_indice' method must be implemented in subclasses.")

    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        Prepares the simulation environment and refreshes any relevant tensors.
        """
        raise NotImplementedError("The 'prepare_sim' method must be implemented in subclasses.")

    def refresh_sim_tensors(self):
        """
        Refreshes the state tensors in the simulation to ensure they are up-to-date.
        """
        raise NotImplementedError("The 'refresh_sim_tensors' method must be implemented in subclasses.")

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        """Clear the contact-forces history for the specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            1-D tensor of environment IDs whose contact-force history is zeroed (empty -> no-op).
        """
        raise NotImplementedError("The 'clear_contact_forces_history' method must be implemented in subclasses.")

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        Applies the specified torques to the robot's degrees of freedom (DOF).

        Args:
            torques (tensor): Tensor containing torques to apply.
        """
        raise NotImplementedError("The 'apply_torques_at_dof' method must be implemented in subclasses.")

    def simulate_at_each_physics_step(self):
        """
        Advances the simulation by a single physics step.
        """
        raise NotImplementedError("The 'simulate_at_each_physics_step' method must be implemented in subclasses.")

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        raise NotImplementedError("The 'setup_viewer' method must be implemented in subclasses.")

    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        raise NotImplementedError("The 'render' method must be implemented in subclasses.")

    def time(self) -> float:
        """Get current simulation time in seconds.

        This method is used for clock synchronization between the simulator and
        policy inference in sim2sim setups. The time value is published via ZMQ
        to allow policies (especially WBT policies) to advance motion timesteps
        in sync with the simulation.

        Returns
        -------
        float
            Current simulation time in seconds.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.

        See Also
        --------
        SimulatorBridge.step : Publishes this time via ClockPub
        """
        raise NotImplementedError("The 'time' method must be implemented in subclasses.")

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        """Get DOF forces for a specific environment (simulator-agnostic interface).

        This method provides a unified interface for accessing joint forces across
        all simulators. Used by the bridge system for sim2sim force feedback.

        Parameters
        ----------
        env_id : int, default=0
            Environment index to query forces for.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_dof] with measured joint forces, dtype torch.float32.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        RuntimeError
            If DOF force sensors are not enabled or forces not available.

        See Also
        --------
        BasicSdk2Bridge._get_actuator_forces : Uses this method for bridge force feedback
        """
        raise NotImplementedError("The 'get_dof_forces' method must be implemented in subclasses.")

    def draw_debug_viz(self):
        pass

    # ----- Camera Controller Helper Methods -----

    def _init_camera_controller(self) -> None:
        """Initialize shared camera controller for viewer tracking.

        Should be called by subclasses after robot assets are loaded and before
        viewer setup. Creates a CameraController instance configured for the
        interactive viewer.

        The camera controller provides unified camera positioning logic that:
        - Supports multiple camera modes (Fixed, Spherical, Cartesian)
        - Handles robot tracking with configurable body attachment
        - Applies camera smoothing for stable viewing

        Raises
        ------
        ValueError
            If viewer.enabled=True but viewer.camera is None
        Exception
            If camera controller initialization fails
        """
        if self.headless or not self.simulator_config.viewer.enable_tracking:
            logger.debug("Skipping camera controller (headless or tracking disabled)")
            return

        if self.simulator_config.viewer.camera is None:
            raise ValueError("viewer.camera must be provided when viewer.enable_tracking=True")

        try:
            # Conditional import to avoid circular dependencies
            from holosoma.simulator.shared.camera_controller import CameraController

            self.camera_controller = CameraController(
                config=self.simulator_config.viewer.camera,
                simulator=self,
            )
            logger.info("Viewer camera controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize viewer camera controller: {e}")
            raise

    # ----- Bridge System Helper Methods -----

    def _init_bridge(self) -> None:
        """Initialize bridge system if enabled.

        Should be called by subclasses after robot assets are loaded
        and before simulation starts. Uses conditional import to avoid
        SDK dependencies when bridge is disabled.

        Raises
        ------
        Exception
            If bridge initialization fails
        """
        if not self.simulator_config.bridge.enabled:
            logger.info("Robot bridge disabled")
            return

        try:
            # Conditional import to avoid SDK dependencies
            from holosoma.simulator.shared.simulator_bridge import SimulatorBridge

            self.bridge = SimulatorBridge(self, self.simulator_config.bridge)
            self.bridge.register_hooks(self.hooks)
            logger.info("Bridge system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bridge system: {e}")
            raise

    def _hook_base_rates(self) -> dict[Phase, float]:
        """Base tick rate (Hz) of each periodic phase, so hooks can request a rate via ``every="30Hz"``.

        Per-substep phases tick at ``fps``; per-frame phases at the control rate ``fps / control_decimation``.
        """
        sim = self.simulator_config.sim
        fps = float(sim.fps)
        control_hz = fps / sim.control_decimation_steps
        return {
            Phase.PRE_STEP: fps,
            Phase.POST_STEP: fps,
            Phase.FRAME_BEGIN: control_hz,
            Phase.FRAME_END: control_hz,
        }

    def close(self) -> None:
        """Tear down simulator-owned runtime participants."""
        if self._closed:
            return
        self._closed = True
        self.hooks.emit(Phase.CLOSE)

    def _stop_bridge(self) -> None:
        """Tear down the bridge if enabled (joins the multiprocess Unitree DDS child). Safe if unset."""
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

    # ----- Actor/Object Access Interface -----
    # These methods provide unified access to objects registered with ObjectType enum

    def write_state_updates(self) -> None:
        """Write any pending state updates to the simulation.

        Synchronizes deferred state changes to the underlying simulation backend.
        Only needed when using set_actor_states() with write_updates=False for
        batching multiple updates together.

        Usage Patterns
        --------------
        **Automatic sync (current default behavior):**
        >>> sim.set_actor_states(["obj0"], env_ids, states)  # Auto-synced

        **Manual batching for performance:**
        >>> sim.set_actor_states(["obj0"], env_ids, states1, write_updates=False)
        >>> sim.set_actor_states(["obj1"], env_ids, states2, write_updates=False)
        >>> sim.write_state_updates()  # Batch sync

        **Direct access via all_root_states (any actor, incl. "robot"):**
        >>> sim.all_root_states[indices] = new_states  # Direct modification
        >>> sim.write_state_updates()  # Sync changes

        Implementation Notes
        --------------------
        - **IsaacGym**: No-op, state changes are always applied immediately
        - **IsaacSim**: Flushes deferred updates via scene.write_data_to_sim()

        See Also
        --------
        set_actor_states : Method that can defer updates for batching

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    # Default methods - delegate to by_names
    def get_actor_states(self, names: ActorNames, env_ids: EnvIds) -> ActorStates:
        """Get actor states by name (default interface).

        This is the standard interface that maintains backward compatibility.
        Delegates to get_actor_states_by_names().

        Parameters
        ----------
        names : ActorNames
            Actor names to query (see ActorNames type)
        env_ids : EnvIds | None, default=None
            Environment IDs to query (see EnvIds type). If None, returns indices for all environments.

        Returns
        -------
        ActorStates
            Actor states for the specified actors and environments

        Examples
        --------
        >>> # Standard usage (backward compatible)
        >>> actor_names = ["obj0", "table_pantene", "robot"]
        >>> env_ids = torch.tensor([0, 1, 2], device=device)
        >>> states = sim.get_actor_states(actor_names, env_ids)
        >>> # Shape: [9, 13] for 3 objects x 3 environments

        See Also
        --------
        get_actor_states_by_names : Explicit names-based method
        get_actor_states_by_index : Performance-optimized indices method
        """
        return self.get_actor_states_by_names(names, env_ids)

    def set_actor_states(
        self, names: ActorNames, env_ids: EnvIds, states: ActorStates, write_updates: bool = True
    ) -> None:
        """Set actor states by name (default interface).

        This is the standard interface that maintains backward compatibility.
        Delegates to set_actor_states_by_names().

        Parameters
        ----------
        names : ActorNames
            Actor names to update (see ActorNames type)
        env_ids : EnvIds
            Environment IDs to update (see EnvIds type)
        states : ActorStates
            New actor states (see ActorStates type)
        write_updates : bool
            Whether to sync changes immediately

        Examples
        --------
        >>> # Standard usage (backward compatible)
        >>> sim.set_actor_states(["obj0", "table"], env_ids, new_states)

        See Also
        --------
        set_actor_states_by_names : Explicit names-based method
        set_actor_states_by_index : Performance-optimized indices method
        """
        return self.set_actor_states_by_names(names, env_ids, states, write_updates)

    def set_static_body_pose(self, names: ActorNames, env_ids: EnvIds | None, poses: ActorPoses) -> None:
        """Kinematically relocate static (SCENE) scene bodies at runtime.

        Convenience over :meth:`set_actor_states` for the common "move a static obstacle" case:
        it zero-pads the 7-wide pose to a full 13-wide state (static/kinematic bodies ignore the
        velocity columns, moving by pose target, not commanded velocity) and forwards. Works
        the same on every backend (MuJoCo via body_pos/body_quat + forward-kinematics; IsaacGym
        via the fixed-base root-state teleport; IsaacSim via the kinematic rigid-body write).

        Parameters
        ----------
        names : ActorNames
            SCENE (static) body names to relocate. (Free INDIVIDUAL bodies also accept this, but
            for those :meth:`set_actor_states` with a velocity is usually what you want.)
        env_ids : EnvIds or None
            Environments to update (``None`` = all envs, as in :meth:`get_actor_indices`).
        poses : ActorPoses
            ``[len(names) * len(env_ids), 7]`` = ``[x, y, z, qx, qy, qz, qw]`` (xyzw). FULL pose:
            position + orientation, in the **WORLD frame**, the single backend-independent
            convention shared by :meth:`set_actor_states`, :meth:`get_actor_states`, and
            :meth:`get_actor_initial_poses` on every backend.

        Notes
        -----
        A single large teleport imparts no momentum and can let a fast dynamic body tunnel
        through; increment the pose over several steps to sweep/push dynamics. Read the live
        moved pose back via :meth:`get_actor_states` (NOT ``get_actor_initial_poses``, which
        returns the spawn pose).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)
        if not names or len(env_ids) == 0:
            return
        zeros = torch.zeros(poses.shape[0], 6, device=poses.device, dtype=poses.dtype)
        self.set_actor_states(list(names), env_ids, torch.cat([poses, zeros], dim=1))

    # ----- Sensor (camera) interface -----

    def render_sensors(self) -> None:
        """Render all due cameras once, at control frequency, into their cached buffers.

        Called once per control step from the task loop (after sim tensors refresh, before
        observations compute), honoring each camera's ``update_decimation``. Consumers then do
        cheap cached reads via :meth:`get_camera_data`."""
        raise NotImplementedError("The 'render_sensors' method must be implemented in subclasses.")

    def get_camera_data(
        self,
        name: str,
        data_type: str = "rgb",
        env_ids: EnvIds | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Read a camera's most-recent output.

        Output format:

        - **Shape** ``[len(env_ids), H, W, C]``, environment axis FIRST.
        - **Layout** HWC, row-major: row 0 = TOP of the image, column 0 = LEFT.
        - **rgb**: ``torch.uint8`` in ``[0, 255]``, channel order **R, G, B** (C=3).
        - **depth**: ``torch.float32`` meters, distance-to-image-plane, ``+inf`` for no-hit, C=1.
        - **device**: ``device`` if given, else ``self.sim_device``.
        - **Optical frame**: camera looks down its local ``-Z``, ``+Y`` up; the
          mount offset is applied in the mount-body frame.

        Parameters
        ----------
        name : str
            Sensor name (the ``CameraSensorConfig.name`` key).
        data_type : str, default="rgb"
            Modality to read: ``"rgb"`` or ``"depth"``. Must be in the camera's ``data_types``.
        env_ids : EnvIds | None, default=None
            Absolute environment indices into ``[0, num_envs)`` to return (every camera renders
            all envs, so these index the full buffer); ``None`` returns all environments.
        device : torch.device | str | None, default=None
            Where to return the tensor. ``None`` keeps the sim device (no copy). Any other device
            (e.g. ``"cpu"``) returns a cached cross-device copy of the FULL buffer: the first caller
            per ``(data_type, device)`` this render pays the transfer, later callers reuse it (so
            several egress consumers reading the same camera on host cost one device->host sync, not
            one each). The returned tensor is **read-only** — do not mutate it in place; it is shared.

        Raises
        ------
        NotImplementedError
            If the backend has no camera support (no ``sensor_manager``).
        """
        if self.sensor_manager is None or not self.sensor_manager.has_camera(name):
            raise NotImplementedError(
                f"{type(self).__name__} has no camera '{name}'. Created cameras: {self.get_sensor_names()}."
            )
        runtime = self.sensor_manager.get(name)
        if data_type not in runtime.buffers:
            raise RuntimeError(
                f"Camera '{name}' has no '{data_type}' frame. Ensure '{data_type}' is in the camera's "
                f"data_types and render_sensors() has run. Buffered: {sorted(runtime.buffers)}."
            )
        buf = runtime.buffer_on(data_type, device)
        return buf if env_ids is None else buf[env_ids]

    def get_sensor_names(self) -> list[str]:
        """Names of all created sensors, or ``[]`` if none."""
        if self.sensor_manager is None:
            return []
        return self.sensor_manager.names

    def sensor_config_by_name(self, name: str):
        """The ``CameraSensorConfig`` for ``name`` from the active mounted-camera dict.

        Used by camera-consumer hooks (egress) to read a camera's intrinsics. Raises if unknown.
        """
        cam = self.sensor_config.get(name)
        if cam is None:
            raise KeyError(f"No camera named '{name}'. Defined cameras: {list(self.sensor_config)}.")
        return cam

    # Explicit names-based methods
    def get_actor_states_by_names(self, names: ActorNames, env_ids: EnvIds) -> ActorStates:
        """Get actor states by actor names.

        Parameters
        ----------
        names : ActorNames
            Actor names to query (see ActorNames type)
        env_ids : EnvIds
            Environment IDs to query (see EnvIds type)

        Returns
        -------
        ActorStates
            Actor states for the specified actors and environments

        Examples
        --------
        >>> # Explicit names-based access
        >>> states = sim.get_actor_states_by_names(["obj0", "table"], env_ids)
        >>> positions = states[:, :3]      # Extract positions
        >>> quaternions = states[:, 3:7]   # Extract orientations (xyzw)
        >>> velocities = states[:, 7:]     # Extract velocities

        Notes
        -----
        - Returns current simulation states, not initial/default poses
        - State ordering matches ObjectRegistry index calculation
        - Environment origins are already applied in returned positions
        - Unified 13-vector layout ``[pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]``. Both
          linear AND angular velocity are WORLD-frame and the quaternion is xyzw, for every
          backend. Backends whose native storage differs (e.g. MuJoCo's freejoint qvel keeps
          angular velocity body-local) convert at their own get/set boundary so callers see one
          consistent frame. ``set_actor_states`` accepts the same layout (inverse conversion).

        See Also
        --------
        get_actor_states_by_index : Performance-optimized version
        get_actor_initial_poses : Get default/initial poses for reset
        """
        indices: ActorIndices = self.get_actor_indices(names, env_ids)
        return self.get_actor_states_by_index(indices)

    def set_actor_states_by_names(
        self, names: ActorNames, env_ids: EnvIds, states: ActorStates, write_updates: bool = True
    ) -> None:
        """Set actor states by actor names.

        Parameters
        ----------
        names : ActorNames
            Actor names to update (see ActorNames type)
        env_ids : EnvIds
            Environment IDs to update (see EnvIds type)
        states : ActorStates
            New actor states (see ActorStates type)
        write_updates : bool
            Whether to sync changes immediately

        Examples
        --------
        >>> # Explicit names-based update
        >>> sim.set_actor_states_by_names(["obj0", "table"], env_ids, new_states)

        >>> # Reset objects to initial poses with zero velocity
        >>> objects = ["obj0", "table_pantene"]
        >>> env_ids = torch.tensor([0, 1, 2], device=device)
        >>> initial_poses = sim.get_actor_initial_poses(objects, env_ids)
        >>>
        >>> # Build full states: [num_objects * num_envs, 13]
        >>> new_states = torch.zeros(len(objects) * len(env_ids), 13, device=device)
        >>> new_states[:, :7] = initial_poses  # Position + quaternion
        >>> new_states[:, 7:] = 0.0           # Zero velocities
        >>>
        >>> sim.set_actor_states_by_names(objects, env_ids, new_states)

        Notes
        -----
        - Input poses are **WORLD-frame on every backend**: pass world coordinates directly, do
          NOT pre-add ``env_origins`` (and never branch on simulator type). A
          ``get_actor_initial_poses`` result feeds straight back in unchanged. Velocities are
          world-frame too (see :meth:`get_actor_states_by_names` for the 13-vector layout).
        - Need env-LOCAL coordinates? Convert at the call site (position columns only; quat and
          velocity are origin-invariant). For a name-major/env-minor tensor ``t`` over ``names`` x
          ``env_ids``: ``t[:, 0:3] -= self.env_origins[env_ids].repeat(len(names), 1)`` to go
          world->local, ``+=`` to go local->world. (No frame kwarg exists; poses are world-frame.)
        - IsaacGym: write_updates parameter has no effect (always immediate)
        - IsaacSim: write_updates=False allows batching for performance
        - State ordering matches ObjectRegistry index calculation

        See Also
        --------
        set_actor_states_by_index : Performance-optimized version
        write_state_updates : Manual sync method for batched updates
        """
        indices: ActorIndices = self.get_actor_indices(names, env_ids)
        return self.set_actor_states_by_index(indices, states, write_updates)

    # Explicit indices-based methods (performance optimized)
    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """Get actor states by pre-computed indices (performance optimized).

        Parameters
        ----------
        indices : ActorIndices
            Pre-computed actor indices (see ActorIndices type)

        Returns
        -------
        ActorStates
            Actor states matching the provided indices

        Examples
        --------
        >>> # Cache indices for repeated use
        >>> cached_indices = sim.get_actor_indices(["obj0", "table"], env_ids)
        >>>
        >>> # Fast repeated access
        >>> states = sim.get_actor_states_by_index(cached_indices)
        >>>
        >>> # Extract state components
        >>> positions = states[:, :3]      # Extract positions
        >>> quaternions = states[:, 3:7]   # Extract orientations (xyzw)
        >>> velocities = states[:, 7:]     # Extract velocities

        Notes
        -----
        - Performance-optimized method for hot paths
        - No name lookup overhead
        - Indices must be pre-computed using get_actor_indices()

        See Also
        --------
        get_actor_indices : Get indices from names
        get_actor_states_by_names : Names-based version

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """Set actor states by pre-computed indices (performance optimized).

        Parameters
        ----------
        indices : ActorIndices
            Pre-computed actor indices (see ActorIndices type)
        states : ActorStates
            New actor states matching the provided indices
        write_updates : bool
            Whether to sync changes immediately

        Examples
        --------
        >>> # Cache indices for repeated use
        >>> cached_indices = sim.get_actor_indices(["obj0", "table"], env_ids)
        >>>
        >>> # Fast update using cached indices
        >>> sim.set_actor_states_by_index(cached_indices, new_states)

        >>> # Batch operations for performance
        >>> indices1 = sim.get_actor_indices(["obj0"], env_ids)
        >>> indices2 = sim.get_actor_indices(["table"], env_ids)
        >>>
        >>> sim.set_actor_states_by_index(indices1, states1, write_updates=False)
        >>> sim.set_actor_states_by_index(indices2, states2, write_updates=False)
        >>> sim.write_state_updates()  # Single batch sync

        Notes
        -----
        - Performance-optimized method for hot paths
        - No name lookup overhead
        - Indices must be pre-computed using get_actor_indices()
        - IsaacGym: write_updates parameter has no effect (always immediate)
        - IsaacSim: write_updates=False allows batching for performance

        See Also
        --------
        get_actor_indices : Get indices from names
        set_actor_states_by_names : Names-based version
        write_state_updates : Manual sync method for batched updates

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def get_actor_indices(self, names: str | list[str], env_ids: EnvIds | None = None) -> torch.Tensor:
        """Get actor indices by object name(s) for unified tensor access.

        Returns flat tensor indices that can be used to access object states in
        unified tensors like all_root_states. These indices account for the virtual
        address space layout and multi-environment setup.

        Parameters
        ----------
        names : Union[str, list[str]]
            Single object name or list of object names to get indices for.
            Examples: "obj0", ["obj0", "table_pantene", "robot"]
        env_ids : EnvIds | None, default=None
            Environment IDs to query (see EnvIds type). If None, returns poses for all environments.

        Returns
        -------
        torch.Tensor
            Flat tensor indices for accessing unified tensors, shape [num_objects * num_envs],
            dtype torch.long. Can be used directly with all_root_states[indices].

        Examples
        --------
        >>> # Always get indices dynamically - don't hardcode values
        >>> env_ids = torch.tensor([0, 1, 2], device=device)
        >>> indices = sim.get_actor_indices(["obj0", "table"], env_ids)
        >>> states = sim.all_root_states[indices]  # Direct tensor access

        >>> # Preferred: Use unified interface methods
        >>> states = sim.get_actor_states(["obj0", "table"], env_ids)

        Important Notes
        ---------------
        **Index Ordering**: All backends share one convention, **name-major, env-minor**:
        for ``get_actor_indices(names, env_ids)`` the result groups all of ``names[0]``'s
        envs first, then ``names[1]``'s, etc. (matches ``ObjectRegistry.get_initial_poses_batch``).
        ``get_actor_states`` / ``set_actor_states`` rely on this, so a caller pairing
        ``get_actor_initial_poses`` with ``set_actor_states`` must group both the same way.

        **Best Practices**:
        - **Always get indices dynamically** - never assume specific index values
        - Use unified interface methods (get_actor_states, set_actor_states) when possible
        - Don't hardcode index values in task/environment code
        - Cache indices if used repeatedly in performance-critical code

        **Virtual Address Space**:
        Objects are organized by environment rather than by type for performance:

        [robot_env0, obj0_env0, obj1_env0, # env 0 objects
         robot_env1, obj0_env1, obj1_env1, # env 1 objects
         robot_env2, obj0_env2, obj1_env2, # env 2 objects
         ...]

        See Also
        --------
        get_actor_states : Preferred method for getting object states
        set_actor_states : Preferred method for setting object states

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def get_actor_initial_poses(self, names: list[str], env_ids: EnvIds | None = None) -> torch.Tensor:
        """Get the poses set during the initial scene setup and before play.

        Retrieves the default/starting poses for specified actors as configured
        in scene files or object configurations. These are the baseline poses used
        for resets and randomization, not the current simulation poses.

        Parameters
        ----------
        names : list[str]
            List of object names to get initial poses for.
            Examples: ["obj0", "table_pantene", "shelf"]
        env_ids : EnvIds | None, default=None
            Environment IDs to query (see EnvIds type). If None, applies to all environments.

        Returns
        -------
        torch.Tensor
            Initial poses with shape [len(names) * len(env_ids), 7], dtype torch.float32.
            Format: [x, y, z, qx, qy, qz, qw] where quaternion is in xyzw format.

        Examples
        --------
        >>> # Get initial poses for reset/randomization
        >>> env_ids = torch.tensor([0, 1], device=device)
        >>> initial_poses = sim.get_actor_initial_poses(["obj0", "table"], env_ids)
        >>> # Shape: [4, 7] for 2 objects x 2 environments

        >>> # Use for randomization
        >>> new_states = torch.zeros(len(env_ids) * len(objects), 13, device=device)
        >>> new_states[:, :7] = initial_poses  # Position + quaternion
        >>> new_states[:, 7:] = 0.0           # Zero velocities
        >>> # Apply random offsets...
        >>> sim.set_actor_states(objects, env_ids, new_states)

        >>> # Reset to exact initial configuration
        >>> objects = ["obj0", "table_pantene"]
        >>> env_ids = torch.tensor([0, 1, 2], device=device)
        >>> initial_poses = sim.get_actor_initial_poses(objects, env_ids)
        >>>
        >>> # Build full reset states
        >>> reset_states = torch.zeros(len(objects) * len(env_ids), 13, device=device)
        >>> reset_states[:, :7] = initial_poses  # Use exact initial poses
        >>> reset_states[:, 7:] = 0.0           # Zero velocities
        >>> sim.set_actor_states(objects, env_ids, reset_states)

        Notes
        -----
        Initial poses represent the "default" or "starting" configuration from:
        - Scene file object transforms
        - Individual object configuration positions
        - Robot initial state configuration

        These are NOT the current simulation poses - use get_actor_states() for current poses.

        Frame: the returned pose is **WORLD-frame (env_origins already applied) on every backend**,
        the same single convention as ``get_actor_states`` and ``set_actor_states``. Feed a
        ``get_actor_initial_poses`` result STRAIGHT into ``set_actor_states`` to reset (as the
        reset/jitter terms do); to place at a hand-built target, pass world coordinates directly.
        Never add ``env_origins`` yourself and never branch on simulator type.

        See Also
        --------
        get_actor_states : Get current simulation poses (always WORLD frame on every backend)
        set_actor_states : Set object states using these poses (same WORLD frame as here)

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    # ----- Robot-Specific Interface -----

    def set_actor_root_state_tensor(self, set_env_ids: EnvIds, root_states: torch.Tensor) -> None:
        """Set **robot** root states (position/orientation) - backwards compatibility interface.

        This method provides backwards compatibility for existing code that expects
        to set robot states using the legacy interface. Delegates to the standardized
        robot-specific method.

        Parameters
        ----------
        set_env_ids : EnvIds
            Environment IDs to update (see EnvIds type)
        root_states : torch.Tensor
            Robot root states to set, shape [num_envs, 13], dtype torch.float32.
            Format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            where quaternion is in xyzw format.

        Note
        ----
        This method is kept for backwards compatibility. New code should use
        `set_actor_root_state_tensor_robots` directly.
        """
        # Reset the robot, kept for backwards compatiblity
        return self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_actor_root_state_tensor_robots(
        self, env_ids: EnvIds | None = None, root_states: torch.Tensor | None = None
    ) -> None:
        """Set robot root states (position/orientation) - standardized interface.

        This method provides a consistent interface across simulators for setting
        robot root states including position, orientation, and velocities.

        Parameters
        ----------
        env_ids : EnvIds | None, default=None
            Environment IDs to query (see EnvIds type). If None, applies to all environments.
        root_states : torch.Tensor | None, default=None
            Robot states to set, shape [num_envs, 13], dtype torch.float32.
            Format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            where quaternion is in xyzw format. If None, uses current robot_root_states.

        Examples
        --------
        >>> # Reset specific environments to current states
        >>> env_ids = torch.tensor([0, 1, 2], device=device)
        >>> sim.set_actor_root_state_tensor_robots(env_ids)

        >>> # Set specific robot states
        >>> reset_states = torch.zeros(len(env_ids), 13, device=device)
        >>> reset_states[:, :3] = initial_positions  # Set positions
        >>> reset_states[:, 3:7] = initial_quaternions  # Set orientations (xyzw)
        >>> reset_states[:, 7:] = 0.0  # Zero velocities
        >>> sim.set_actor_root_state_tensor_robots(env_ids, reset_states)

        Notes
        -----
        - Works consistently across IsaacGym and IsaacSim
        - Environment origins are automatically handled by the framework
        - Quaternion format is xyzw (standard format)

        See Also
        --------
        set_dof_state_tensor_robots : Set robot joint states
        set_actor_states : Unified interface for all objects

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement set_actor_root_state_tensor_robots")

    def set_dof_state_tensor(self, set_env_ids: EnvIds, dof_states: torch.Tensor | None = None) -> None:
        """Set **robot** DOF states (joint positions/velocities) - backwards compatibility interface.

        This method provides backwards compatibility for existing code that expects
        to set robot DOF states using the legacy interface. Delegates to the standardized
        robot-specific method.

        Parameters
        ----------
        set_env_ids : EnvIds
            Environment IDs to update (see EnvIds type)
        dof_states : torch.Tensor | None, default=None
            DOF states to set. If None, uses current dof_state.

        Note
        ----
        This method is kept for backwards compatibility. New code should use
        `set_dof_state_tensor_robots` directly.
        """
        # Kept for backwards compatiblity
        robot_dof_states = dof_states  # non-robot articulations are not yet supported
        return self.set_dof_state_tensor_robots(set_env_ids, robot_dof_states)

    def set_dof_state_tensor_robots(
        self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None
    ) -> None:
        """Set robot DOF states (joint positions/velocities) - standardized interface.

        This method provides a consistent interface across simulators for setting
        robot joint positions and velocities.

        **IMPORTANT**: Tensor shapes differ between simulators. See simulator-specific
        documentation for the exact format expected by each implementation.

        Parameters
        ----------
        env_ids : EnvIds | None, default=None
            Environment IDs to query (see EnvIds type)
        dof_states : torch.Tensor | None, default=None
            DOF states to set, dtype torch.float32. **Shape varies by simulator**:

            - **IsaacSim**: [num_envs, num_dofs, 2] - 3D tensor
            - **IsaacGym**: [num_envs * num_dofs, 2] - 2D flattened tensor

            Format: positions and velocities (see simulator docs for indexing).
            If None, uses current dof_state.

        Examples
        --------
        See simulator-specific implementations for correct tensor format examples:
        - IsaacSim: `holosoma.simulator.isaacsim.IsaacSim.set_dof_state_tensor_robots`
        - IsaacGym: `holosoma.simulator.isaacgym.IsaacGym.set_dof_state_tensor_robots`

        Notes
        -----
        - **Tensor shape inconsistency**: Different simulators expect different shapes
        - Joint positions should be within DOF limits
        - Joint velocities are typically set to zero for resets
        - Always check simulator-specific documentation for correct tensor format

        See Also
        --------
        set_actor_root_state_tensor_robots : Set robot root states
        get_dof_limits_properties : Get joint limits for validation

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement set_dof_state_tensor_robots")
