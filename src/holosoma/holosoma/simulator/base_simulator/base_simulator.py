from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorInitConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.shared.object_registry import ObjectRegistry
from holosoma.simulator.shared.scene_types import SceneInterface
from holosoma.simulator.types import ActorIndices, ActorNames, ActorStates, EnvIds
from holosoma.utils.experiment_paths import get_video_dir
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

if TYPE_CHECKING:
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

    **Direct Tensor Access (Advanced)**:
    >>> # Always get indices dynamically on or after setup - don't hardcode values
    >>> indices = sim.get_actor_indices(["obj0", "obj1"], env_ids)
    >>>
    >>> # IsaacGym-style direct access (works in both via proxy)
    >>> sim.all_root_states[indices] = new_states
    >>> sim.write_state_updates()  # Sync changes (no-op in IsaacGym)

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
    robot_config: RobotConfig

    sim_dt: float
    viewer: object
    robot_root_states: torch.Tensor
    base_quat: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    contact_forces: torch.Tensor
    contact_forces_history: torch.Tensor
    scene: SceneInterface
    all_root_states: torch.Tensor

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
        self.robot_config = tyro_config.robot
        self.video_config = tyro_config.logger.video
        self.sim_device = device
        self.headless = False
        self.debug_viz_enabled = self.simulator_config.debug_viz
        self.object_registry = ObjectRegistry(device)

        # Virtual gantry system
        self.virtual_gantry: VirtualGantry | None = None

        # Video recording system
        self.video_recorder: VideoRecorderInterface | None = None

        # Bridge system
        self.bridge: SimulatorBridge | None = None

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

    def get_supported_scene_formats(self):
        """Returns the scene formats supported by this simulator in order of preference.

        This method should be implemented by subclasses to specify which scene file
        formats they can load and in what order of preference they should be tried.

        Returns
        -------
        List[str]
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

    def clear_contact_forces_history(self, env_id):
        """
        Clears the contact forces history for the specified environment.
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
            logger.info("Bridge system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bridge system: {e}")
            raise

    def _step_bridge(self) -> None:
        """Step bridge system if enabled.

        Should be called by subclasses during each physics step,
        typically before physics simulation. Handles bridge state
        publishing and command processing.
        """
        if self.bridge is not None:
            self.bridge.step()

    # ----- Video Recording Interface -----
    def on_episode_start(self, env_id: int = 0) -> None:
        """Called when an episode starts.

        This method provides a hook for video recording and other episode-based
        functionality. Subclasses can override this method to add additional
        episode start logic while calling super() to maintain video recording.

        Parameters
        ----------
        env_id : int, default=0
            The environment ID where the episode is starting.
        """
        if self.virtual_gantry is not None and env_id == 0:
            # Follow robot on start (may want this configurable later)
            self.virtual_gantry.set_position_to_robot()

        if self.video_recorder is not None:
            self.video_recorder.on_episode_start(env_id)

    def on_episode_end(self, env_id: int = 0) -> None:
        """Called when an episode ends.

        This method provides a hook for video recording and other episode-based
        functionality. Subclasses can override this method to add additional
        episode end logic while calling super() to maintain video recording.

        Parameters
        ----------
        env_id : int, default=0
            The environment ID where the episode is ending.
        """
        if self.video_recorder is not None:
            self.video_recorder.on_episode_end(env_id)

    def capture_video_frame(self, env_id: int = 0) -> None:
        """Capture a video frame during simulation.

        This method should be called during each simulation step when video
        recording is active. It delegates to the video recorder if one is
        configured and currently recording.

        Parameters
        ----------
        env_id : int, default=0
            The environment ID where the frame is being captured.
        """
        if self.video_recorder is not None:
            self.video_recorder.capture_frame(env_id)

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

        **Direct tensor access (IsaacGym-style):**
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
        - Environment origins are automatically applied by the framework
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
        **Index Ordering**: Actor indices ordering may differ between simulator
        implementations. This is intentional for now.

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
        Environment origins are NOT applied to initial poses - they represent the base configuration.

        See Also
        --------
        get_actor_states : Get current simulation poses
        set_actor_states : Set object states using these poses

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
