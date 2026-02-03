"""MuJoCo simulator implementation.

The simulator follows the BaseSimulator interface while providing MuJoCo-specific
implementations for terrain rendering, contact detection, and physics simulation, etc.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np
import torch
from loguru import logger

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.mujoco.backends import WARP_AVAILABLE, ClassicBackend, WarpBackend
from holosoma.simulator.mujoco.command_registry import CommandRegistry
from holosoma.simulator.mujoco.fields import prepare_fields, prepare_manager_fields
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
from holosoma.simulator.mujoco.tensor_views import (
    create_base_linear_acceleration_view,
)
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.shared.virtual_gantry import create_virtual_gantry
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.adapters import mujoco_draw_adapter


class MuJoCoScene:
    """MuJoCo Scene implementation following SceneInterface protocol.

    Provides a scene interface for MuJoCo simulations that manages environment
    origins and provides compatibility with the holosoma scene system.
    """

    def __init__(self, env_origins: torch.Tensor, device: str) -> None:
        """Initialize MuJoCo Scene.

        Parameters
        ----------
        env_origins : torch.Tensor
            Environment origins tensor with shape [num_envs, 3].
        device : str
            Device string ('cpu' or 'cuda').

        Raises
        ------
        TypeError
            If env_origins is not a torch.Tensor.
        ValueError
            If env_origins doesn't have the correct shape.
        """
        logger.info(f"Initializing MuJoCo Scene with env_origins shape: {env_origins.shape}, device: {device}")

        # Validate input tensor
        if not isinstance(env_origins, torch.Tensor):
            raise TypeError(f"env_origins must be torch.Tensor, got {type(env_origins)}")

        if env_origins.dim() != 2 or env_origins.shape[1] != 3:
            raise ValueError(f"env_origins must have shape [num_envs, 3], got {env_origins.shape}")

        # Ensure tensor is on correct device with correct dtype
        self._env_origins = env_origins.to(device=device, dtype=torch.float32)
        self._device = device

        logger.info(f"MuJoCo Scene initialized successfully - {self._env_origins.shape[0]} environments")

    @property
    def env_origins(self) -> torch.Tensor:
        """Get environment origins tensor.

        Returns
        -------
        torch.Tensor
            Environment origins with shape [num_envs, 3].
        """
        return self._env_origins


class MuJoCo(BaseSimulator):
    """MuJoCo physics simulator with terrain support.

    This class provides a MuJoCo-based physics simulator that provides compatibility with
    the holosoma simulator interface with unified state access and the shared terrain system.
    """

    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str) -> None:
        """Initialize MuJoCo simulator.

        Parameters
        ----------
        tyro_config : FullSimConfig
            Tyro configuration containing simulator, robot, and terrain settings.
        device : str
            Device type for simulation ('cpu' or 'cuda').

        Raises
        ------
        ValueError
            If robot configuration is missing from tyro_config.
        """
        simulator_config = tyro_config.simulator

        logger.info("=== MuJoCo Simulator Initialization Started ===")
        logger.info(f"Device: {device}")
        logger.info(f"Simulator config: {simulator_config}")

        super().__init__(tyro_config, terrain_manager, device)

        # Set robot config for consistency with Isaac simulators
        if not hasattr(tyro_config, "robot"):
            raise ValueError("Robot configuration is required but missing from tyro_config")

        # Store full config for backend access
        self.tyro_config = tyro_config
        self.device = device
        self.robot_config = tyro_config.robot

        # Save num_envs on init() rather than create_envs() so other modules can rely on it
        self.num_envs = self.training_config.num_envs

        # MuJoCo-specific attributes
        self.root_model: mujoco.MjModel | None = None
        self.root_data: mujoco.MjData | None = None

        # Name mapping for prefix handling, because the robot is placed at a named site within
        # Mujoco.
        self.clean_to_prefixed_names: dict[str, str] = {}  # "hip_joint" -> "robot_hip_joint"
        self.prefixed_to_clean_names: dict[str, str] = {}  # "robot_hip_joint" -> "hip_joint"

        # Minimal state tensors (placeholders)
        self.dof_pos = torch.zeros(0, device=device)
        self.dof_vel = torch.zeros(0, device=device)
        self.contact_forces = torch.zeros(0, device=device)

        # Viewer
        self.viewer: mujoco.viewer.Handle | None = None

        # World ID for multi-environment visualization (which environment to view)
        self.current_world_id: int = 0

        # Text overlay visibility toggle
        self.show_text_overlay: bool = True

        # Command system for keyboard/joystick controls
        # Initialize commands tensor matching IsaacGym format:
        #    [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        # Shape: [num_envs, 9] to match IsaacGym command structure
        self.commands: torch.Tensor | None = None  # Will be initialized in create_envs when num_envs is known

        logger.info("=== MuJoCo Simulator Initialization Completed ===")

    def _build_name_maps(self) -> None:
        """Build bidirectional name maps for clean <-> prefixed name translation.

        Creates mapping dictionaries to translate between clean names (used by holosoma)
        and prefixed names (used internally by MuJoCo) for joints, bodies, and actuators.
        """
        self.clean_to_prefixed_names.clear()
        self.prefixed_to_clean_names.clear()

        prefix = self.scene_manager.robot_prefix

        # Build joint name maps
        assert self.root_model
        for joint_id in range(self.root_model.njnt):
            prefixed_name = self.root_model.joint(joint_id).name
            if prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        # Build body name maps
        for body_id in range(self.root_model.nbody):
            prefixed_name = self.root_model.body(body_id).name
            if prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        # Build actuator name maps
        for actuator_id in range(self.root_model.nu):
            prefixed_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if prefixed_name and prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        logger.info(f"Built name maps: {len(self.clean_to_prefixed_names)} clean->prefixed mappings")

    def _build_body_index_mapping(self) -> None:
        """Build MuJoCo body ID to holosoma body index mapping for contact forces.

        Creates a mapping from MuJoCo's internal body IDs to holosoma's body indices,
        which is essential for correctly attributing contact forces to the right
        bodies in the contact force tensor.
        """
        self.mujoco_to_holosoma_body_map: dict[int, int] = {}

        logger.info("=== Building MuJoCo body ID to holosoma index mapping ===")

        # holosoma body_names excludes world, so index 0 = first robot body
        for holosoma_idx, body_name in enumerate(self.body_names):
            # Find corresponding MuJoCo body ID
            prefixed_name = self._get_prefixed_name(body_name)
            mujoco_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
            if mujoco_body_id != -1:
                self.mujoco_to_holosoma_body_map[mujoco_body_id] = holosoma_idx
                logger.info(
                    f"Body mapping: '{body_name}' -> '{prefixed_name}' | MuJoCo ID {mujoco_body_id} -> "
                    f"holosoma idx {holosoma_idx}"
                )
            else:
                logger.warning(f"Body mapping FAILED: '{body_name}' -> '{prefixed_name}' | MuJoCo ID not found")

        logger.info(f"=== Body mapping complete: {len(self.mujoco_to_holosoma_body_map)} mappings created ===")

    def _get_prefixed_name(self, clean_name: str) -> str:
        """Get prefixed name from clean name using map lookup.

        Parameters
        ----------
        clean_name : str
            Clean name without prefix.

        Returns
        -------
        str
            Prefixed name for MuJoCo lookup, or original name if not found.
        """
        return self.clean_to_prefixed_names.get(clean_name, clean_name)

    def _get_clean_name(self, prefixed_name: str) -> str:
        """Get clean name from prefixed name using map lookup.

        Parameters
        ----------
        prefixed_name : str
            Prefixed name from MuJoCo.

        Returns
        -------
        str
            Clean name for holosoma use, or original name if not found.
        """
        return self.prefixed_to_clean_names.get(prefixed_name, prefixed_name)

    def set_headless(self, headless: bool) -> None:
        """Set headless mode for the simulator.

        Parameters
        ----------
        headless : bool
            Whether to run in headless mode (no visualization).
        """
        super().set_headless(headless)
        self.headless = headless

    def setup(self) -> None:
        """Initialize simulator parameters and environment."""
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self) -> None:
        """Configure terrain - deferred until load_assets."""
        return

    def clear_lines(self) -> None:
        """Clear debug visualization lines."""
        mujoco_draw_adapter.clear_lines(self)

    def draw_sphere(
        self, pos: torch.Tensor, radius: float, color: torch.Tensor, env_id: int, pos_id: int | None = None
    ) -> None:
        """Draw a debug sphere at the specified position.

        Parameters
        ----------
        pos : torch.Tensor
            Position of the sphere.
        radius : float
            Radius of the sphere.
        color : torch.Tensor
            Color of the sphere.
        env_id : int
            Environment ID.
        pos_id : Optional[int]
            Position ID for the sphere.
        """
        mujoco_draw_adapter.draw_sphere(self, pos, radius, color, env_id, pos_id=pos_id)

    def draw_line(self, start_point: torch.Tensor, end_point: torch.Tensor, color: torch.Tensor, env_id: int) -> None:
        """Draw a debug line between two points.

        Parameters
        ----------
        start_point : torch.Tensor
            Starting point of the line.
        end_point : torch.Tensor
            Ending point of the line.
        color : torch.Tensor
            Color of the line.
        env_id : int
            Environment ID.
        """
        mujoco_draw_adapter.draw_line(self, start_point, end_point, color, env_id)

    def load_assets(self):
        """Load assets using compositional MjSpec approach.

        Creates the scene manager, sets up the scene components (terrain, lighting,
        materials, robot), compiles the final model, and initializes robot properties
        and joint addressing for simulation.
        """
        logger.info("=== Loading assets ===")

        # Create scene manager
        self.scene_manager = MujocoSceneManager(self.simulator_config)
        self._setup_scene()

        # Compile once at the end
        self.root_model = self.scene_manager.compile()
        self.root_data = mujoco.MjData(self.root_model)

        # Apply post-compilation settings
        self.root_model.opt.timestep = self.sim_dt

        # Backend selection based on configuration
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            if not WARP_AVAILABLE:
                raise RuntimeError(
                    "WarpBackend requested (mujoco_backend='warp') but dependencies not available.\n\n"
                    "To enable GPU acceleration, reinstall with warp support:\n"
                    "  bash scripts/setup_mujoco.sh --with-warp\n\n"
                    "Or install dependencies manually:\n"
                    "  pip install warp-lang mujoco-warp\n\n"
                    "System requirements: CUDA-capable GPU required"
                )
            logger.info("Initializing WarpBackend (GPU multi-environment)")
            self.backend = WarpBackend(self.root_model, self.root_data, self.tyro_config, self.device)
            # Sync CPU initial state (set by _set_initial_joint_angles) to GPU
            self.backend.initialize_state(self.root_model, self.root_data)
        else:
            logger.info("Initializing ClassicBackend (CPU single-environment)")
            self.backend = ClassicBackend(self.root_model, self.root_data, self.tyro_config, self.device)

        # Setup robot indexes, etc
        self._set_robot_properties()
        self._set_robot_joint_addressing()
        self._set_initial_joint_angles()

        # Initialize virtual gantry after the robot using config
        gantry_cfg = self.simulator_config.virtual_gantry
        self.virtual_gantry = create_virtual_gantry(
            sim=self,
            enable=gantry_cfg.enabled,
            attachment_body_names=gantry_cfg.attachment_body_names,
            cfg=gantry_cfg,
        )

        # Initialize bridge system using base class helper
        self._init_bridge()

        if self.video_config.enabled:
            self.video_recorder = MuJoCoVideoRecorder(self.video_config, self)
            self.video_recorder.setup_recording()

        # For debugging
        self.print_mujoco_model_tree()

        logger.info(f"Assets loaded - num_dof: {self.num_dof}, num_bodies: {self.num_bodies}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")

    def _setup_scene(self) -> None:
        """Setup scene by composing terrain, lighting, materials, and robot components.

        Follows a specific composition order: terrain first (if not 'none' or 'fake'),
        then lighting and materials, and finally the robot. This ensures proper
        collision configuration and scene element integration.
        """
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state.mesh_type not in ["none", "fake"]:
            # For now, use mesh type to decide whether to programmatically
            # setup scene, terrain, etc. Cannot use "none" since env code relies on none
            # to literally mean none, so we use "fake"
            # This also means robot self_collisions are ignored because we're not in control
            # of the terrain/floor/ground, etc. In this case, the robot MJCF XML needs to handle
            # for collisions (or not).
            self.scene_manager.add_terrain(terrain_state, self.training_config.num_envs)
            self.scene_manager.add_lighting()
            self.scene_manager.add_materials()

        # Always add robot after terrain, in case it references ground/floor, etc for contacts
        self.scene_manager.add_robot(
            terrain_state, self.robot_config, xml_filter=self.simulator_config.robot_mjcf_filter
        )

    def _set_robot_properties(self) -> None:
        """Set robot properties including DOF names, body names, and index mappings.

        Extracts robot joint and body information from the compiled MuJoCo model,
        filters out non-robot elements, and creates the necessary mappings for
        holosoma compatibility.
        """
        # Get all joint names
        assert self.root_model
        all_joint_names = [self.root_model.joint(i).name for i in range(self.root_model.njnt)]

        # Filter out freejoints
        # TODO: make more robust/not hardcoded names, also handle objects
        prefix = self.scene_manager.robot_prefix
        exclude_names = [
            f"{prefix}freejoint",
            f"{prefix}floating_base_joint",
            f"{prefix}",  # keep named joints only
            "",  # keep named joints only
        ]

        robot_joint_names = [n for n in all_joint_names if n not in exclude_names]

        # Build name maps first
        self._build_name_maps()

        self.num_dof = len(robot_joint_names)
        # Use map lookup for clean names
        self.dof_names = [self._get_clean_name(name) for name in robot_joint_names]
        self.num_bodies = self.root_model.nbody
        # Use map lookup for body names
        all_body_names = [self._get_clean_name(self.root_model.body(i).name) for i in range(self.root_model.nbody)]
        self.body_names = [name for name in all_body_names if name != "world"]

        # Build body index mapping for contact forces (after body_names is defined)
        self._build_body_index_mapping()

        # Add _body_list attribute for compatibility with whole_body_tracking environment
        # Needs to be encapsulated and added to base simulator interface
        self._body_list = self.body_names

        logger.info(f"Total joints: {len(all_joint_names)}, Robot DOFs: {self.num_dof}")
        logger.info(f"Robot joint names (prefixed): {robot_joint_names}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")

    def _set_robot_joint_addressing(self) -> None:
        """Setup proper joint addressing using named freejoint and MuJoCo APIs.

        Configures addressing for the robot's freejoint (for root body control)
        and all DOF joints, storing the necessary qpos and qvel addresses for
        efficient state access during simulation.
        """
        logger.info("=== Setting up robot joint addressing ===")

        # Find the named freejoint for robot root control (use prefixed name)
        assert self.root_model
        has_freejoint = True
        freejoint_name = self._get_prefixed_name("floating_base_joint")
        self.robot_freejoint_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)

        if self.robot_freejoint_id == -1:
            logger.warning(f"Robot freejoint '{freejoint_name}' not found in model")
            self.robot_freejoint_id = 0
            has_freejoint = False

        # Validate it's actually a freejoint
        if has_freejoint and self.root_model.jnt_type[self.robot_freejoint_id] != mujoco.mjtJoint.mjJNT_FREE:
            joint_type = self.root_model.jnt_type[self.robot_freejoint_id]
            raise ValueError(f"Joint '{freejoint_name}' is not a freejoint, got type {joint_type}")

        # Get addressing info for freejoint
        self.robot_qpos_addr = self.root_model.jnt_qposadr[self.robot_freejoint_id]
        self.robot_qvel_addr = self.root_model.jnt_dofadr[self.robot_freejoint_id]

        logger.info(
            f"Robot freejoint addressing: ID={self.robot_freejoint_id}, "
            f"qpos_addr={self.robot_qpos_addr}, qvel_addr={self.robot_qvel_addr}"
        )

        # Setup DOF joint addressing using proper MuJoCo APIs
        self.dof_qpos_addrs = []
        self.dof_qvel_addrs = []

        for dof_name in self.dof_names:
            # Add prefix for MuJoCo lookup (dof_names are clean, need prefixed version)
            joint_name = self._get_prefixed_name(dof_name)
            joint_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if joint_id == -1:
                raise ValueError(f"DOF joint '{joint_name}' (clean name: '{dof_name}') not found in model")

            qpos_addr = self.root_model.jnt_qposadr[joint_id]
            qvel_addr = self.root_model.jnt_dofadr[joint_id]

            self.dof_qpos_addrs.append(qpos_addr)
            self.dof_qvel_addrs.append(qvel_addr)

        logger.info(f"Setup {len(self.dof_qpos_addrs)} DOF joint addresses")
        logger.info("=== Robot joint addressing setup completed ===")

    def _set_initial_joint_angles(self) -> None:
        """Set initial joint angles from robot configuration.

        Applies the default joint angles specified in the robot configuration
        to the MuJoCo model's initial state, then performs forward kinematics
        to update body positions.
        """
        logger.info("Setting initial joint angles from robot config")

        assert self.root_model
        assert self.root_data

        default_joint_angles = self.robot_config.init_state.default_joint_angles
        joint_angles_set = 0
        joint_angles_failed = 0
        for joint_name, angle in default_joint_angles.items():
            # Add prefix for MuJoCo lookup
            mujoco_joint_name = self._get_prefixed_name(joint_name)
            joint_id = None
            for i in range(self.root_model.njnt):
                if self.root_model.joint(i).name == mujoco_joint_name:
                    joint_id = i
                    break

            if joint_id is None:
                logger.warning(f"Joint '{joint_name}' (MuJoCo name: '{mujoco_joint_name}') not found in model")
                joint_angles_failed += 1
                continue

            try:
                # Get the qpos address for this joint
                joint_qposadr = self.root_model.jnt_qposadr[joint_id]
                self.root_data.qpos[joint_qposadr] = angle
                joint_angles_set += 1
                logger.info(
                    f"Set joint '{joint_name}' -> '{mujoco_joint_name}' (ID: {joint_id}, "
                    f"qpos_addr: {joint_qposadr}) to angle {angle}"
                )
            except Exception as e:
                logger.warning(f"Failed to set angle for joint '{joint_name}': {e}")
                joint_angles_failed += 1

        if joint_angles_failed > 0:
            raise RuntimeError("Failed to set joint angles")

        logger.info(
            f"Joint angle setting complete: {joint_angles_set} set, {joint_angles_failed} "
            f"failed out of {len(default_joint_angles)} total"
        )

        # Forward kinematics to update body positions based on joint angles
        mujoco.mj_forward(self.root_model, self.root_data)
        logger.info("Applied forward kinematics with initial joint angles")

    def get_supported_scene_formats(self) -> list[str]:
        """Get supported scene formats.

        Returns
        -------
        list[str]
            List of supported scene formats (currently empty).
        """
        return []  # not yet supported

    def create_envs(self, num_envs, env_origins, base_init_state):
        """Create environments - enhanced implementation with robot support.

        Parameters
        ----------
        num_envs : int
            Number of environments to create (currently limited to 1).
        env_origins : torch.Tensor
            Environment origin positions.
        base_init_state : dict[str, Any]
            Initial state configuration for the base.

        Raises
        ------
        ValueError
            If num_envs > 1 (multiple environments not yet supported).
        """
        if num_envs > 1 and self.simulator_config.mujoco_backend != MujocoBackend.WARP:
            raise ValueError(
                f"MuJoCo ClassicBackend only supports single environment, got {num_envs}. "
                f"Use --simulator.config.mujoco-backend=warp for multi-environment support."
            )

        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        # Create Scene following SceneInterface protocol
        self.scene = MuJoCoScene(self.env_origins, self.sim_device)

        # Initialize state tensors based on actual DOF count
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)

        # Initialize contact forces tensor with correct shape [num_envs, num_bodies, 3]
        # This matches the interface expected by holosoma (IsaacGym/IsaacSim pattern)
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.sim_device)

        # Initialize contact forces history tensor to match IsaacGym/IsaacSim pattern
        # Shape: [num_envs, history_length, num_bodies, 3]
        history_length = self.simulator_config.contact_sensor_history_length
        self.contact_forces_history = torch.zeros(
            self.num_envs, history_length, self.num_bodies, 3, device=self.sim_device
        )

        # Initialize command system (Phase 1)
        # Command tensor format matching IsaacGym: [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        self.commands = torch.zeros(self.num_envs, 9, device=self.sim_device, dtype=torch.float32)
        logger.info(f"Initialized command system with shape: {self.commands.shape}")

    def _set_robot_initial_state(self) -> None:
        """Set complete initial robot state (position, orientation, velocities).

        Applies the robot's initial state configuration to the MuJoCo model,
        including root body position, orientation, and velocities.
        """
        assert self.root_data
        assert self.robot_config
        assert self.robot_qpos_addr is not None
        assert self.robot_qvel_addr is not None

        # Set complete initial robot state (position, orientation, velocities)
        initial_pos = self.robot_config.init_state.pos
        initial_rot = self.robot_config.init_state.rot  # [x,y,z,w] quaternion
        initial_lin_vel = self.robot_config.init_state.lin_vel
        initial_ang_vel = self.robot_config.init_state.ang_vel

        # Apply initial state to robot root body if it exists

        # Convert quaternion: holosoma [x,y,z,w] → MuJoCo [w,x,y,z]
        initial_rot_mj = [initial_rot[3], initial_rot[0], initial_rot[1], initial_rot[2]]

        # Use the existing _set_robot_joint_addressing() results
        # Set position: [x, y, z, qw, qx, qy, qz] (7 elements)
        self.root_data.qpos[self.robot_qpos_addr : self.robot_qpos_addr + 3] = initial_pos
        self.root_data.qpos[self.robot_qpos_addr + 3 : self.robot_qpos_addr + 7] = initial_rot_mj

        # Set velocity: [vx, vy, vz, wx, wy, wz] (6 elements)
        self.root_data.qvel[self.robot_qvel_addr : self.robot_qvel_addr + 3] = initial_lin_vel
        self.root_data.qvel[self.robot_qvel_addr + 3 : self.robot_qvel_addr + 6] = initial_ang_vel

    def prepare_sim(self) -> None:
        """Prepare simulation - enhanced implementation with ObjectRegistry integration.

        Resets simulation data, sets initial robot state, configures the object registry,
        and creates tensor views for efficient state access during simulation.
        """
        # Reset simulation data
        assert self.root_data
        mujoco.mj_resetData(self.root_model, self.root_data)

        self._set_robot_initial_state()

        # Setup ObjectRegistry for robot-only (scenes not yet implemented)
        self.object_registry.setup_ranges(self.num_envs, robot_count=1, scene_count=0, individual_count=0)

        # Register robot with initial pose
        # TODO: use robot model/data rather than config here?
        robot_poses = torch.zeros(self.num_envs, 7, device=self.sim_device)
        robot_poses[:, :3] = torch.tensor(self.robot_config.init_state.pos, device=self.sim_device)
        robot_poses[:, 3:7] = torch.tensor(self.robot_config.init_state.rot, device=self.sim_device)  # [x,y,z,w]
        self.object_registry.register_object("robot", ObjectType.ROBOT, 0, robot_poses)
        self.object_registry.finalize_registration()

        # Calculate indices for robot freejoint components
        pos_indices = slice(self.robot_qpos_addr, self.robot_qpos_addr + 3)
        quat_indices = slice(self.robot_qpos_addr + 3, self.robot_qpos_addr + 7)
        vel_indices = slice(self.robot_qvel_addr, self.robot_qvel_addr + 3)
        ang_vel_indices = slice(self.robot_qvel_addr + 3, self.robot_qvel_addr + 6)

        # Create robot root states proxy via backend factory
        root_addrs = {
            "pos_indices": pos_indices,
            "quat_indices": quat_indices,
            "vel_indices": vel_indices,
            "ang_vel_indices": ang_vel_indices,
        }
        self.robot_root_states = self.backend.create_root_view(root_addrs)  # type: ignore[assignment]

        # Create all_root_states as a view of robot_root_states (single robot case)
        self.all_root_states = self.robot_root_states

        # Calculate indices for DOF positions and velocities
        dof_pos_indices = (
            slice(min(self.dof_qpos_addrs), max(self.dof_qpos_addrs) + 1) if self.dof_qpos_addrs else slice(0, 0)
        )
        dof_vel_indices = (
            slice(min(self.dof_qvel_addrs), max(self.dof_qvel_addrs) + 1) if self.dof_qvel_addrs else slice(0, 0)
        )
        dof_acc_indices = (
            slice(min(self.dof_qvel_addrs), max(self.dof_qvel_addrs) + 1) if self.dof_qvel_addrs else slice(0, 0)
        )

        # Create DOF state proxy via backend factory
        dof_addrs = {"dof_pos_indices": dof_pos_indices, "dof_vel_indices": dof_vel_indices}
        self.dof_state = self.backend.create_dof_state_view(dof_addrs, self.num_dof)  # type: ignore[assignment]

        # Create individual DOF views via backend factories
        self.dof_pos = self.backend.create_dof_pos_view(dof_pos_indices, self.num_dof)  # type: ignore[assignment]
        self.dof_vel = self.backend.create_dof_vel_view(dof_vel_indices, self.num_dof)  # type: ignore[assignment]
        self.dof_acc = self.backend.create_dof_acc_view(dof_acc_indices, self.num_dof)  # type: ignore[assignment]

        # Create contact forces via backend factory
        self.contact_forces = self.backend.create_force_view(self.num_bodies)  # type: ignore[assignment]

        # Create unified applied forces accessor for external force application (e.g., virtual gantry)
        self.applied_forces = self.backend.get_applied_forces_view()

        # Create base_quat, base_angular_vel, base_linear_acc views via backend
        self.base_quat = self.backend.create_quaternion_view(quat_indices)  # type: ignore[assignment]
        self.base_angular_vel = self.backend.create_angular_velocity_view(ang_vel_indices)  # type: ignore[assignment]

        # Base linear acceleration: backend-specific handling
        base_lin_acc_indices = slice(0, 3)
        if WarpBackend is not None and isinstance(self.backend, WarpBackend):
            # WarpBackend: direct GPU tensor access
            self.base_linear_acc = self.backend.qacc_t[:, base_lin_acc_indices]  # type: ignore[assignment,attr-defined]
        else:
            # ClassicBackend: use view system
            self.base_linear_acc = create_base_linear_acceleration_view(  # type: ignore[assignment]
                qacc_array=self.root_data.qacc,
                indices=base_lin_acc_indices,
                num_envs=self.num_envs,
                device=self.sim_device,
            )

        # Initialize rigid body state tensors (required by BaseTask)
        self._rigid_body_pos = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_rot = torch.zeros(
            self.num_envs, self.num_bodies, 4, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_ang_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )

    def prepare_randomization_fields(self, field_names: list[str]) -> None:
        """Prepare model fields for per-environment randomization.

        Delegates to field_preparation.prepare_fields().

        Parameters
        ----------
        field_names : list[str]
            List of MuJoCo field names to expand for per-environment use.
        """
        prepare_fields(self, field_names)

    def prepare_manager_fields(self, **managers) -> None:
        """Scan managers for field requirements and prepare them.

        Delegates to field_preparation.prepare_manager_fields().

        Parameters
        ----------
        **managers : Any
            Manager instances to scan for field requirements.
        """
        prepare_manager_fields(self, **managers)

    def refresh_sim_tensors(self) -> None:
        """Refresh simulation tensors with actual robot data.

        Updates rigid body state tensors and contact forces from the current
        MuJoCo simulation state. Most state tensors use proxy views that
        automatically reflect the current state.
        """
        if self.num_bodies <= 0:
            logger.info("No bodies to refresh (empty world)")
            return

        # NOTE: With the proxy system, most state tensors (dof_pos, dof_vel, dof_state, robot_root_states)
        # automatically reflect the current MuJoCo state, so we only need to update the non-proxy tensors.

        # Try to get rigid body states via backend (zero-copy for WarpBackend)
        rigid_body_views = self.backend.get_rigid_body_state_views()

        if rigid_body_views is not None:
            # Fast path: zero-copy GPU tensors (WarpBackend)
            # Eliminates 132 tensor allocations per frame for G1 robot (33 bodies x 4 tensors)
            positions, orientations, linear_vel, angular_vel = rigid_body_views
            self._rigid_body_pos[:] = positions
            self._rigid_body_rot[:] = orientations
            self._rigid_body_vel[:] = linear_vel
            self._rigid_body_ang_vel[:] = angular_vel
        else:
            # Slow path: CPU loop with tensor allocation (ClassicBackend)
            assert self.root_model
            assert self.root_data
            for body_id in range(self.num_bodies):
                assert body_id < self.root_model.nbody, (
                    f"Body ID {body_id} exceeds model bodies {self.root_model.nbody}"
                )

                # Positions (direct access to global coordinates)
                self._rigid_body_pos[0, body_id] = (
                    torch.from_numpy(self.root_data.xpos[body_id]).float().to(self.sim_device)
                )

                # Quaternions (convert MuJoCo w,x,y,z to holosoma x,y,z,w)
                mj_quat = self.root_data.xquat[body_id]  # [w, x, y, z]
                holosoma_quat = [mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]]  # [x, y, z, w]
                self._rigid_body_rot[0, body_id] = torch.tensor(
                    holosoma_quat, device=self.sim_device, dtype=torch.float32
                )

                # Velocities using mj_objectVelocity (recommended approach)
                body_vel = np.zeros(6)  # [angular_vel, linear_vel]
                mujoco.mj_objectVelocity(
                    self.root_model, self.root_data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0
                )

                # Extract angular and linear velocities
                self._rigid_body_ang_vel[0, body_id] = torch.from_numpy(body_vel[:3]).float().to(self.sim_device)
                self._rigid_body_vel[0, body_id] = torch.from_numpy(body_vel[3:]).float().to(self.sim_device)

        # Update contact forces and history via backend delegation
        if hasattr(self, "contact_forces_history") and hasattr(self, "contact_forces"):
            self.backend.refresh_sim_tensors(self.contact_forces_history)

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        """Clear contact forces history for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Tensor of environment IDs to clear history for.
        """
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0

    def apply_torques_at_dof(self, torques: torch.Tensor) -> None:
        """Apply torques with backend-specific optimization.

        Parameters
        ----------
        torques : torch.Tensor
            Torques to apply to each DOF.

        Raises
        ------
        ValueError
            If torque count doesn't match actuator count or actuator not found.
        """
        assert self.root_model
        assert self.root_data

        if self.root_model.nu == 0:
            logger.warning("No actuators found in MuJoCo model")
            return

        # Check if backend supports direct tensor writes
        ctrl_tensor = self.backend.get_ctrl_tensor()

        if ctrl_tensor is not None:
            # Fast path: Direct zero-copy write (WarpBackend)
            ctrl_tensor[:] = torques
        else:
            # Slow path: Loop-based write (ClassicBackend)
            torques_np = torques.detach().cpu().numpy().flatten()

            # Verify we have the expected number of actuators
            if len(torques_np) != self.root_model.nu:
                raise ValueError(f"Torque count mismatch: got {len(torques_np)}, expected {self.root_model.nu}")

            # Map holosoma DOF indices to MuJoCo actuator indices
            for i, dof_name in enumerate(self.dof_names):
                # Add prefix for MuJoCo actuator lookup (dof_names are clean, need prefixed version)
                actuator_name = self._get_prefixed_name(dof_name)
                actuator_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id == -1:
                    raise ValueError(f"Actuator for DOF '{dof_name}' (MuJoCo name: '{actuator_name}') not found")
                self.root_data.ctrl[actuator_id] = torques_np[i]

    def draw_debug_viz(self):
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def simulate_at_each_physics_step(self) -> None:
        """Advance simulation by one step."""

        if self.virtual_gantry:
            # Apply virtual gantry forces before step
            self.virtual_gantry.step()

        # Step bridge for updated torques before step using base class helper
        self._step_bridge()

        # Delegate simulation step to backend
        self.backend.step()

        # Call video recorder capture frame if recording is active
        if self.video_recorder and self.video_recorder.is_recording:
            self.capture_video_frame()

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """Get actor states using MuJoCo best practices with robot-only validation.

        Parameters
        ----------
        indices : ActorIndices
            Actor indices to get states for.

        Returns
        -------
        ActorStates
            Actor states tensor with shape [num_actors, 13] containing
            [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] for each actor.

        Raises
        ------
        NotImplementedError
            If non-robot objects are requested.
        RuntimeError
            If robot body ID exceeds model bodies.
        """
        assert self.root_model
        assert self.root_data

        resolved_objects = self.object_registry.resolve_indices(indices)
        all_states: list[torch.Tensor] = []
        for obj_name, env_ids in resolved_objects:
            # TODO: objects, scenes
            if obj_name != "robot":
                raise NotImplementedError(
                    f"MuJoCo simulator currently only supports robot state access. "
                    f"Object '{obj_name}' is not supported."
                )

            robot_body_id = 1  # FIXME: Assuming robot root is body 1 (after world body 0)

            assert self.root_model is not None
            if robot_body_id >= self.root_model.nbody:
                raise RuntimeError(f"Robot body ID {robot_body_id} exceeds model bodies {self.root_model.nbody}")

            # Use data.xpos/xquat for positions/orientations (global coordinates)
            pos = torch.from_numpy(self.root_data.xpos[robot_body_id]).float().to(self.sim_device)  # [3]
            quat_mj = self.root_data.xquat[robot_body_id]  # [w,x,y,z] MuJoCo format

            # Convert quaternion: MuJoCo [w,x,y,z] → holosoma [x,y,z,w]
            quat_holosoma = torch.tensor(
                [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]], device=self.sim_device, dtype=torch.float32
            )

            # Use data.cvel for velocities: angular velocity
            angular_velocity = torch.from_numpy(self.root_data.cvel[robot_body_id, 0:3]).float().to(self.sim_device)
            linear_velocity = torch.from_numpy(self.root_data.cvel[robot_body_id, 3:6]).float().to(self.sim_device)

            # Convert COM velocity to body origin velocity
            offset = (
                torch.from_numpy(self.root_data.xpos[robot_body_id] - self.root_data.subtree_com[robot_body_id])
                .float()
                .to(self.sim_device)
            )
            lin_world = linear_velocity + torch.cross(angular_velocity, offset)

            # Pack in holosoma format [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
            state = torch.cat([pos, quat_holosoma, lin_world, angular_velocity])  # [13]

            # Repeat for all requested environments (currently just 1)
            states_for_envs = state.unsqueeze(0).repeat(len(env_ids), 1)  # [num_envs, 13]
            all_states.append(states_for_envs)

        return torch.cat(all_states, dim=0) if all_states else torch.empty(0, 13, device=self.sim_device)

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """Set actor states using MuJoCo best practices with robot-only validation.

        Parameters
        ----------
        indices : ActorIndices
            Actor indices to set states for.
        states : ActorStates
            Actor states to set with shape [num_actors, 13].
        write_updates : bool
            Whether to apply forward kinematics after setting states.

        Raises
        ------
        NotImplementedError
            If non-robot objects are requested.
        RuntimeError
            If insufficient qpos or qvel elements.
        """
        assert self.root_data is not None

        resolved_objects = self.object_registry.resolve_indices(indices)

        state_offset = 0
        for obj_name, env_ids in resolved_objects:
            if obj_name != "robot":
                # TODO: objects, scenes
                raise NotImplementedError(
                    f"MuJoCo simulator currently only supports robot state setting. "
                    f"Object '{obj_name}' is not supported."
                )

            num_states = len(env_ids)
            obj_states = states[state_offset : state_offset + num_states]  # [num_envs, 13]
            state_offset += num_states

            # Set robot state for each environment
            for i, _ in enumerate(env_ids):
                state = obj_states[i]  # [13]

                pos = state[:3].detach().cpu().numpy()
                quat_holosoma = state[3:7].detach().cpu().numpy()  # [x,y,z,w]
                lin_vel = state[7:10].detach().cpu().numpy()
                ang_vel = state[10:13].detach().cpu().numpy()

                # Convert quaternion: holosoma [x,y,z,w] → MuJoCo [w,x,y,z]
                quat_mj = np.array([quat_holosoma[3], quat_holosoma[0], quat_holosoma[1], quat_holosoma[2]])

                # Set via freejoint (assuming robot has freejoint)
                # For single environment, we update qpos/qvel directly
                if len(self.root_data.qpos) >= 7:  # Ensure we have enough qpos elements
                    self.root_data.qpos[0:3] = pos  # Root position
                    self.root_data.qpos[3:7] = quat_mj  # Root orientation [w,x,y,z]
                else:
                    raise RuntimeError(f"Insufficient qpos elements: {len(self.root_data.qpos)}, need at least 7")

                if len(self.root_data.qvel) >= 6:  # Ensure we have enough qvel elements
                    self.root_data.qvel[0:3] = lin_vel  # Root linear velocity
                    self.root_data.qvel[3:6] = ang_vel  # Root angular velocity
                else:
                    raise RuntimeError(f"Insufficient qvel elements: {len(self.root_data.qvel)}, need at least 6")

        if write_updates:
            mujoco.mj_forward(self.root_model, self.root_data)

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """Get actor indices using ObjectRegistry with robot-only validation.

        Parameters
        ----------
        names : Union[str, ActorNames]
            Actor name(s) to get indices for.
        env_ids : Optional[EnvIds]
            Environment IDs to get indices for (None = all environments).

        Returns
        -------
        ActorIndices
            Actor indices for the specified names and environments.

        Raises
        ------
        NotImplementedError
            If non-robot objects are requested.
        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            # TODO: objects, scenes
            if name != "robot":
                raise NotImplementedError(
                    f"MuJoCo simulator currently only supports robot access. "
                    f"Requested object '{name}' is not supported. Only 'robot' is available."
                )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_initial_poses(self, names: list[str], env_ids: EnvIds | None = None) -> ActorPoses:
        """Get initial poses using ObjectRegistry with robot-only validation.

        Parameters
        ----------
        names : list[str]
            Actor names to get initial poses for.
        env_ids : Optional[EnvIds]
            Environment IDs to get poses for (None = all environments).

        Returns
        -------
        ActorPoses
            Initial poses for the specified actors and environments.

        Raises
        ------
        NotImplementedError
            If non-robot objects are requested.
        """
        for name in names:
            # TODO: objects, scenes
            if name != "robot":
                raise NotImplementedError(
                    f"MuJoCo simulator currently only supports robot initial poses. "
                    f"Requested object '{name}' is not supported. Only 'robot' is available."
                )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def write_state_updates(self) -> None:
        """Write state updates.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("WIP")

    def set_actor_root_state_tensor(self, set_env_ids: torch.Tensor | None, root_states: torch.Tensor | None) -> None:
        """Legacy compatibility method for LeggedRobotBase.

        This method provides backward compatibility with the existing LeggedRobotBase code
        that calls set_actor_root_state_tensor. It delegates to the robot-specific method.

        Parameters
        ----------
        set_env_ids : Optional[torch.Tensor]
            Which environments to update (None = all).
        root_states : Optional[torch.Tensor]
            Root states tensor (can be all_root_states or robot_root_states).
        """
        # Handle the case where all_root_states tensor is passed
        if root_states is not None and root_states is self.all_root_states:
            # Use robot states view directly
            self.set_actor_root_state_tensor_robots(set_env_ids, self.robot_root_states[set_env_ids])
        else:
            # Otherwise, assume it's already robot states
            self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_dof_state_tensor(self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None) -> None:
        """Legacy compatibility method for LeggedRobotBase.

        This method provides backward compatibility with the existing LeggedRobotBase code
        that calls set_dof_state_tensor. It delegates to the robot-specific method.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        dof_states : Optional[torch.Tensor]
            DOF states tensor (flattened IsaacGym format).
        """
        self.set_dof_state_tensor_robots(env_ids, dof_states)

    def set_actor_root_state_tensor_robots(
        self, env_ids: EnvIds | None = None, root_states: torch.Tensor | None = None
    ) -> None:
        """Set robot root states via backend delegation.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        root_states : Optional[torch.Tensor]
            Robot states to set. Can be either:
            - Pre-sliced tensor [len(env_ids), 13] matching env_ids
            - Full global tensor [num_envs, 13] (will be sliced automatically)
            Format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz].
            If None, uses current robot_root_states.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if root_states is None:
            root_states = self.robot_root_states[env_ids]
        # CRITICAL: Normalize calling convention - if caller passes full global tensor
        # but only updating subset of envs, slice it to match env_ids dimension
        elif len(root_states) != len(env_ids):
            if len(root_states) == self.num_envs:
                # Full global tensor provided, slice to match env_ids
                root_states = root_states[env_ids]
            else:
                raise ValueError(
                    f"root_states dimension mismatch: got {len(root_states)}, "
                    f"expected either {len(env_ids)} (pre-sliced) or {self.num_envs} (global)"
                )

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.num_dof == 0:
            logger.info("No robot DOFs available - skipping root state update")
            return

        # Delegate to backend
        root_addrs = {"robot_qpos_addr": self.robot_qpos_addr, "robot_qvel_addr": self.robot_qvel_addr}
        self.backend.set_root_state(env_ids, root_states, root_addrs)

    def set_dof_state_tensor_robots(
        self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None
    ) -> None:
        """Set robot DOF states via backend delegation.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        dof_states : Optional[torch.Tensor]
            DOF states to set. Format depends on tensor shape:
            - 2D [num_selected_envs, num_dofs, 2]: IsaacSim format [pos, vel] per DOF
            - 2D [num_selected_envs * num_dofs, 2]: IsaacGym flattened format
            If None, uses current dof_state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state  # type: ignore[assignment]

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.num_dof == 0:
            logger.info("No robot DOFs available - skipping DOF state update")
            return

        assert dof_states
        if dof_states.dim() != 2 and dof_states.shape[0] != len(env_ids) * self.num_dof:
            raise ValueError(
                f"Unsupported dof_states tensor format: {dof_states.shape}. "
                f"Expected [num_envs, num_dofs, 2] or [num_envs * num_dofs, 2]"
            )

        # Delegate to backend
        dof_addrs = {"dof_qpos_addrs": self.dof_qpos_addrs, "dof_qvel_addrs": self.dof_qvel_addrs}
        self.backend.set_dof_state(env_ids, dof_states, dof_addrs)

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get DOF limits properties - simplified IsaacSim pattern.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (dof_pos_limits, dof_vel_limits, torque_limits).
        """
        # Initialize tensors directly in method (like IsaacSim)
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)

        # Populate from robot config (like IsaacSim)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]

            # Apply soft limits (like IsaacSim)
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """Find rigid body index in body_names list.

        Parameters
        ----------
        body_name : str
            Name of the body to find.

        Returns
        -------
        int
            Index of the body in the body_names list.

        Raises
        ------
        RuntimeError
            If the body name is not found.
        """
        # Returns MuJoCo body ID that works with apply_force()
        prefixed_name = self._get_prefixed_name(body_name)
        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
        if body_id >= 0:
            return body_id

        raise RuntimeError(f"Body '{body_name}' not found in body_names: {self.body_names}")

    def setup_viewer(self) -> None:
        """Set up MuJoCo viewer using official mujoco.viewer API with keyboard callback."""
        logger.info("=== Setting up MuJoCo viewer ===")

        if self.headless:
            logger.info("Headless mode enabled - skipping viewer setup")
            self.viewer = None
            return

        self.viewer = mujoco.viewer.launch_passive(self.root_model, self.root_data, key_callback=self._key_callback)
        logger.info("=== Viewer setup completed with keyboard callback ===")

    def _add_text_overlay(
        self,
        text: str,
        font: int | None = None,
        gridpos: int | None = None,
        text2: str = "",
    ) -> None:
        """Add screen-space text overlay (HUD) to the MuJoCo viewer.

        This creates a fixed screen-space overlay that doesn't move with the camera,
        similar to a heads-up display (HUD).

        Parameters
        ----------
        text : str
            Primary text to display (left column).
        font : Optional[int]
            Font scale from mujoco.mjtFontScale enum. If None, uses default (150% scale).
            Options: mjFONTSCALE_50, mjFONTSCALE_100, mjFONTSCALE_150, etc.
        gridpos : Optional[int]
            Grid position from mujoco.mjtGridPos enum. If None, uses TOPLEFT.
            Options: mjGRID_TOPLEFT, mjGRID_TOPRIGHT, mjGRID_BOTTOMLEFT, mjGRID_BOTTOMRIGHT.
        text2 : str
            Secondary text to display (right column), defaults to empty string.
        """
        if self.viewer is None:
            return

        # Use the passive viewer's set_texts method for screen-space HUD overlay
        # Format: (font, gridpos, text1, text2)
        self.viewer.set_texts((font, gridpos, text, text2))

    def render(self, sync_frame_time: bool = True) -> None:
        """Render simulation to the viewer

        Parameters
        ----------
        sync_frame_time : bool
            Whether to synchronize frame time (currently unused).
        """
        if self.viewer is None:
            logger.warning("Cannot render, no viewer")
            return

        # Sync GPU -> CPU for WarpBackend with current world_id
        # (no-op for ClassicBackend which returns same data)
        self.root_data = self.backend.get_render_data(world_id=self.current_world_id)

        self.viewer.sync()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    def time(self) -> float:
        """Get current simulation time in seconds.

        Returns the MuJoCo simulation time, used for clock synchronization
        in sim2sim setups. This allows policies to stay synchronized with
        the simulation state.

        Returns
        -------
        float
            Current MuJoCo simulation time in seconds.
        """
        assert self.root_data is not None
        return self.root_data.time

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        """Get DOF forces for a specific environment.

        Returns actuator forces from MuJoCo's force sensors, providing
        measured joint forces for bridge system sim2sim force feedback.

        Parameters
        ----------
        env_id : int, default=0
            Environment index (currently only supports env 0).

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_dof] with measured joint forces, dtype torch.float32.

        Raises
        ------
        RuntimeError
            If multiple environments requested (not yet supported).
        """
        if env_id != 0:
            raise RuntimeError(f"MuJoCo classic currently only supports single environment (env_id=0), got {env_id}")

        assert self.root_data is not None
        return torch.from_numpy(self.root_data.actuator_force[: self.num_dof]).float().to(self.sim_device)

    def _update_text_overlay(self) -> None:
        """Update text overlay based on current state (event-driven).

        This method is called only when state changes occur (e.g., key presses),
        not on every render frame. This prevents the viewer's keyboard input
        system from being disrupted by frequent set_texts() calls.
        """
        if self.viewer is None:
            return

        if not self.show_text_overlay:
            # Clear text overlays when disabled
            self.viewer.set_texts([])
            return

        # Determine virtual gantry status
        if self.virtual_gantry and self.virtual_gantry.enabled:
            gantry_status = "active"
        else:
            gantry_status = "inactive"

        # Build text overlay content
        text = (
            f"Virtual gantry is {gantry_status} \n"
            "Press '7' to raise it \n"
            "Press '8' to lower it \n"
            "Press '9' to toggle it \n"
            "Press backspace to reset the environment \n"
            "Press 'g' to hide this menu"
        )

        # Use default font and position (None values will use MuJoCo defaults)
        self._add_text_overlay(text)

    def _key_callback(self, keycode: int) -> None:
        """Handle keyboard input with unified command registry and world_id toggling.

        Parameters
        ----------
        keycode : int
            GLFW keycode for the pressed key.
        """
        if self.commands is None:
            return

        # Handle text overlay toggle
        # G key (71): Toggle text overlay visibility
        if keycode == 71:  # 'G' key
            self.show_text_overlay = not self.show_text_overlay
            status = "ON" if self.show_text_overlay else "OFF"
            logger.info(f"Text overlay: {status}")
            # Update overlay immediately when toggled
            self._update_text_overlay()
            return

        # Handle world_id toggling for multi-environment visualization (WarpBackend only)
        # LEFT ARROW (263): Previous environment
        # RIGHT ARROW (262): Next environment
        # Numbers 0-9 (48-57): Jump to specific environment
        if self.num_envs > 1:
            if keycode == 263:  # LEFT ARROW - Previous environment
                self.current_world_id = (self.current_world_id - 1) % self.num_envs
                logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                return
            if keycode == 262:  # RIGHT ARROW - Next environment
                self.current_world_id = (self.current_world_id + 1) % self.num_envs
                logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                return
            if 48 <= keycode <= 57:  # Number keys 0-9
                requested_id = keycode - 48  # Convert keycode to number (0-9)
                if requested_id < self.num_envs:
                    self.current_world_id = requested_id
                    logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                else:
                    logger.warning(f"Environment {requested_id} does not exist (max: {self.num_envs - 1})")
                return

        # Use unified command registry
        if not hasattr(self, "_command_registry"):
            self._command_registry = CommandRegistry(self)
            # Register callback for UI updates on command execution
            self._command_registry.on_command_executed = self._update_text_overlay

        # Single call handles both gantry and robot commands
        if self._command_registry.execute_command(keycode):
            return  # Command handled

        # Log unhandled keys
        logger.debug(f"Unhandled keycode: {keycode}")

    def _zero_commands(self) -> None:
        """Zero all commands (Phase 1 helper method)."""
        if hasattr(self, "commands") and self.commands is not None:
            self.commands.fill_(0.0)
            logger.info("Zeroed all commands")

    def __del__(self) -> None:
        """Cleanup viewer on simulator destruction."""
        logger.info("=== MuJoCo Simulator Cleanup Started ===")
        if hasattr(self, "viewer") and self.viewer is not None:
            try:
                logger.info("Closing MuJoCo viewer")
                # Official mujoco.viewer handles cleanup automatically, set to None to release reference
                self.viewer = None
                logger.info("MuJoCo viewer reference released")
            except Exception as e:
                logger.warning(f"Error during viewer cleanup: {e}")
        logger.info("=== MuJoCo Simulator Cleanup Completed ===")

    def _update_contact_forces(self) -> None:
        """Update contact forces tensor using MuJoCo's canonical mj_contactForce() API.

        This method extracts contact forces from MuJoCo's contact detection system and
        accumulates them per body to match holosoma's expected interface.

        Key concepts:
        - MuJoCo contacts are detected between geoms (collision geometries)
        - Multiple geoms can belong to the same body (e.g., robot foot with multiple collision shapes)
        - holosoma expects forces per body, so we need to aggregate geom-level forces to body-level
        - mj_contactForce() returns the 6D force/torque that geom1 exerts on geom2
        - We only use the first 3 components (forces), ignoring torques for now

        Shape: self.contact_forces = [num_envs, num_bodies, 3] = [1, num_bodies, 3]
        """
        assert self.root_model
        assert self.root_data

        # Reset contact forces to zero before accumulating new forces
        # This is essential because we accumulate forces from multiple contacts per body
        self.contact_forces.fill_(0.0)

        # Early return if no contacts detected
        if self.root_data.ncon == 0:
            return

        # Temporary buffer for mj_contactForce() output: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        forcetorque = np.zeros(6, dtype=np.float64)

        # Iterate through all active contacts in the simulation
        # Each contact represents a collision between two geoms
        for contact_idx in range(self.root_data.ncon):
            contact = self.root_data.contact[contact_idx]

            # Extract the 6D force/torque vector for this contact using MuJoCo's canonical API
            # This gives us the force that geom1 exerts on geom2 at the contact point
            mujoco.mj_contactForce(self.root_model, self.root_data, contact_idx, forcetorque)

            # Extract only the force components (first 3 elements), ignoring torques
            contact_force = forcetorque[:3]  # [force_x, force_y, force_z]

            # Map geoms to their parent bodies using MuJoCo's geom_bodyid mapping
            # This is necessary because contacts are geom-level but holosoma expects body-level forces
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            mj_body1_id = self.root_model.geom_bodyid[geom1_id]
            mj_body2_id = self.root_model.geom_bodyid[geom2_id]

            # Map MuJoCo body IDs to holosoma indices using pre-built mapping
            holosoma_body1_idx = self.mujoco_to_holosoma_body_map.get(mj_body1_id)
            holosoma_body2_idx = self.mujoco_to_holosoma_body_map.get(mj_body2_id)

            # Contact logging is now handled centrally in legged_robot_base._log_contact_forces()

            # Apply Newton's 3rd law: mj_contactForce() result is geom1 exerts on geom2, so geom2's
            # body gets +force, geom1's body gets -force. Note: skips bodies not in our map
            if holosoma_body1_idx is not None:
                self.contact_forces[0, holosoma_body1_idx] -= (
                    torch.from_numpy(contact_force).float().to(self.sim_device)
                )
            if holosoma_body2_idx is not None:
                self.contact_forces[0, holosoma_body2_idx] += (
                    torch.from_numpy(contact_force).float().to(self.sim_device)
                )

    def print_mujoco_model_tree(self) -> None:
        """Print comprehensive MuJoCo model structure for debugging."""
        assert self.root_model
        assert self.root_data

        model_path = self.scene_manager.robot_model_path
        print(f"Analyzing compiled model (robot source: {model_path})")

        model = self.root_model  # Use compiled model instead of reloading from XML
        data = self.root_data  # Use existing data instead of creating new

        print("=" * 80)
        print("MUJOCO MODEL STRUCTURE ANALYSIS")
        print("=" * 80)

        # 1. BASIC MODEL INFO
        print("\n📊 MODEL OVERVIEW:")
        print(f"   Model file: {model_path}")
        print(f"   Total bodies: {model.nbody}")
        print(f"   Total joints: {model.njnt}")
        print(f"   Total DOFs: {model.nv}")
        print(f"   Total qpos elements: {model.nq}")
        print(f"   Total actuators: {model.nu}")
        print(f"   Total geoms: {model.ngeom}")

        # 2. BODY LIST (Simple, no hierarchy to avoid infinite loops)
        print("\n🏗️  BODY LIST:")
        print(f"   {'ID':<3} {'Name':<30} {'Parent ID':<9} {'Parent Name'}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 9} {'-' * 20}")

        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            parent_id = model.body_parentid[body_id]
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id != -1 else "WORLD"
            print(f"   {body_id:<3} {body_name:<30} {parent_id:<9} {parent_name}")

        # 3. JOINT DETAILS (This is the most important part!)
        print("\n🔗 JOINT STRUCTURE:")
        print(f"   {'ID':<3} {'Name':<30} {'Type':<8} {'Body':<20} {'qpos_addr':<9} {'qvel_addr':<9}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 8} {'-' * 20} {'-' * 9} {'-' * 9}")

        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            joint_type = model.jnt_type[joint_id]
            body_id = model.jnt_bodyid[joint_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]

            # Joint type names
            type_names = {0: "FREE", 1: "BALL", 2: "SLIDE", 3: "HINGE"}
            type_name = type_names.get(joint_type, f"TYPE_{joint_type}")

            print(f"   {joint_id:<3} {joint_name:<30} {type_name:<8} {body_name:<20} {qpos_addr:<9} {qvel_addr:<9}")

        # 4. DOF ANALYSIS (What holosoma expects)
        print("\n🎯 DOF ANALYSIS (holosoma perspective):")

        # Get all non-freejoint joints
        dof_joints = []
        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            joint_type = model.jnt_type[joint_id]

            # Skip freejoint (type 0) and floating_base joints
            if joint_type != 0 and "floating_base" not in joint_name.lower():
                dof_joints.append((joint_id, joint_name))

        print(f"   Expected DOF count: {len(dof_joints)}")
        print(f"\n   {'Idx':<3} {'DOF Name':<30} {'MJ_ID':<5} {'qpos_addr':<9} {'qvel_addr':<9}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 5} {'-' * 9} {'-' * 9}")

        for idx, (joint_id, joint_name) in enumerate(dof_joints):
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]
            print(f"   {idx:<3} {joint_name:<30} {joint_id:<5} {qpos_addr:<9} {qvel_addr:<9}")

        # 5. ACTUATOR MAPPING
        print("\n⚙️  ACTUATOR MAPPING:")
        print(f"   {'ID':<3} {'Name':<30} {'Joint':<30}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 30}")

        for actuator_id in range(model.nu):
            actuator_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"actuator_{actuator_id}"
            )
            # Get the joint this actuator controls
            joint_id = model.actuator_trnid[actuator_id, 0]  # First transmission element
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            print(f"   {actuator_id:<3} {actuator_name:<30} {joint_name:<30}")

        # 6. CURRENT STATE SNAPSHOT
        print("\n📸 CURRENT STATE SNAPSHOT:")
        print(f"   qpos (first 10): {data.qpos[:10]}")
        print(f"   qvel (first 10): {data.qvel[:10]}")
        print(f"   ctrl (all): {data.ctrl}")

        print("\n" + "=" * 80)
        print("END OF MODEL ANALYSIS")
        print("=" * 80)
