"""MuJoCo simulator implementation.

The simulator follows the BaseSimulator interface while providing MuJoCo-specific
implementations for terrain rendering, contact detection, and physics simulation, etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, cast

import mujoco
import mujoco.viewer
import torch
from loguru import logger

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.sensor import CameraSensorConfig, LidarSensorConfig
from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.mujoco import light_setup
from holosoma.simulator.mujoco.backends import WARP_AVAILABLE, ClassicBackend, IMujocoBackend, WarpBackend
from holosoma.simulator.mujoco.backends.base import apply_sensor_scene_flags, mj_to_holosoma_quat
from holosoma.simulator.mujoco.fields import prepare_fields, prepare_manager_fields
from holosoma.simulator.mujoco.model_debug import print_mujoco_model_tree
from holosoma.simulator.mujoco.robot_addressing import (
    apply_dof_properties_from_config,
    apply_initial_joint_angles,
    build_joint_name_map,
    build_name_maps,
    dof_address_index,
    read_dof_limits,
    resolve_actor_addressing,
    resolve_dof_addresses,
    resolve_robot_body_ids,
    validate_dof_actuators,
    validate_dof_addressing,
)
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
from holosoma.simulator.mujoco.sensor_setup import create_sensors, offset_world_cameras, render_sensors
from holosoma.simulator.mujoco.tensor_views import (
    create_base_linear_acceleration_view,
)
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.mujoco.viewer_input import ViewerInputController
from holosoma.simulator.shared.external_wrench import WrenchTarget
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.shared.root_states import robot_rows, root_pose_changed
from holosoma.simulator.shared.root_states_view import UnifiedRootStatesView
from holosoma.simulator.shared.scene_types import EnvOriginsScene
from holosoma.simulator.shared.virtual_gantry import create_virtual_gantry
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.adapters import mujoco_draw_adapter
from holosoma.utils.robot_names import configure_robot_names
from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse


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
        self.hooks.add(Phase.CLOSE, self._close_backend, name="mujoco.backend.close")
        self.hooks.add(Phase.CLOSE, self._close_viewer, name="mujoco.viewer.close")

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

        # Physics backend (WarpBackend or ClassicBackend); assigned in load_assets.
        self.backend: IMujocoBackend

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

        # World ID for multi-environment visualization (which environment to view).
        # Mutated by the viewer input controller; read by render() to pick which world to show.
        self.current_world_id: int = 0

        # Interactive-viewer keyboard/HUD handler; created in setup_viewer (headful only).
        self.viewer_input: ViewerInputController | None = None

        # Command system for keyboard/joystick controls
        # Initialize commands tensor matching IsaacGym format:
        #    [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        # Shape: [num_envs, 9] to match IsaacGym command structure
        self.commands: torch.Tensor | None = None  # Will be initialized in create_envs when num_envs is known

        logger.info("=== MuJoCo Simulator Initialization Completed ===")

    def _build_name_maps(self) -> None:
        """Build bidirectional name maps for clean <-> prefixed name translation.

        Creates mapping dictionaries to translate between clean names (used by holosoma)
        and prefixed names (used internally by MuJoCo) for the ROBOT's joints, bodies,
        and actuators.

        Robot elements are identified from the per-actor spec metadata captured at
        spawn (``scene_manager.robot_spec_meta``), NOT by ``name.startswith(prefix)``.
        """
        meta = self.scene_manager.robot_spec_meta
        assert meta is not None, "robot_spec_meta must be captured in scene_manager.add_robot"
        configure_robot_names(self, list(meta.dof_joint_names), list(meta.body_names), root_body=meta.root_body)
        self.joint_name_map = build_joint_name_map(meta, self.robot_config)
        self.body_name_map = {name: f"{meta.prefix}{name}" for name in self.body_names}
        self.clean_to_prefixed_names, self.prefixed_to_clean_names = build_name_maps(meta)
        self.clean_to_prefixed_names.update(self.joint_name_map)
        self.clean_to_prefixed_names.update(self.body_name_map)
        self.prefixed_to_clean_names.update({v: k for k, v in self.joint_name_map.items()})
        self.prefixed_to_clean_names.update({v: k for k, v in self.body_name_map.items()})

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

    def get_supported_scene_formats(self) -> list[str]:
        """See base class.

        MuJoCo-specific notes:
        - Supports XML / MCJF as the preferred format.
        - Also supports URDF, but MuJoCo parsing of URDF is less rich than the native xml/mcjf

        Returns
        -------
        List[str]
            ["xml", "urdf"]
        """
        return ["xml", "urdf"]

    def get_mujoco_backend_type(self) -> MujocoBackend:
        """Which MuJoCo physics backend this sim uses: ``MujocoBackend.WARP`` (GPU, multi-env)
        or ``MujocoBackend.CLASSIC`` (CPU, single-env).

        Parity with ``get_simulator_type()``, which returns ``MUJOCO`` for both backends and so
        can't make this distinction.
        """
        return self.simulator_config.mujoco_backend

    def load_assets(self) -> None:
        """Load assets using compositional MjSpec approach.

        Creates the scene manager, sets up the scene components (terrain, lighting,
        materials, robot), compiles the final model, and initializes robot properties
        and joint addressing for simulation.
        """
        logger.info("=== Loading assets ===")

        # Create scene manager
        self.scene_manager = MujocoSceneManager(self.simulator_config)  # type: ignore[arg-type]
        self._setup_scene()

        # Compile once at the end
        self.root_model = self.scene_manager.compile()
        self.root_data = mujoco.MjData(self.root_model)

        # Apply post-compilation settings
        self.root_model.opt.timestep = self.sim_dt

        # Resolve per-actor freejoint addressing (depends only on the compiled model + scene config).
        # Runs before backend creation so the WarpBackend's put_model sees the final model.
        self._set_object_addressing()

        # Config per-DOF armature/friction override the MJCF values; also before the backend takes the model.
        meta = self.scene_manager.robot_spec_meta
        assert meta is not None, "robot_spec_meta must be captured in scene_manager.add_robot"
        apply_dof_properties_from_config(self.root_model, meta, self.robot_config)
        if self.robot_config.dof_armature is not None:
            # Armature changes the reference inertia and constraint weights. Rebuild these
            # constants before either backend uses them or Warp copies the model to the GPU.
            mujoco.mj_setConst(self.root_model, self.root_data)

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
        self.virtual_gantry.register_hooks(self.hooks)

        # Initialize bridge system using base class helper
        self._init_bridge()

        # Create mounted camera sensors (cameras were added to the spec pre-compile in
        # _setup_scene; this resolves their compiled ids and renderers). No-op if none.
        self._create_sensors()

        if self.video_config.enabled:
            self.video_recorder = MuJoCoVideoRecorder(self.video_config, self)
            self.video_recorder.register_hooks(self.hooks)
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
            self.scene_manager.apply_render()
            light_setup.build_lights(self.scene_manager.world_spec, self.scene_config.lights)
            self.scene_manager.add_materials()

        # Always add robot after terrain, in case it references ground/floor, etc for contacts
        self.scene_manager.add_robot(
            terrain_state, self.robot_config, xml_filter=self.simulator_config.robot_mjcf_filter
        )

        # Add individual free rigid bodies from the scene config
        supported_formats = self.get_supported_scene_formats()
        for name, obj in self.scene_config.rigid_objects.items():
            self.scene_manager.add_rigid_object(
                name,
                obj,
                supported_formats,
                xml_filter=self.simulator_config.robot_mjcf_filter,
                scene_asset_root=self.scene_config.asset_root,
            )

        # Add multi-body scene files (1->N): each file expands to N free/static bodies.
        for scene_file_name, scene_file in self.scene_config.scene_files.items():
            self.scene_manager.add_scene_file(
                scene_file_name,
                scene_file,
                supported_formats,
                xml_filter=self.simulator_config.robot_mjcf_filter,
                scene_asset_root=self.scene_config.asset_root,
            )

        # Add native camera/site elements after all mount bodies exist but before compile.
        cameras = {
            name: config for name, config in self.sensor_config.items() if isinstance(config, CameraSensorConfig)
        }
        lidars = {name: config for name, config in self.sensor_config.items() if isinstance(config, LidarSensorConfig)}
        self.scene_manager.add_cameras(cameras)
        self.scene_manager.add_lidar_sites(lidars)

    def _set_robot_properties(self) -> None:
        """Set robot properties including DOF names, body names, and index mappings.

        Robot elements (DOF joints, bodies) are identified from the per-actor spec
        metadata captured at spawn (``scene_manager.robot_spec_meta``), which holds
        exactly the robot's own named elements.

        ``body_names`` is the holosoma-facing, 0-based robot-body list (world body and
        all non-robot bodies excluded). ``body_ids`` is the matching list of raw
        MuJoCo body ids, used to gather robot rows out of the full, ``model.nbody``-wide
        physics tensors (xpos/cfrc/...) in ``refresh_sim_tensors``, so physics-tensor
        width stays independent of ``num_bodies``.
        """
        assert self.root_model
        meta = self.scene_manager.robot_spec_meta
        assert meta is not None, "robot_spec_meta must be captured in scene_manager.add_robot"

        # Build clean<->prefixed name maps from the recorded robot elements first.
        self._build_name_maps()

        # Map each robot body (by prefixed name) to its compiled MuJoCo body id,
        # in body_names order. This is the cross-backend ``body_ids`` mapping
        # (body_ids[holosoma_idx] -> backend body id; mirrors IsaacGym/IsaacSim) used to
        # map a 0-based body_names index to the full-model xfrc/applied_forces layout
        # (e.g. virtual gantry). The tensor form gathers robot rows out of the
        # full-model physics tensors in refresh_sim_tensors.
        self.body_ids = resolve_robot_body_ids(self.root_model, self.body_names, self.body_name_map)
        self._body_ids_t = torch.tensor(self.body_ids, dtype=torch.long, device=self.sim_device)
        self.backend.configure_rigid_body_refresh(self.body_ids)

        logger.info(f"Total joints: {self.root_model.njnt}, Robot DOFs: {self.num_dof}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")
        logger.info(f"Robot body ids (mujoco): {self.body_ids}")

    def _set_robot_joint_addressing(self) -> None:
        """Store qpos/qvel addresses for the robot's actuated DOF joints.

        Root (freejoint) addressing is handled uniformly for all actors in
        _set_object_addressing; this covers only the robot-specific DOF joints.

        Then enforces the articulation contract the state views and torque paths rely on:
        distinct DOF qpos/qvel addresses and one pure torque motor per DOF,
        resolved by name into ``dof_actuator_ids`` (the ctrl columns ``apply_torques_at_dof``
        writes; see ``robot_addressing``).
        """
        logger.info("=== Setting up robot DOF joint addressing ===")
        assert self.root_model
        self.dof_qpos_addrs, self.dof_qvel_addrs = resolve_dof_addresses(
            self.root_model, self.dof_names, self.joint_name_map
        )
        validate_dof_addressing(self.dof_qpos_addrs, self.dof_qvel_addrs)
        meta = self.scene_manager.robot_spec_meta
        assert meta is not None, "robot_spec_meta must be captured in scene_manager.add_robot"
        self.dof_actuator_ids = validate_dof_actuators(self.root_model, meta, self.dof_names, self.joint_name_map)
        # Warp ctrl column index for the full-width write; None when the DOF actuators are ctrl's columns in
        # order (every shipped robot), where the write is a plain slice assignment.
        self._dof_actuator_ids_t: torch.Tensor | None = (
            None
            if self.dof_actuator_ids == list(range(self.root_model.nu))
            else torch.tensor(self.dof_actuator_ids, dtype=torch.long, device=self.sim_device)
        )
        logger.info("=== Robot joint addressing setup completed ===")

    @property
    def scene_file_static_names(self) -> set[str]:
        """Names of scene-file (1->N) bodies the file marked static. Derived from
        ``scene_manager.scene_file_bodies``. This exposes just the static-classification
        view that every backend shares."""
        return {name for name, (_root, is_static) in self.scene_manager.scene_file_bodies.items() if is_static}

    def _set_object_addressing(self) -> None:
        """Build per-actor addressing for the unified get/set_actor_states path.

        Free actors (robot + free rigid bodies) get ``object_addrs[name] =
        {qpos_addr, qvel_addr}``, the freejoint's 7-dof pose / 6-dof velocity slices.
        Static (fixed, jointless) objects have no qpos slice, so they get
        ``static_object_body_ids[name] = body_id`` and their pose is read from xpos.

        Each actor is resolved by its exact root-body name (not prefix-startswith,
        which collides for prefixes like "a_" vs "a_b_").
        """
        assert self.root_model
        # Static (SCENE) actors: standalone ``fixed`` objects + scene-file bodies the file
        # marked static. Free/static must be decided the same way as in every backend.
        static_names = {
            name for name, obj in self.scene_config.rigid_objects.items() if obj.fixed
        } | self.scene_file_static_names
        # A fixed-base robot is welded like a static object: no freejoint, pose read from xpos.
        # Config-side twin of the registry's is_static flag (addressing runs before registration).
        if self.robot_config.asset.fix_base_link:
            static_names |= {"robot"}
        self.object_addrs, self.static_object_body_ids = resolve_actor_addressing(
            self.root_model, self.scene_manager, static_names
        )

        # Raw MuJoCo body id of each OBJECT actor's root body, for external-force application
        # (xfrc_applied is full-model-width, indexed by raw body id; robot bodies are addressed
        # separately via self.body_ids).
        object_roots = {
            **self.scene_manager.rigid_object_root_bodies,
            **{name: root for name, (root, _is_static) in self.scene_manager.scene_file_bodies.items()},
        }
        self._object_root_body_ids: dict[str, int] = {}
        for name, root_body in object_roots.items():
            obj_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, root_body)
            if obj_body_id == -1:
                raise ValueError(f"Root body '{root_body}' for actor '{name}' not found in model.")
            self._object_root_body_ids[name] = obj_body_id

    def _collect_spawned_actors(
        self,
    ) -> tuple[torch.Tensor, list[tuple[str, bool, torch.Tensor, torch.Tensor | None]]]:
        """MuJoCo stores WORLD poses. Every actor, the robot included, is described at its
        per-env world pose with env_origins added (the uniform get_actor_initial_poses frame);
        prepare_sim places each scene object from that pose via its own mechanism (free:
        freejoint qpos; static: model body_pos, since a welded body has no qpos), while the
        robot is placed by the task layer at reset. A scene-file body's composed in-file pose
        already lives in the compiled model. Velocity is None for fixed/scene-file bodies.
        """
        robot_pose = self._object_pose(
            self.robot_config.init_state.position, self.robot_config.init_state.orientation, add_origins=True, wxyz=True
        )

        items = [
            (
                name,
                obj.fixed,
                self._object_pose(obj.position, obj.orientation, add_origins=True, wxyz=True),
                self._object_velocity(obj.linear_velocity, obj.angular_velocity),
            )
            for name, obj in self.scene_config.rigid_objects.items()
        ]
        for actor_name, (_root, is_static) in self.scene_manager.scene_file_bodies.items():
            pos, quat_wxyz = self._scene_file_body_world_pose(actor_name, is_static)
            items.append((actor_name, is_static, self._object_pose(pos, quat_wxyz, add_origins=True, wxyz=True), None))

        return robot_pose, items

    def _scene_file_body_world_pose(self, actor_name: str, is_static: bool) -> tuple[list[float], list[float]]:
        """Composed world pose (pos[3], quat wxyz[4]) of a scene-file body from the model.

        The file was attached at its configured world pose, so MjSpec already composed
        that with the body's authored relative pose. A free body's composed pose is its
        freejoint qpos0 (wxyz); a static body's is its post-forward xpos/xquat (wxyz).
        """
        assert self.root_model and self.root_data
        if is_static:
            mujoco.mj_forward(self.root_model, self.root_data)
            body_id = self.static_object_body_ids[actor_name]
            return list(self.root_data.xpos[body_id]), list(self.root_data.xquat[body_id])
        qpos_addr = self.object_addrs[actor_name]["qpos_addr"]
        qpos0 = self.root_model.qpos0
        return list(qpos0[qpos_addr : qpos_addr + 3]), list(qpos0[qpos_addr + 3 : qpos_addr + 7])

    def _object_pose(
        self,
        position: Sequence[float],
        orientation: Sequence[float],
        add_origins: bool,
        wxyz: bool = False,
    ) -> torch.Tensor:
        """Build a [num_envs, 7] world pose (xyzw quat) from a config position/orientation."""
        poses = torch.zeros(self.num_envs, 7, device=self.sim_device)
        poses[:, :3] = torch.tensor(position, device=self.sim_device)
        if add_origins:
            poses[:, :3] += self.env_origins
        if wxyz:
            w, x, y, z = orientation
            poses[:, 3:7] = torch.tensor([x, y, z, w], device=self.sim_device)  # wxyz -> xyzw
        else:
            poses[:, 3:7] = torch.tensor(orientation, device=self.sim_device)  # already xyzw
        return poses

    def _object_velocity(self, linear_velocity: Sequence[float], angular_velocity: Sequence[float]) -> torch.Tensor:
        """Build a [num_envs, 6] world-frame initial velocity [vx,vy,vz,wx,wy,wz] from config."""
        vel = torch.tensor([*linear_velocity, *angular_velocity], device=self.sim_device, dtype=torch.float32)
        return vel.unsqueeze(0).expand(self.num_envs, 6)

    def _set_object_initial_states(self) -> None:
        """Write each rigid object's initial pose into its freejoint qpos.

        Mirrors _set_robot_initial_state: objects are attached at a unit site, so
        their world pose is set here via qpos after compile + addressing. Velocity
        defaults to zero (RigidObjectConfig carries no initial velocity). Requires
        object_addrs (call after _set_object_addressing).
        """
        assert self.root_data
        for name, obj in self.scene_config.rigid_objects.items():
            if obj.fixed:
                continue  # static body: no freejoint qpos slice; pose fixed at spawn frame
            qpos_addr = self.object_addrs[name]["qpos_addr"]
            w, x, y, z = obj.orientation  # config is wxyz; MuJoCo qpos quat is wxyz too
            self.root_data.qpos[qpos_addr : qpos_addr + 3] = obj.position
            self.root_data.qpos[qpos_addr + 3 : qpos_addr + 7] = [w, x, y, z]

    def _set_initial_joint_angles(self) -> None:
        """Set initial joint angles from robot configuration.

        Applies the default joint positions specified in the robot configuration
        to the MuJoCo model's initial state, then performs forward kinematics
        to update body positions.
        """
        assert self.root_model
        assert self.root_data
        apply_initial_joint_angles(
            self.root_model,
            self.root_data,
            self.robot_config.default_dof_positions(),
            self.joint_name_map,
        )

    def create_envs(  # type: ignore[override]
        self, num_envs: int, env_origins: torch.Tensor, base_init_state: torch.Tensor
    ) -> None:
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
        self.scene = EnvOriginsScene(self.env_origins, self.sim_device)

        # World-mount cameras are worldbody children, so (unlike body-attached cameras) they don't
        # inherit the per-env body offset; place each env's world camera at its own origin now that
        # env_origins exist. WarpBackend only (Classic is single-env); the backend no-ops if it has
        # no world cameras or num_envs==1.
        self._offset_world_cameras()

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
        assert self.robot_config  # type: ignore[truthy-bool]

        if self.robot_config.asset.fix_base_link:
            # Fixed base: no freejoint qpos/qvel. The pose is baked into the attach frame and the
            # per-env spread in prepare_sim places it via set_static_body_world_pose.
            return

        # Set complete initial robot state (position, orientation, velocities)
        init_state = self.robot_config.init_state
        initial_pos = init_state.position
        initial_lin_vel = init_state.lin_vel
        initial_ang_vel = init_state.ang_vel
        # RobotInitState.orientation is [w,x,y,z], MuJoCo's qpos quaternion convention.
        initial_rot_mj = list(init_state.orientation)

        # Robot freejoint qpos/qvel slice (single source of truth: object_addrs)
        qpos_addr = self.object_addrs["robot"]["qpos_addr"]
        qvel_addr = self.object_addrs["robot"]["qvel_addr"]

        # Set position: [x, y, z, qw, qx, qy, qz] (7 elements)
        self.root_data.qpos[qpos_addr : qpos_addr + 3] = initial_pos
        self.root_data.qpos[qpos_addr + 3 : qpos_addr + 7] = initial_rot_mj

        # Set velocity: [vx, vy, vz, wx, wy, wz] (6 elements)
        self.root_data.qvel[qvel_addr : qvel_addr + 3] = initial_lin_vel
        self.root_data.qvel[qvel_addr + 3 : qvel_addr + 6] = initial_ang_vel

    def _create_floating_root_views(self) -> None:
        """Bind robot_root_states / base_quat / base_angular_vel / base_linear_acc to the robot's
        freejoint qpos/qvel/qacc slices (floating base only; a fixed base has no freejoint)."""
        assert self.root_data

        # Calculate indices for robot freejoint components
        robot_qpos_addr = self.object_addrs["robot"]["qpos_addr"]
        robot_qvel_addr = self.object_addrs["robot"]["qvel_addr"]
        pos_indices = slice(robot_qpos_addr, robot_qpos_addr + 3)
        quat_indices = slice(robot_qpos_addr + 3, robot_qpos_addr + 7)
        vel_indices = slice(robot_qvel_addr, robot_qvel_addr + 3)
        ang_vel_indices = slice(robot_qvel_addr + 3, robot_qvel_addr + 6)

        # Create robot root states proxy via backend factory
        root_addrs = {
            "pos_indices": pos_indices,
            "quat_indices": quat_indices,
            "vel_indices": vel_indices,
            "ang_vel_indices": ang_vel_indices,
        }
        self.robot_root_states = self.backend.create_root_view(root_addrs)

        # Create base_quat, base_angular_vel, base_linear_acc views via backend
        self.base_quat = self.backend.create_quaternion_view(quat_indices)
        self.base_angular_vel = self.backend.create_angular_velocity_view(ang_vel_indices)

        # Base linear acceleration: backend-specific handling
        base_lin_acc_indices = slice(0, 3)
        if isinstance(self.backend, WarpBackend):
            # WarpBackend: direct GPU tensor access
            self.base_linear_acc = self.backend.qacc_t[:, base_lin_acc_indices]
        else:
            # ClassicBackend: use view system
            self.base_linear_acc = create_base_linear_acceleration_view(
                qacc_array=self.root_data.qacc,
                indices=base_lin_acc_indices,
                num_envs=self.num_envs,
                device=self.sim_device,
            )

    def prepare_sim(self) -> None:
        """Prepare simulation - enhanced implementation with ObjectRegistry integration.

        Resets simulation data, sets initial robot state, configures the object registry,
        and creates tensor views for efficient state access during simulation.
        """
        # Reset simulation data
        assert self.root_data
        mujoco.mj_resetData(self.root_model, self.root_data)

        self._set_robot_initial_state()
        self._set_object_initial_states()

        # Robot + individual free bodies (objects already attached in load_assets, so the
        # spawn hook is a no-op; this only does the registry bookkeeping + poses).
        self._register_scene_assets()

        if self.robot_config.asset.fix_base_link:
            # Fixed base: no freejoint to view. The root state is a plain tensor holding the
            # registered spawn pose (init pose + env origins, exactly what the static spread below
            # applies) with zero velocity — what get_actor_states reports for a static body. It is
            # refreshed on the static robot teleport path (set_actor_states_by_index).
            self.robot_root_states = torch.zeros(self.num_envs, 13, device=self.sim_device)
            self.robot_root_states[:, :7] = self.get_actor_initial_poses(["robot"])
            self.base_quat = self.robot_root_states[:, 3:7]
            self.base_angular_vel = self.robot_root_states[:, 10:13]
            self.base_linear_acc = torch.zeros(self.num_envs, 3, device=self.sim_device)
        else:
            self._create_floating_root_views()

        # Unified all-actors view; indexing routes through get/set_actor_states_by_index,
        # which resolve each actor's freejoint slice (or static-body kinematics). MuJoCo has
        # no contiguous all-actors root tensor. robot_root_states remains the robot-only view
        # that set_actor_root_state_tensor uses (it detects this proxy by identity).
        self.all_root_states = UnifiedRootStatesView(self)

        # Calculate indices for DOF positions and velocities
        dof_pos_indices = dof_address_index(self.dof_qpos_addrs)
        dof_vel_indices = dof_address_index(self.dof_qvel_addrs)
        dof_acc_indices = dof_vel_indices

        # Create DOF state proxy via backend factory
        dof_addrs = {"dof_pos_indices": dof_pos_indices, "dof_vel_indices": dof_vel_indices}
        self.dof_state = self.backend.create_dof_state_view(dof_addrs, self.num_dof)

        # Create individual DOF views via backend factories
        self.dof_pos = self.backend.create_dof_pos_view(dof_pos_indices, self.num_dof)
        self.dof_vel = self.backend.create_dof_vel_view(dof_vel_indices, self.num_dof)
        self.dof_acc = self.backend.create_dof_acc_view(dof_acc_indices, self.num_dof)

        # contact_forces stays the robot-only, num_bodies-wide tensor allocated in
        # create_envs; refresh_sim_tensors gathers robot rows into it each frame. It is
        # intentionally not bound to the backend's full-model-width force view.

        # Create unified applied forces accessor for external force application (e.g., virtual gantry)
        self.applied_forces = self.backend.get_applied_forces_view()

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

        # The mj_resetData above zeroed the derived fields (xpos/xquat/cvel), and the
        # initial state was written only into qpos. Run a forward so get_actor_states
        # and the rigid-body views reflect the initial qpos before the first step.
        assert self.root_data
        mujoco.mj_forward(self.root_model, self.root_data)

        # WarpBackend.initialize_state (the only CPU->GPU sync) runs in load_assets,
        # before the initial pose / joint-angle qpos writes that happen here. Re-sync
        # the CPU state to the GPU so every env starts from the configured initial
        # state rather than the MJCF qpos0 defaults still on the GPU.
        if isinstance(self.backend, WarpBackend):
            self.backend.initialize_state(self.root_model, self.root_data)

        # initialize_state tiles one CPU qpos/model across all envs, so every env's objects sit
        # at the SAME world point. Spread each object to its env origin per its registered
        # per-env pose: free bodies via their freejoint qpos (set_actor_state), static (welded,
        # no qpos) bodies via their model body_pos (set_static_body_world_pose): the two
        # complementary placement paths, both keyed off get_actor_initial_poses.
        env_ids = torch.arange(self.num_envs, device=self.sim_device)

        free_names = self.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
        if free_names:
            poses = self.get_actor_initial_poses(free_names, env_ids)  # [n*num_envs, 7]
            vels = self.get_actor_initial_velocities(free_names, env_ids)  # [n*num_envs, 6]
            self.set_actor_states(free_names, env_ids, torch.cat([poses, vels], dim=1))

        # Spread static scene bodies while preserving their compiled, authored orientations.
        static_names = [
            n for n in self.object_registry.static_names() if n != "robot" and n in self.static_object_body_ids
        ]
        if static_names:
            body_ids = [self.static_object_body_ids[n] for n in static_names]
            # [num_envs, n_static, 3] world positions (env origins already applied at registration).
            positions = (
                self.get_actor_initial_poses(static_names, env_ids)[:, :3]
                .view(len(static_names), self.num_envs, 3)
                .transpose(0, 1)
            )
            self.backend.set_static_body_world_pose(body_ids, positions)  # position-only at spawn

        if self.robot_config.asset.fix_base_link:
            # The robot's configured pose is absolute, including orientation. Match the root-state
            # mirror instead of retaining an authored root rotation composed by the attach frame.
            poses = self.get_actor_initial_poses(["robot"], env_ids).view(self.num_envs, 1, 7)
            self.backend.set_static_body_world_pose(
                [self.static_object_body_ids["robot"]], poses[..., :3], poses[..., 3:7]
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

    def prepare_manager_fields(self, **managers: Any) -> None:
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

        # Permuted Warp DOFs use gathered buffers; native-order and Classic views read the engine directly.
        if isinstance(self.backend, WarpBackend):
            self.backend.refresh_dof_views()

        positions, orientations, linear_vel, angular_vel = self.backend.refresh_rigid_body_states()
        self._rigid_body_pos.copy_(positions)
        self._rigid_body_rot.copy_(orientations)
        self._rigid_body_vel.copy_(linear_vel)
        self._rigid_body_ang_vel.copy_(angular_vel)

        # Contact forces: backend returns full-model-width [num_envs, nbody, 3];
        # gather robot rows and rotate the rolling history (newest at index 0).
        if hasattr(self, "contact_forces_history") and hasattr(self, "contact_forces"):
            full_forces = self.backend.compute_contact_forces()  # [num_envs, nbody, 3]
            self.contact_forces[:] = full_forces[:, self._body_ids_t]
            self.contact_forces_history[:] = torch.cat(
                [self.contact_forces.unsqueeze(1), self.contact_forces_history[:, :-1]], dim=1
            )

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        """Clear contact forces history for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Tensor of environment IDs to clear history for.
        """
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        """Write efforts in canonical DOF order, or in ``dof_indices`` order for a subset.

        Unselected actuator controls are preserved, including actuators outside the robot.
        ClassicBackend raises ``ValueError`` if the effort count differs from the selected DOF count.
        """
        assert self.root_model
        assert self.root_data

        if self.root_model.nu == 0:
            logger.warning("No actuators found in MuJoCo model")
            return

        # Canonical DOFs may have different native actuator columns on either backend.
        ctrl_tensor = self.backend.get_ctrl_tensor()

        if ctrl_tensor is not None:
            # Fast path: Direct zero-copy write (WarpBackend). ctrl is [num_envs, nu].
            if dof_indices is None:
                if self._dof_actuator_ids_t is None:
                    ctrl_tensor[:] = torques  # DOF actuators are ctrl's columns in order
                else:
                    ctrl_tensor[:, self._dof_actuator_ids_t] = torques.to(ctrl_tensor.dtype)
            else:
                act_ids = [self.dof_actuator_ids[dof_idx] for dof_idx in dof_indices]
                idx_t = torch.as_tensor(act_ids, device=ctrl_tensor.device, dtype=torch.long)
                ctrl_tensor[:, idx_t] = torques.to(ctrl_tensor.dtype)
        else:
            # Slow path: addressed NumPy write (ClassicBackend). Full path -> every DOF in order.
            target_dofs = list(range(self.num_dof)) if dof_indices is None else dof_indices
            torques_np = torques.detach().cpu().numpy().flatten()

            # Verify torque count matches the DOFs being written.
            if len(torques_np) != len(target_dofs):
                raise ValueError(f"Torque count mismatch: got {len(torques_np)}, expected {len(target_dofs)}")

            for torque, dof_idx in zip(torques_np, target_dofs):
                self.root_data.ctrl[self.dof_actuator_ids[dof_idx]] = torque

    def draw_debug_viz(self) -> None:
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def _step_dynamics(self) -> None:
        """Advance simulation by one physics step."""
        # Must run before backend.step() consumes xfrc_applied.
        self.flush_external_wrench()
        self.backend.step()

    def forward_kinematics(self) -> None:
        """Recompute body/geom/camera poses from written state (classic: ``mj_forward``; warp:
        ``mjw.kinematics`` + ``camlight``). No integration."""
        self.backend.forward_kinematics()

    def _write_external_wrench_native(self, targets: list[WrenchTarget]) -> None:
        """Set world-frame wrenches into xfrc_applied (world frame at CoM).

        xfrc_applied is full-model-width, indexed by raw mujoco body id (robot: self.body_ids;
        objects: self._object_root_body_ids). The view is a WarpBackend torch tensor
        [num_envs, nbody, 6] or a ClassicBackend numpy [nbody, 6] (single env).
        """
        forces_view = self.applied_forces
        # Runs every physics substep with a stable body set: cache the column -> mujoco-body-id
        # mapping and write all of a target's columns in one indexed assignment.
        if not hasattr(self, "_wrench_id_cache"):
            self._wrench_id_cache: dict[tuple[str, tuple[int, ...]], tuple[list[int], torch.Tensor]] = {}
        for target in targets:
            cols = target.write_cols
            key = (target.actor_name, tuple(cols))
            if key not in self._wrench_id_cache:
                if target.is_robot:
                    ids = [self.body_ids[col] for col in cols]
                else:
                    ids = [self._object_root_body_ids[target.actor_name]]  # objects are single-body
                # Index tensor must live on the buffer's device (warp backend; unused for classic).
                dev = forces_view.device if isinstance(forces_view, torch.Tensor) else "cpu"
                self._wrench_id_cache[key] = (ids, torch.tensor(ids, dtype=torch.long, device=dev))
            ids, ids_t = self._wrench_id_cache[key]
            wrench = target.wrench[:, cols, :]  # [num_envs, n_cols, 6], world frame
            if isinstance(forces_view, torch.Tensor):
                # In-place indexed assignment keeps the warp buffer graph-capture-safe (never rebound).
                forces_view[:, ids_t, :] = wrench.to(forces_view.device, forces_view.dtype)
            else:
                # Env 0 only: ClassicBackend rejects num_envs > 1 at construction, so the
                # accumulator's single env row is exactly row 0.
                forces_view[ids, :] = wrench[0].detach().cpu().numpy()

    def _actor_freejoint_addrs(self, obj_name: str) -> tuple[int, int]:
        """Return (qpos_addr, qvel_addr) for an actor's freejoint, or raise if unknown."""
        if obj_name not in self.object_addrs:
            raise KeyError(f"No body addressing for actor '{obj_name}' (known: {list(self.object_addrs)}).")
        addrs = self.object_addrs[obj_name]
        return addrs["qpos_addr"], addrs["qvel_addr"]

    @staticmethod
    def _freejoint_angvel_to_world(states: torch.Tensor) -> torch.Tensor:
        """Rotate a state tensor's angular-velocity (cols 10:13) from body-local to world frame.

        MuJoCo's freejoint qvel stores LINEAR velocity in world frame but ANGULAR velocity in the
        body's LOCAL frame. The unified actor-state representation (base_simulator) is world-frame for
        BOTH (matching IsaacGym/IsaacSim), so convert at this backend boundary using the actor's
        world orientation (xyzw quat in cols 3:7). In-place on a copy; returns it.
        """
        if states.shape[0] == 0:
            return states
        out = states.clone()
        out[:, 10:13] = quat_rotate(states[:, 3:7], states[:, 10:13], w_last=True)
        return out

    @staticmethod
    def _freejoint_angvel_to_local(states: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_freejoint_angvel_to_world`: world ang-vel (cols 10:13) -> body-local.

        Used on the SET path so a world-frame angular velocity from the unified representation lands
        correctly in MuJoCo's body-local freejoint qvel slot (the get/set pair stays symmetric)."""
        if states.shape[0] == 0:
            return states
        out = states.clone()
        out[:, 10:13] = quat_rotate_inverse(states[:, 3:7], states[:, 10:13], w_last=True)
        return out

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """Get actor states for any registered actor (robot or rigid object).

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
        KeyError
            If an actor has no resolved body addressing.
        """
        assert self.root_model
        assert self.root_data

        if len(indices) == 0:
            # Empty input is a uniform no-op across backends (matches IsaacGym/IsaacSim).
            return torch.empty(0, 13, device=self.sim_device)

        resolved_objects = self.object_registry.resolve_indices(indices)
        all_states: list[torch.Tensor] = []
        for obj_name, env_ids in resolved_objects:
            if obj_name in self.static_object_body_ids:
                # Static body: no qpos slice, read its constant world pose from xpos.
                states_for_envs = self._static_body_state(obj_name, env_ids)
            else:
                # Free body / robot: read the live freejoint state, per-env (GPU for
                # WarpBackend). Velocity is the freejoint qvel (body-local angular),
                # matching the set path and the robot's base_angular_vel proxy.
                qpos_addr, qvel_addr = self._actor_freejoint_addrs(obj_name)
                states_for_envs = self.backend.get_actor_state(env_ids, qpos_addr, qvel_addr)  # [num_envs, 13]
            all_states.append(states_for_envs)

        if not all_states:
            return torch.empty(0, 13, device=self.sim_device)
        # Convert each freejoint's body-local angular velocity to the world-frame the unified
        # actor-state representation uses (static bodies carry zero velocity, so this is a no-op
        # for them). Linear velocity and pose are already world-frame.
        return self._freejoint_angvel_to_world(torch.cat(all_states, dim=0))

    def _static_body_state(self, obj_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Read a static (jointless) body's world pose into a [len(env_ids), 13] state.

        Static bodies have no qpos slice; their world pose lives in the model kinematics
        (xpos/xquat). On the WarpBackend those are per-world tensors, so each env's own row
        is read: a static scene-file body placed at per-env origins reports its true per-env
        position. On the ClassicBackend (single env) the CPU xpos is broadcast. Velocity is
        zero (welded body).
        """
        assert self.root_data
        body_id = self.static_object_body_ids[obj_name]
        xpos_t = getattr(self.backend, "xpos_t", None)
        if xpos_t is not None:
            # WarpBackend: per-world kinematics [num_envs, nbody, 3/4]; index this body per env.
            pos = xpos_t[env_ids, body_id]  # [N, 3]
            quat = mj_to_holosoma_quat(cast("WarpBackend", self.backend).xquat_t[env_ids, body_id])  # [N, 4] wxyz->xyzw
            vel = torch.zeros(len(env_ids), 6, device=self.sim_device)
            return torch.cat([pos, quat, vel], dim=1)  # [N, 13]
        # ClassicBackend (single env): broadcast the CPU model pose.
        pos = torch.from_numpy(self.root_data.xpos[body_id]).float().to(self.sim_device)  # [3]
        quat_holo = mj_to_holosoma_quat(self.root_data.xquat[body_id])  # [w,x,y,z] -> [x,y,z,w]
        quat = torch.tensor(quat_holo, dtype=torch.float32, device=self.sim_device)
        state = torch.cat([pos, quat, torch.zeros(6, device=self.sim_device)])  # [13], zero vel
        return state.unsqueeze(0).expand(len(env_ids), 13).clone()

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """Set actor states for any registered actor (robot or rigid object).

        Writes each actor's pose/velocity into its own freejoint qpos/qvel slice,
        resolved via object_addrs (no longer hardcoded to the robot's qpos[0:3]).

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
        KeyError
            If an actor has no resolved body addressing.
        """
        assert self.root_data is not None

        if len(indices) == 0:
            # Empty input is a uniform no-op across backends (matches IsaacGym/IsaacSim);
            # short-circuit before resolve/convert so we skip the stray write_state_updates
            # (mj_forward on ClassicBackend) at the tail.
            return

        resolved_objects = self.object_registry.resolve_indices(indices)

        # Incoming angular velocity is world-frame (unified representation); MuJoCo's freejoint qvel
        # wants it body-local. Convert up front using each row's world quat (cols 3:7) so the
        # per-actor slices below write the correct local ang-vel, symmetric with the read path.
        states = self._freejoint_angvel_to_local(states)

        state_offset = 0
        for obj_name, env_ids in resolved_objects:
            num_states = len(env_ids)
            obj_states = states[state_offset : state_offset + num_states]  # [num_envs, 13]
            state_offset += num_states

            if obj_name in self.static_object_body_ids:
                # Static (welded, jointless) body, scene object or fixed-base robot: no qpos slice,
                # so it moves via the model body_pos/body_quat + forward-kinematics path rather than
                # set_actor_state. This is a kinematic teleport: the pose is honored (collisions
                # recompute at the new pose) but the velocity columns (7:13) are ignored (a welded
                # body carries no qvel), and a write whose pose equals the current pose is a no-op
                # (the cross-backend fixed-base rule). Pose is [pos(0:3), quat(3:7) xyzw].
                current = self._static_body_state(obj_name, env_ids)
                if root_pose_changed(obj_states[:, :7], current[:, :7]):
                    body_id = self.static_object_body_ids[obj_name]
                    self.backend.set_static_body_world_pose(
                        [body_id],
                        obj_states[:, :3].unsqueeze(1),  # [N, 1, 3]
                        obj_states[:, 3:7].unsqueeze(1),  # [N, 1, 4] xyzw
                        env_ids=env_ids,
                    )
                    if obj_name == "robot":
                        current = self._static_body_state(obj_name, env_ids)
                if obj_name == "robot":
                    # Fixed-base robot: robot_root_states is a plain tensor, not a live view, so
                    # mirror the current pose (zero velocity) back into it. Also after a same-pose
                    # no-op: the caller may have written velocities into the mirror.
                    self.robot_root_states[env_ids] = current
                continue

            # Write each requested env's state into this actor's freejoint slice on the
            # live backend storage (GPU for WarpBackend), per-env. The CPU render
            # snapshot does not sync back to the GPU, so writes must go to the backend.
            qpos_addr, qvel_addr = self._actor_freejoint_addrs(obj_name)
            self.backend.set_actor_state(env_ids, obj_states, qpos_addr, qvel_addr)

        if write_updates:
            self.write_state_updates()

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """Get actor indices using ObjectRegistry (robot or rigid object).

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
        """
        if isinstance(names, str):
            names = [names]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_initial_poses(self, names: list[str], env_ids: EnvIds | None = None) -> ActorPoses:
        """Get initial poses for any registered actor (robot or rigid object).

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
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def get_actor_initial_velocities(self, names: list[str], env_ids: EnvIds | None = None) -> torch.Tensor:
        """Get initial velocities for any registered actor (sibling of get_actor_initial_poses).

        Returns [len(names) * len(env_ids), 6] world-frame [vx,vy,vz,wx,wy,wz], same row order as
        get_actor_initial_poses (so the two concatenate into the 13-vector set_actor_states wants).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_initial_velocities_batch(names, env_ids)

    def write_state_updates(self) -> None:
        """Flush pending state writes so derived quantities reflect them.

        Mirrors the cross-backend batch flush (IsaacGym: no-op; IsaacSim:
        ``scene.write_data_to_sim()``). For MuJoCo:
          - ClassicBackend: run ``mj_forward`` on the CPU data so xpos/xquat/cvel and
            other derived quantities reflect the qpos/qvel just written.
          - WarpBackend: a no-op. The next ``step()`` (a captured CUDA graph that
            begins with forward kinematics) consumes the updated GPU qpos/qvel; no
            out-of-graph GPU forward runs here.
        """
        if isinstance(self.backend, WarpBackend):
            return
        assert self.root_model
        assert self.root_data
        mujoco.mj_forward(self.root_model, self.root_data)

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
        """Write world-frame root states using the base simulator's row and quaternion conventions.

        None selects all environments or the current ``robot_root_states``, respectively.
        For a fixed base, a same-pose write is a no-op; a changed pose teleports the robot.
        Fixed-base writes ignore velocity columns.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)
        root_states = robot_rows(self, env_ids, root_states)

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.robot_config.asset.fix_base_link:
            # Fixed base: the unified static path (same-pose no-op / pose teleport, velocity columns
            # ignored, as for any welded body). It also refreshes robot_root_states for these envs.
            self.set_actor_states(["robot"], env_ids, root_states)
            return

        # Incoming angular velocity is world-frame (the robot_root_states / all_root_states
        # convention); MuJoCo's freejoint qvel wants it body-local. Convert before the backend
        # write (which scatters cols 10:13 raw into qvel), symmetric with the view __getitem__
        # read path and with set_actor_states_by_index.
        root_states = self._freejoint_angvel_to_local(root_states)

        # Delegate to backend
        root_addrs = {
            "robot_qpos_addr": self.object_addrs["robot"]["qpos_addr"],
            "robot_qvel_addr": self.object_addrs["robot"]["qvel_addr"],
        }
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
            DOF states in the IsaacGym-flattened 2D format ``[N * num_dof, 2]`` where
            ``[:, 0] = positions`` and ``[:, 1] = velocities``. The required first
            dimension depends on the active backend:

            - **WarpBackend**: the FULL global tensor ``[num_envs * num_dof, 2]``.
              The backend selects rows for ``env_ids`` internally, so even a partial
              reset must pass states for all environments (this is how ``self.dof_state``
              is shaped).
            - **ClassicBackend** (single environment): ``[len(env_ids) * num_dof, 2]``.
              With ``num_envs == 1`` this is equivalent to the global shape above.

            If None, uses ``self.dof_state`` (already in the correct global format).

            Note: a 3D ``[num_envs, num_dofs, 2]`` tensor is NOT accepted by either
            MuJoCo backend (unlike the IsaacSim implementation of this method).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.num_dof == 0:
            logger.info("No robot DOFs available - skipping DOF state update")
            return

        assert dof_states is not None
        # Both MuJoCo backends consume the IsaacGym-flattened 2D format only; neither
        # handles the 3D [num_envs, num_dofs, 2] form (that is IsaacSim-specific).
        if dof_states.dim() != 2:
            raise ValueError(
                f"Unsupported dof_states tensor shape {tuple(dof_states.shape)}: expected a 2D "
                f"[N * num_dof, 2] tensor (IsaacGym-flattened, [:, 0]=pos, [:, 1]=vel)."
            )
        # The required first dimension is backend-specific: WarpBackend.set_dof_state
        # indexes the FULL global tensor by env_ids ([num_envs * num_dof, 2]), whereas
        # ClassicBackend.set_dof_state reshapes the whole tensor by len(env_ids)
        # ([len(env_ids) * num_dof, 2]). Validate against the active backend so a
        # partial-reset global tensor is not falsely rejected for the warp path.
        if isinstance(self.backend, WarpBackend):
            expected_rows = self.num_envs * self.num_dof
        else:
            expected_rows = len(env_ids) * self.num_dof
        if dof_states.shape[0] != expected_rows:
            is_warp = isinstance(self.backend, WarpBackend)
            rows_label = "num_envs" if is_warp else "len(env_ids)"
            raise ValueError(
                f"Unsupported dof_states first dimension {dof_states.shape[0]}: "
                f"expected {expected_rows} (= {rows_label}"
                f" * num_dof) for the active "
                f"{type(self.backend).__name__}."
            )

        # Delegate to backend
        dof_addrs = {"dof_qpos_addrs": self.dof_qpos_addrs, "dof_qvel_addrs": self.dof_qvel_addrs}
        self.backend.set_dof_state(env_ids, dof_states, dof_addrs)

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached setup limits; repeated reads do not allocate or write simulator state."""
        return self._dof_limit_properties

    @cached_property
    def _dof_limit_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard DOF limits as the compiled model enforces them (config lists where given, MJCF otherwise).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(dof_pos_limits [num_dof, 2], dof_vel_limits [num_dof], torque_limits [num_dof])``.
        """
        meta = self.scene_manager.robot_spec_meta
        assert meta is not None, "robot_spec_meta must be captured in scene_manager.add_robot"
        position, velocity, effort = read_dof_limits(self.root_model, meta, self.robot_config)
        self.dof_pos_limits = torch.tensor(position, dtype=torch.float, device=self.sim_device).reshape(-1, 2)
        self.dof_vel_limits = torch.tensor(velocity, dtype=torch.float, device=self.sim_device)
        self.torque_limits = torch.tensor(effort, dtype=torch.float, device=self.sim_device)
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """Return the canonical body-observation index; raise ``RuntimeError`` for an unknown name."""
        try:
            return self.body_names.index(body_name)
        except ValueError:
            raise RuntimeError(f"Body '{body_name}' not found in body_names: {self.body_names}")

    # ----- Camera sensors (mounted, follow-by-spec-hierarchy) -----

    def _create_sensors(self) -> None:
        """Create one shared manager for mounted cameras and LiDARs after model compilation."""
        create_sensors(self)

    def _offset_world_cameras(self) -> None:
        """Place each env's world-mount cameras at its own origin (WarpBackend multi-env only).

        Delegates to ``sensor_setup.offset_world_cameras``; no-op for Classic (single-env) and
        the no-camera / no-world-camera cases.
        """
        offset_world_cameras(self)

    def render_sensors(self) -> None:
        """Capture all due mounted sensors into their cached buffers."""
        render_sensors(self)

    def setup_viewer(self) -> None:
        """Set up MuJoCo viewer using official mujoco.viewer API with keyboard callback."""
        logger.info("=== Setting up MuJoCo viewer ===")

        if self.headless:
            logger.info("Headless mode enabled - skipping viewer setup")
            self.viewer = None
            return

        self.viewer_input = ViewerInputController(self)
        self.viewer = mujoco.viewer.launch_passive(
            self.root_model, self.root_data, key_callback=self.viewer_input.on_key
        )
        # Native camera-frustum gizmo (mjVIS_CAMERA) on the live viewer's own MjvOption; a viewer-only
        # decoration drawn from each <camera>'s intrinsics, separate from the sensor renders.
        apply_sensor_scene_flags(self.debug_viz_enabled, self.viewer.opt)
        logger.info("=== Viewer setup completed with keyboard callback ===")

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

        if self.simulator_config.viewer.enable_tracking:
            if "robot" in self.object_addrs:
                qpos_addr = self.object_addrs["robot"]["qpos_addr"]
                self.viewer.cam.lookat[:] = self.root_data.qpos[qpos_addr : qpos_addr + 3]
            else:
                self.viewer.cam.lookat[:] = self.root_data.xpos[self.static_object_body_ids["robot"]]

        self.viewer.sync()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    def _physics_time(self) -> float:
        """Return the active MuJoCo backend's simulation clock in seconds."""
        return self.backend.physics_time()

    def _close_viewer(self) -> None:
        """Close the viewer handle once."""
        viewer = self.viewer
        self.viewer = None
        if viewer is not None:
            viewer.close()

    def _close_backend(self) -> None:
        """Close the backend if asset loading acquired one."""
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.close()

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        """Return selected torque-motor efforts as float32 ``[num_dof]`` in canonical DOF order.

        Values exclude other authored actuators and precede the joint's combined-force clamp.
        Classic supports only ``env_id=0`` and raises ``RuntimeError`` otherwise.
        Warp raises ``IndexError`` for an environment outside ``[0, num_envs)``.
        """
        if isinstance(self.backend, WarpBackend):
            if not 0 <= env_id < self.num_envs:
                raise IndexError(f"Environment {env_id} is outside [0, {self.num_envs}).")
            return self.backend.actuator_force_t[env_id, self.dof_actuator_ids].float()

        if env_id != 0:
            raise RuntimeError(f"MuJoCo classic currently only supports single environment (env_id=0), got {env_id}")

        assert self.root_data is not None
        # actuator_force is indexed by actuator id; gather the robot DOFs' actuators in dof_names order.
        return torch.from_numpy(self.root_data.actuator_force[self.dof_actuator_ids]).float().to(self.sim_device)

    def print_mujoco_model_tree(self) -> None:
        """Print comprehensive MuJoCo model structure for debugging."""
        assert self.root_model
        assert self.root_data
        print_mujoco_model_tree(self.root_model, self.root_data, self.scene_manager.robot_model_path)
