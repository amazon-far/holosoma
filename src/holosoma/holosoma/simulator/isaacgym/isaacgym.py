from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from loguru import logger
from rich.progress import Progress
from torch import Tensor

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.isaacgym.physics import (
    apply_mass_from_config,
    apply_physx_asset_options,
    apply_rigid_shape_properties,
)
from holosoma.simulator.isaacgym.urdf_scene_loader import URDFSceneLoader
from holosoma.simulator.isaacgym.video_recorder import IsaacGymVideoRecorder
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.shared.root_states_view import UnifiedRootStatesView
from holosoma.simulator.shared.terrain import Terrain
from holosoma.simulator.shared.virtual_gantry import (
    GantryCommand,
    GantryCommandData,
    create_virtual_gantry,
)
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.draw import draw_line, draw_sphere
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.rotations import quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import to_torch, torch_rand_float

# Change-of-basis quaternion (xyzw) from the holosoma camera frame (-Z fwd / +Y up) to
# IsaacGym's native camera frame (+X fwd / +Z up): the rotation sending +X->-Z, +Z->+Y, +Y->-X.
# MuJoCo/IsaacSim use the -Z fwd / +Y up frame directly.
_CANONICAL_TO_ISAACGYM_XYZW = (0.5, -0.5, -0.5, -0.5)


class Scene:
    """Scene wrapper for IsaacGym to provide unified interface.

    This class provides a unified scene interface for IsaacGym, currently
    focusing on environment origins. It ensures consistent tensor format
    and device placement for environment origin data.

    Parameters
    ----------
    env_origins : torch.Tensor
        Environment origins as tensor with shape [num_envs, 3].
    device : str
        Device identifier for tensor operations (e.g., 'cuda:0', 'cpu').

    Attributes
    ----------
    _env_origins : torch.Tensor
        Internal storage for environment origins tensor.

    Raises
    ------
    ValueError
        If env_origins doesn't have the expected shape [num_envs, 3].
    """

    def __init__(self, env_origins: torch.Tensor, device: str):
        # Ensure consistent tensor format
        if not isinstance(env_origins, torch.Tensor):
            env_origins = torch.tensor(env_origins, device=device, dtype=torch.float32)

        # Ensure correct device and dtype
        self._env_origins = env_origins.to(device=device, dtype=torch.float32)

        # Validate shape
        if self._env_origins.dim() != 2 or self._env_origins.shape[1] != 3:
            raise ValueError(f"env_origins must have shape [num_envs, 3], got {self._env_origins.shape}")

    @property
    def env_origins(self) -> torch.Tensor:
        """Get environment origins tensor.

        Returns
        -------
        torch.Tensor
            Environment origins as [num_envs, 3] float32 tensor.
        """
        return self._env_origins


class IsaacGym(BaseSimulator):
    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str):
        super().__init__(tyro_config, terrain_manager, device)

        # Gym viewer handle; created by setup_viewer() (skipped when headless), so it stays
        # None on headless runs and render() must tolerate that.
        self.viewer = None
        self.visualize_viewer = False

        # For force visualization
        self.vis_force_range = False

        # Actor/Object management
        self.object_assets: dict[str, Any] = {}
        self.object_handles: dict[str, list[Any]] = {}
        self.gym_object_indices: dict[str, list[torch.Tensor]] = {}

    def set_headless(self, headless):
        super().set_headless(headless)

    def set_startup_randomization_callback(self, callback):
        """Set a callback to be invoked during environment startup for domain randomization.

        This is an IsaacGym-specific method that allows the environment to inject
        domain randomization logic during environment creation. The callback will be
        invoked after all environments are created but before prepare_sim() is called.

        Args:
            callback (callable): A callable that takes no arguments and returns None.
                                Typically this is randomization_manager.setup().
        """
        self.startup_randomization_callback = callback

    def setup(self):
        self.sim_params = self._parse_sim_params()
        self.sim_dt = self.sim_params.dt

        self.physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(str(self.sim_device))
        if sim_device_type == "cpu":
            # Force CPU
            self.sim_params.use_gpu_pipeline = False
            self.sim_params.physx.use_gpu = False

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True,
        # otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and self.sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # Cameras (video recorder or perception sensors) need a graphics context even when
        # headless; without it create_camera_sensor returns no image (black frame).
        self._cameras_enabled = self.video_config.enabled or bool(self.sensor_config)
        self.graphics_device_id = self.sim_device_id
        if self.headless and not self._cameras_enabled:
            self.graphics_device_id = -1
        elif self._cameras_enabled:
            logger.info("Cameras enabled: keeping graphics enabled for camera sensor support")

        if self.video_config.enabled:
            self.video_recorder = IsaacGymVideoRecorder(self.video_config, self)

        sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if sim is None:
            logger.error("*** Failed to create sim")
            sys.exit(1)

        logger.info("Creating Sim...", "green")

        self.sim = sim

    def _parse_sim_params(self):
        # TODO: this sim params are not loaded from the config file
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.simulator_config.sim.fps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.up_axis = 1  # 0 is y, 1 is z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = self.simulator_config.sim.physx.solver_type
        sim_params.physx.num_position_iterations = self.simulator_config.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.simulator_config.sim.physx.num_velocity_iterations
        sim_params.physx.num_threads = self.simulator_config.sim.physx.num_threads
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = self.robot_config.contact_pairs_multiplier * 1024 * 1024
        sim_params.use_gpu_pipeline = True

        gymutil.parse_sim_config(dataclasses.asdict(self.simulator_config.sim), sim_params)
        return sim_params

    def get_supported_scene_formats(self) -> list[str]:
        """See base class.

        IsaacGym-specific notes:
        - Currently supports URDF scenes only

        Returns
        -------
        List[str]
            ["urdf"]
        """
        return ["urdf"]

    def setup_terrain(self):
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        mesh_type = terrain_state.mesh_type
        # IsaacGym's ground has no separable visual, so hide_visual is satisfied only where it is
        # already met (headless draws nothing) and warned otherwise. set_headless runs before this,
        # so self.headless is reliable.
        if terrain_state.hide_visual and not self.headless:
            logger.warning(
                "terrain hide_visual=True is not supported on IsaacGym in headful mode (its ground "
                "geometry has no separable visual); run headless or use MuJoCo/IsaacSim."
            )
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type in ["trimesh", "load_obj"]:
            terrain = terrain_state.terrain
            self._create_trimesh(terrain)
        else:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")

    def _create_trimesh(self, terrain: Terrain):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg."""
        logger.info("Creating trimesh terrain")
        tm_params = gymapi.TriangleMeshParams()
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        assert terrain_state.mesh is not None
        vertices = terrain_state.mesh.vertices.astype(np.float32)
        triangles = terrain_state.mesh.faces.astype(np.uint32)
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]

        tm_params.static_friction = terrain_state.static_friction
        tm_params.dynamic_friction = terrain_state.dynamic_friction
        tm_params.restitution = terrain_state.restitution
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(order="C"), triangles.flatten(order="C"), tm_params)
        logger.info("Created trimesh terrain")

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        logger.info("Creating plane terrain")
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        plane_params.static_friction = terrain_state.static_friction
        plane_params.dynamic_friction = terrain_state.dynamic_friction
        plane_params.restitution = terrain_state.restitution
        self.gym.add_ground(self.sim, plane_params)
        logger.info("Created plane terrain")

    def load_assets(self):
        self._load_scene()

        asset_root = self.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())

        asset_file = self.robot_config.asset.urdf_file
        self.robot_asset = self._setup_robot_asset_when_env_created(asset_root, asset_file, self.robot_config.asset)
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = self._setup_robot_props_when_env_created()

        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), (
            f"Number of bodies ({self.num_bodies}) must be equal to number of body names "
            f"({len(self.robot_config.body_names)})"
        )
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

    @property
    def has_scene_objects(self):
        # For now, use object_assets as a proxy, should be more direct/explicit though
        return len(self.object_assets) > 0

    def _load_scene(self):
        """
        Load scene files using the new unified configuration structure
        """
        if self.scene_config is None:
            return

        if not self.scene_config.scene_files and not self.scene_config.rigid_objects:
            logger.info("No scene files or rigid objects configured for loading")
            return

        # Initialize URDF scene loader
        if not hasattr(self, "urdf_scene_loader"):
            self.urdf_scene_loader = URDFSceneLoader(self.gym, self.sim, self.device)

        assets, _ = self.urdf_scene_loader.load_scene_files(self.scene_config)
        self.object_assets.update(assets)
        logger.info(f"IsaacGym: Loaded {len(assets)} scene objects from scene files")

    def _setup_robot_asset_when_env_created(self, asset_root, asset_file, asset_cfg):
        asset_path = Path(asset_root) / asset_file
        gym_asset_root = str(asset_path.parent)
        gym_asset_file = asset_path.name

        asset_options = gymapi.AssetOptions()

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        # Asset-import knobs that stay on RobotAssetConfig (no PhysicsConfig analogue).
        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "armature",
            "thickness",
            "disable_gravity",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(getattr(asset_options, option), getattr(asset_cfg, option))
            setattr(asset_options, option, option_value)

        # density + the PhysX solver knobs (damping / velocity caps) come from the shared link_physics
        # via the same helper the object path uses (physics.apply_physx_asset_options), so a robot link
        # and a scene object map the physx/density load-time options identically. None keeps defaults.
        apply_physx_asset_options(asset_options, asset_cfg.link_physics)

        self.robot_asset = self.gym.load_asset(self.sim, gym_asset_root, gym_asset_file, asset_options)
        return self.robot_asset

    def _setup_robot_props_when_env_created(self):
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        # save body names from the asset
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)

        return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(self, num_envs, env_origins, base_init_state):
        """
        Main interface called by base_task to create environments.
        Automatically detects if scene objects are loaded and creates environments accordingly.
        """
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        self.envs = []
        self.robot_handles = []
        self.robot_indices = []

        if self.has_scene_objects:
            logger.info(f"Creating {self.num_envs} environments with {len(self.object_assets)} scene objects...")
            self.object_handles = {name: [] for name in self.object_assets}

        logger.info(f"Creating {self.num_envs} environments...")
        with Progress() as progress:
            task = progress.add_task(f"Creating {self.num_envs} environments...", total=self.num_envs)
            for i in range(self.num_envs):
                # create env instance
                env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                self._build_each_env(i, env_handle)
                progress.update(task, advance=1)

        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)

        self.scene = Scene(self.env_origins, self.device)

        # Initialize virtual gantry using config
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

        if self.video_recorder:
            self.video_recorder.setup_recording()
            self.video_recorder.register_hooks(self.hooks)

        # Initialize command system for keyboard controls
        # Command tensor format: [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        self.commands = torch.zeros(self.num_envs, 9, device=self.device, dtype=torch.float32)
        logger.info(f"Initialized command system with shape: {self.commands.shape}")

        # After building, register objects and setup mappings/indexes. Unconditional (like
        # MuJoCo/IsaacSim): a robot-only scene still registers "robot" so the unified
        # get/set_actor_states API works on every backend.
        self._register_objects()

        # Register the per-env camera handles into the SensorManager.
        self._create_sensors()

        # Invoke startup randomization for domain randomization
        # This must happen AFTER all envs are created but BEFORE prepare_sim()
        self._invoke_startup_randomization()

        return self.envs, self.robot_handles, self.object_handles

    def _invoke_startup_randomization(self):
        """Invoke startup randomization callback if one has been registered.

        This is IsaacGym-specific functionality to support domain randomization
        during environment creation, before prepare_sim() is called.
        """
        callback = getattr(self, "startup_randomization_callback", None)
        if callback is not None:
            self.startup_randomization_callback()

    def _build_each_env(self, env_id, env_ptr):
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # keep the base height the same as the initial height
        # TODO: Move randomization of robot position out of simulator, kept now for backwards
        # compatibility.
        pos = self.env_origins[env_id].clone()
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=str(self.device)).squeeze(1)
        pos[2] = self.base_init_state[2]
        start_pose.p = gymapi.Vec3(*pos)

        if env_id == 0:
            # Map objects indices into envs by name
            self.gym_object_indices = {name: [] for name in self.object_assets}
            self.gym_object_indices["robot"] = []

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

        robot_handle = self.gym.create_actor(
            env_ptr,
            self.robot_asset,
            start_pose,
            self.robot_config.asset.robot_type,
            env_id,
            1 - int(self.robot_config.asset.enable_self_collisions),
            0,
        )
        if self.simulator_config.sim.physx.enable_dof_force_sensors:
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
        self._body_list = self.gym.get_actor_rigid_body_names(env_ptr, robot_handle)

        # Apply the robot's shared link_physics friction/restitution/compliance to all robot shapes
        # via the apply_rigid_shape_properties seam objects use (isaacgym sub-config; ignores the
        # rest), uniform across shapes (body_names='.*'). This is the spawn-time baseline; if friction
        # DR is enabled it composes over this at startup (operation='abs'). density + the PhysX damping
        # /velocity knobs already landed at asset load (link_physics.density / .physx). Whole-robot
        # mass is not applied here (it would clobber every link); per-link mass is a deferred feature.
        if self.robot_config.asset.link_physics is not None:
            apply_rigid_shape_properties(self.gym, env_ptr, robot_handle, self.robot_config.asset.link_physics, "robot")

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        if self.robot_config.apply_dof_armature_in_isaacgym:
            dof_props_asset = self._apply_dof_armature_from_config_to_props(dof_props_asset)
        dof_props = self._process_dof_props(dof_props_asset, env_id)
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props, recomputeInertia=True)
        self.envs.append(env_ptr)
        self.robot_handles.append(robot_handle)
        robot_idx = self.gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
        self.robot_indices.append(robot_idx)
        self.gym_object_indices["robot"].append(robot_idx)

        # Create scene objects
        for object_name, object_asset in self.object_assets.items():
            start_pose = gymapi.Transform()
            pose = self.urdf_scene_loader.get_initial_pose(object_name)

            start_pose.p = gymapi.Vec3(*pose[:3])
            quat = pose[3:7]
            start_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

            # Add environment origin offset -- ideally we move to lower/upper with create_env
            pos = self.env_origins[env_id].clone()
            start_pose.p.x += pos[0]
            start_pose.p.y += pos[1]
            start_pose.p.z += pos[2]

            object_handle = self.gym.create_actor(
                env_ptr,
                object_asset,
                start_pose,
                object_name,
                env_id,
                -1,  # self_collisions # TODO: check this later
                0,  # group
            )
            if self.simulator_config.sim.physx.enable_dof_force_sensors:
                self.gym.enable_actor_dof_force_sensors(env_ptr, object_handle)

            # Apply physics properties after actor creation
            self.apply_physics_properties_to_actor(env_ptr, object_handle, object_name)

            # Keep handles for registration and direct user access for tasks/envs
            self.object_handles[object_name].append(object_handle)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.gym_object_indices[object_name].append(object_idx)

        # Mounted cameras for this env: created per-env and attached to the mount body so they
        # follow it. IsaacGym has no batched camera: one sensor per env. The handle lists span all
        # envs (parallel to self.envs), so initialize them once and append per env.
        if env_id == 0:
            self._camera_handles: dict[str, list[Any]] = {name: [] for name in self.sensor_config}
        self._build_env_cameras(env_id, env_ptr, robot_handle)

    # ----- Camera sensors -----

    def _resolve_mount_body_handle(self, env_ptr, robot_handle, mount):
        """Return the rigid-body handle for a sensor mount within ``env_ptr``.

        ``robot_link`` -> a named robot link (name the root link to mount on the base);
        ``actor`` -> a scene actor's (single) body. Uses IsaacGym's per-actor body lookup.
        """
        if mount.target_kind == "robot_link":
            body_name = mount.target
            handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, body_name)
            if handle < 0:
                raise ValueError(f"Camera robot mount body '{body_name}' not found in robot bodies {self._body_list}.")
            return handle
        if mount.target_kind == "actor":
            actor_handles = self.object_handles.get(mount.target)
            if not actor_handles:
                raise ValueError(f"Camera actor mount '{mount.target}' is not a registered scene object.")
            actor_handle = actor_handles[-1]  # the handle just built for this env
            body_name = self.gym.get_actor_rigid_body_names(env_ptr, actor_handle)[0]
            return self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
        raise ValueError(f"Unknown camera mount target_kind '{mount.target_kind}'.")

    def _build_env_cameras(self, env_id, env_ptr, robot_handle):
        """Create and body-attach one camera sensor per configured camera, for this env."""
        for cam_name, cam in self.sensor_config.items():
            props = gymapi.CameraProperties()
            props.width = cam.width
            props.height = cam.height
            props.enable_tensors = True  # GPU image tensors (zero-copy read)
            props.near_plane = cam.near  # agnostic-core clipping, honored on every backend
            props.far_plane = cam.far
            # IsaacGym takes a horizontal fov; derive it from the vertical fov and aspect.
            aspect = cam.width / cam.height
            props.horizontal_fov = math.degrees(2 * math.atan(math.tan(math.radians(cam.vertical_fov) / 2) * aspect))
            ig = cam.isaacgym
            if ig and ig.supersampling_horizontal is not None:
                props.supersampling_horizontal = ig.supersampling_horizontal
            if ig and ig.supersampling_vertical is not None:
                props.supersampling_vertical = ig.supersampling_vertical
            if ig and ig.use_collision_geometry is not None:
                props.use_collision_geometry = ig.use_collision_geometry

            cam_handle = self.gym.create_camera_sensor(env_ptr, props)
            if cam_handle < 0:
                raise RuntimeError(f"IsaacGym failed to create camera sensor '{cam_name}' (graphics disabled?).")

            local_tf = self._mount_to_isaacgym_transform(cam.mount)
            if cam.mount.target_kind == "world":
                # Free-floating: place at the env-local pose and leave it fixed (no body to follow).
                # Unlike attach_camera_to_body, set_camera_transform takes a WORLD transform (the
                # env_ptr only selects which env's camera, it does not offset the pose), so add this
                # env's origin — otherwise every env's world camera lands near the global origin and
                # only env 0 (origin [0,0,0]) frames its scene; envs 1..N look at empty space.
                origin = self.env_origins[env_id]
                world_tf = gymapi.Transform(
                    p=gymapi.Vec3(
                        local_tf.p.x + float(origin[0]),
                        local_tf.p.y + float(origin[1]),
                        local_tf.p.z + float(origin[2]),
                    ),
                    r=local_tf.r,
                )
                self.gym.set_camera_transform(cam_handle, env_ptr, world_tf)
            else:
                body_handle = self._resolve_mount_body_handle(env_ptr, robot_handle, cam.mount)
                self.gym.attach_camera_to_body(cam_handle, env_ptr, body_handle, local_tf, gymapi.FOLLOW_TRANSFORM)
            self._camera_handles[cam_name].append(cam_handle)

    def _mount_to_isaacgym_transform(self, mount) -> gymapi.Transform:
        """Mount (pos + w-first quat, -Z fwd/+Y up) to an IsaacGym local Transform.

        Composes the mount orientation with the change-of-basis so the camera looks where the
        mount intends on IsaacGym's native (+X fwd / +Z up) optical axis.
        """
        wxyz = torch.tensor([mount.orientation], dtype=torch.float32)  # [1,4] w,x,y,z
        xyzw = wxyz[:, [1, 2, 3, 0]]
        basis = torch.tensor([_CANONICAL_TO_ISAACGYM_XYZW], dtype=torch.float32)
        native_xyzw = quat_mul(xyzw, basis, w_last=True)[0]
        tf = gymapi.Transform()
        tf.p = gymapi.Vec3(*mount.position)
        tf.r = gymapi.Quat(float(native_xyzw[0]), float(native_xyzw[1]), float(native_xyzw[2]), float(native_xyzw[3]))
        return tf

    def _create_sensors(self) -> None:
        """Register the per-env camera handles (built in the env loop) into the SensorManager."""
        cameras = self.sensor_config
        if not cameras:
            return
        from holosoma.simulator.shared.camera_sensor import SensorManager

        # IsaacGym uses self.device (may be "cpu").
        sim = self.simulator_config.sim
        self.sensor_manager = SensorManager(self.device, control_hz=sim.fps / sim.control_decimation_steps)
        for cam_name, cam in cameras.items():
            self.sensor_manager.register_camera(cam_name, cam)

    def render_sensors(self) -> None:
        """Render all camera sensors once (per control step), honoring update_decimation."""
        if self.sensor_manager is None:
            return
        due = self.sensor_manager.collect_due()
        if not due:
            return
        # Update the graphics scene from the latest physics state, then render all camera sensors
        # in one pass. fetch_results + step_graphics are required before render or the image is
        # blank (same sequence the video recorder uses).
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            for runtime in due:
                handles = self._camera_handles[runtime.name]  # per-env handles (parallel to self.envs)
                if "rgb" in runtime.config.data_types:
                    frames = []
                    for e in range(self.num_envs):
                        # IsaacGym IMAGE_COLOR is RGBA uint8 [H,W,4] on GPU (channel 0 = R); drop
                        # alpha to get R,G,B.
                        gpu_t = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[e], handles[e], gymapi.IMAGE_COLOR
                        )
                        frames.append(gymtorch.wrap_tensor(gpu_t)[..., :3].clone())  # [H,W,4] RGBA -> RGB
                    runtime.set_buffer("rgb", torch.stack(frames, dim=0).to(self.device))  # [N,H,W,3]
                if "depth" in runtime.config.data_types:
                    frames = []
                    for e in range(self.num_envs):
                        # IsaacGym IMAGE_DEPTH is [H,W] float32, negative distance along the camera
                        # axis (looks down -Z) with -inf for no-hit. Negate to get positive
                        # meters; -inf becomes the +inf no-hit sentinel.
                        gpu_t = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[e], handles[e], gymapi.IMAGE_DEPTH
                        )
                        frames.append((-gymtorch.wrap_tensor(gpu_t)).clone())  # [H,W] +meters, +inf no-hit
                    runtime.set_buffer("depth", torch.stack(frames, dim=0).unsqueeze(-1).to(self.device))  # [N,H,W,1]
        finally:
            self.gym.end_access_image_tensors(self.sim)

    def _process_rigid_shape_props(self, props, env_id):
        """No-op. Randomization manager will handle friction domain randomization."""
        return props

    def _apply_dof_armature_from_config_to_props(self, props):
        dof_armature_from_config = self.robot_config.dof_armature_list

        for i in range(len(props)):
            props["armature"][i] = dof_armature_from_config[i]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.hard_dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_pos_limits_termination = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit

                self.dof_pos_limits_termination[i, 0] = (
                    m - 0.5 * r * self.robot_config.termination_close_to_dof_pos_limit
                )
                self.dof_pos_limits_termination[i, 1] = (
                    m + 0.5 * r * self.robot_config.termination_close_to_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        """No-op. Randomization manager will handle body mass/com domain randomization."""
        return props

    def get_dof_limits_properties(self):
        # assert the isaacgym dof limits are the same as the config
        for i in range(self.num_dof):
            # import pdb; pdb.set_trace()
            assert abs(self.hard_dof_pos_limits[i, 0].item() - self.robot_config.dof_pos_lower_limit_list[i]) < 1e-5, (
                f"DOF {i} lower limit does not match: {self.hard_dof_pos_limits[i, 0].item()} != "
                f"{self.robot_config.dof_pos_lower_limit_list[i]}"
            )
            assert abs(self.hard_dof_pos_limits[i, 1].item() - self.robot_config.dof_pos_upper_limit_list[i]) < 1e-5, (
                f"DOF {i} upper limit does not match: {self.hard_dof_pos_limits[i, 1].item()} != "
                f"{self.robot_config.dof_pos_upper_limit_list[i]}"
            )
            assert abs(self.dof_vel_limits[i].item() - self.robot_config.dof_vel_limit_list[i]) < 1e-5, (
                f"DOF {i} velocity limit does not match: {self.dof_vel_limits[i].item()} != "
                f"{self.robot_config.dof_vel_limit_list[i]}"
            )
            assert abs(self.torque_limits[i].item() - self.robot_config.dof_effort_limit_list[i]) < 1e-5, (
                f"DOF {i} effort limit does not match: {self.torque_limits[i].item()} != "
                f"{self.robot_config.dof_effort_limit_list[i]}"
            )

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        return self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], body_name)

    def _get_base_body_name(self, preference_order: list[str]) -> str:
        """Get the base body name with fallback logic.

        Args:
            preference_order: List of body names to try in order

        Returns:
            The first body name found in the robot's body list

        Raises:
            ValueError: If none of the preferred body names are found
        """
        # Use the robot configuration's base_link field if available as first priority
        if hasattr(self.robot_config, "base_link") and self.robot_config.base_link:
            if self.robot_config.base_link in self._body_list:
                return self.robot_config.base_link

        # Fallback to preference order
        for preferred_name in preference_order:
            if preferred_name in self._body_list:
                return preferred_name

        raise ValueError(
            f"None of the preferred base body names {preference_order} found in robot body names: {self._body_list}"
        )

    def prepare_sim(self) -> None:
        self.gym.prepare_sim(self.sim)
        # Refresh tensors BEFORE we acquire them https://forums.developer.nvidia.com/t/isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/4
        self.refresh_sim_tensors()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        # jacobian and mass matrix
        robot_name = self.robot_config.asset.robot_type
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, robot_name)
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, robot_name)

        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.massmatrix = gymtorch.wrap_tensor(_massmatrix)

        # Full per-env rigid-body width (robot + any scene bodies); the robot occupies
        # the first num_bodies rows. Stored so force application (e.g. the virtual gantry)
        # can size tensors to the real buffer width, not just the robot's body count.
        self.bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, self.bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 10:13]

        # DOF forces
        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_forces = gymtorch.wrap_tensor(_dof_forces).view(self.num_envs, self.num_dof)

        self.refresh_sim_tensors()

        # _root_states_raw is the raw gym buffer passed to the C-API (gymtorch.unwrap_tensor)
        # and viewed by robot_root_states. all_root_states is the unified all-actors view.
        self._root_states_raw: Tensor = gymtorch.wrap_tensor(actor_root_state)
        self.all_root_states = UnifiedRootStatesView(self)  # type: ignore[assignment]
        num_actors = self._get_num_actors_per_env()

        # robot_root_states shares memory with the raw root-state buffer.
        self.robot_root_states = self._root_states_raw.view(self.num_envs, num_actors, actor_root_state.shape[-1])[
            ..., 0, :
        ]

        self.base_quat = self.robot_root_states[..., 3:7]  # isaacgym uses xyzw

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        # Slice to robot bodies only (the first num_bodies rows), matching _rigid_body_*
        # above and contact_forces_history below. The net-contact tensor spans all actor
        # bodies (robot + any spawned objects), so an unsliced view would be wider than
        # the robot-only history buffer.
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[
            :, : self.num_bodies, :
        ]  # shape: num_envs, num_bodies, xyz axis
        # To be compatible with isaacsim, we add the contact forces history
        self.contact_forces_history = torch.zeros(
            self.num_envs, self.simulator_config.contact_sensor_history_length, self.num_bodies, 3, device=self.device
        )
        # (num_envs, history_length, num_bodies, xyz axis), the first index is the most recent
        self.contact_forces_history[:, 0, :, :] = self.contact_forces.clone()  # deep copy

        # Initialize acceleration tensors ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            self.dof_acc = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.prev_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.base_linear_acc = torch.zeros(self.num_envs, 3, device=self.device)
            self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Apply each free object's configured initial velocity now that the root-state tensor
        # is acquired. Pose is already set at spawn (create_actor), so re-writing it (merged
        # with the velocity) is a cheap round-trip; set_actor_states wants the full 13-vector.
        free_names = self.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
        if free_names and self.num_envs > 0:
            env_ids = torch.arange(self.num_envs, device=self.device)
            poses = self.get_actor_initial_poses(free_names, env_ids)  # [n*num_envs, 7]
            vels = self.get_actor_initial_velocities(free_names, env_ids)  # [n*num_envs, 6]
            self.set_actor_states(free_names, env_ids, torch.cat([poses, vels], dim=1))

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0

    def _get_num_actors_per_env(self):
        return self._root_states_raw.shape[0] // self.num_envs
        # num_actors = (
        #     self.root_states.shape[0] - self.total_num_objects
        # ) // self.num_envs
        # return num_actors

    def apply_torques_at_dof(self, torques):
        """Apply torques with detailed logging to match MuJoCo implementation."""
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, pos_tensor):
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim, gymtorch.unwrap_tensor(force_tensor), gymtorch.unwrap_tensor(pos_tensor), gymapi.ENV_SPACE
        )

    def draw_debug_viz(self):
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def simulate_at_each_physics_step(self):
        if not hasattr(self, "step_counter"):
            self.step_counter = 0

        self.gym.simulate(self.sim)

        if self.sim_device == "cpu":
            self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)

        # Update accelerations ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            # Update DOF acceleration using numerical differentiation
            self.dof_acc = (self.dof_vel - self.prev_dof_vel) / self.sim_dt
            self.prev_dof_vel = self.dof_vel.clone()

            # Update base linear acceleration using numerical differentiation
            current_base_vel = self.robot_root_states[..., 7:10]
            self.base_linear_acc = (current_base_vel - self.prev_base_lin_vel) / self.sim_dt
            self.prev_base_lin_vel = current_base_vel.clone()

        # refresh force sensor tensor at each physics step (0.005s)
        self.gym.refresh_force_sensor_tensor(self.sim)
        if hasattr(self, "contact_forces_history") and hasattr(self, "contact_forces"):
            self.contact_forces_history = torch.cat(
                [self.contact_forces.clone().unsqueeze(1), self.contact_forces_history[:, :-1, :, :]], dim=1
            )

        self.step_counter += 1

    def setup_viewer(self):
        self.enable_viewer_sync = True
        self.visualize_viewer = True
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Camera tracking offsets (preserved when toggling tracking on)
        self.camera_tracking_offset = np.array([2.0, 0.0, 2.5])  # Default: behind and above robot
        self.camera_tracking_lookat_offset = np.array([0.0, 0.0, 0.0])  # Default: look at robot center

        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "backward_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "heading_left_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "heading_right_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "zero_command")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "toggle_camera_tracking")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "push_robots")

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_N, "next_task"
        # )
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "toggle_video_record")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "walk_stand_toggle")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "height_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "height_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "waist_yaw_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "waist_yaw_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "vis_force_range")

        # Virtual gantry commands
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "gantry_length_decrease")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "gantry_length_increase")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "gantry_toggle")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "gantry_force_adjust")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "gantry_force_sign_toggle")

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_UP, "force_left_up"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_DOWN, "force_left_down"
        # )

        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_LEFT, "force_right_down"
        # )
        # self.gym.subscribe_viewer_keyboard_event(
        #     self.viewer, gymapi.KEY_RIGHT, "force_right_up"
        # )

        sim_params = self.sim_params
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 3.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time=True):
        # No viewer in headless mode (setup_viewer() is skipped), so every viewer call below
        # would dereference a None handle. Skipping render() drops only the interactive
        # window and its debug overlay; the gantry's forces are applied in
        # simulate_at_each_physics_step(), independent of rendering.
        if self.viewer is None:
            return
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
            elif evt.action == "forward_command" and evt.value > 0:
                self.commands[:, 0] += 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "backward_command" and evt.value > 0:
                self.commands[:, 0] -= 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "left_command" and evt.value > 0:
                self.commands[:, 1] -= 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "right_command" and evt.value > 0:
                self.commands[:, 1] += 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "heading_left_command" and evt.value > 0:
                self.commands[:, 3] -= 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "heading_right_command" and evt.value > 0:
                self.commands[:, 3] += 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "zero_command" and evt.value > 0:
                self.commands[:, :4] = 0
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "toggle_camera_tracking" and evt.value > 0:
                was_enabled = self.simulator_config.viewer.enable_tracking
                self.simulator_config = dataclasses.replace(
                    self.simulator_config,
                    viewer=dataclasses.replace(self.simulator_config.viewer, enable_tracking=not was_enabled),
                )

                # If ENABLING tracking, capture current camera offset
                if self.simulator_config.viewer.enable_tracking and not was_enabled:
                    self._capture_camera_offset()

                status = "ON" if self.simulator_config.viewer.enable_tracking else "OFF"
                logger.info(f"Camera tracking: {status}")
            elif evt.action == "push_robots" and evt.value > 0:
                logger.info("Push Robots")
                self._push_robots(torch.arange(self.num_envs, device=self.device))
            # elif evt.action == "next_task" and evt.value > 0:
            #     self.next_task()
            elif evt.action == "walk_stand_toggle" and evt.value > 0:
                self.commands[:, 4] = 1 - self.commands[:, 4]
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "height_up" and evt.value > 0:
                self.commands[:, 8] += 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "height_down" and evt.value > 0:
                self.commands[:, 8] -= 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "waist_yaw_up" and evt.value > 0:
                self.commands[:, 5] += 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "waist_yaw_down" and evt.value > 0:
                self.commands[:, 5] -= 0.1
                logger.info(f"Current Command: {self.commands[:,]}")
            elif evt.action == "vis_force_range" and evt.value > 0:
                self.vis_force_range = 1 - self.vis_force_range
                logger.info(f"Vis force range: {self.vis_force_range}")
            # Virtual gantry commands
            elif evt.action == "gantry_length_decrease" and evt.value > 0:
                if self.virtual_gantry:
                    command_data = GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": -0.1})
                    self.virtual_gantry.handle_command(command_data)
            elif evt.action == "gantry_length_increase" and evt.value > 0:
                if self.virtual_gantry:
                    command_data = GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": 0.1})
                    self.virtual_gantry.handle_command(command_data)
            elif evt.action == "gantry_toggle" and evt.value > 0:
                if self.virtual_gantry:
                    command_data = GantryCommandData(GantryCommand.TOGGLE)
                    self.virtual_gantry.handle_command(command_data)
            elif evt.action == "gantry_force_adjust" and evt.value > 0:
                if self.virtual_gantry:
                    command_data = GantryCommandData(GantryCommand.FORCE_ADJUST)
                    self.virtual_gantry.handle_command(command_data)
            elif evt.action == "gantry_force_sign_toggle" and evt.value > 0:
                if self.virtual_gantry:
                    command_data = GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE)
                    self.virtual_gantry.handle_command(command_data)

        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # Update camera tracking if enabled
        if self.simulator_config.viewer.enable_tracking:
            # Get first robot's position
            robot_pos = self.robot_root_states[0, :3].cpu().numpy()

            # Use stored camera offset instead of hardcoded values
            cam_pos = gymapi.Vec3(*(robot_pos + self.camera_tracking_offset))
            cam_target = gymapi.Vec3(*(robot_pos + self.camera_tracking_lookat_offset))
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    def time(self) -> float:
        """Get current simulation time.

        Returns:
            float: Current simulation time in seconds
        """
        return self.gym.get_sim_time(self.sim)

    def get_dof_forces(self, env_id: int = 0):
        """Get DOF forces for a specific environment.

        This method provides access to measured joint forces from DOF force sensors.
        The sensors must be enabled via `enable_dof_force_sensors` in the configuration.

        Args:
            env_id: Environment index (default: 0)

        Returns:
            torch.Tensor: Tensor of shape [num_dof] with measured joint forces

        Raises:
            RuntimeError: If DOF force sensors are not enabled or forces not available
        """
        if not hasattr(self, "dof_forces"):
            raise RuntimeError(
                "DOF forces not available. Ensure 'enable_dof_force_sensors' is set to True "
                "in simulator.sim.physx configuration"
            )

        return self.dof_forces[env_id]

    def _capture_camera_offset(self):
        """Capture current camera position relative to robot.

        This method is called when toggling camera tracking ON to preserve
        the current camera angle/distance as the new tracking offset.
        """
        # Get current camera transform (4x4 homogeneous matrix)
        cam_transform = self.gym.get_viewer_camera_transform(self.viewer, None)

        # Extract camera position from transform
        cam_pos = np.array([cam_transform.p.x, cam_transform.p.y, cam_transform.p.z])

        # Get robot position
        robot_pos = self.robot_root_states[0, :3].cpu().numpy()

        # Calculate relative offset
        self.camera_tracking_offset = cam_pos - robot_pos

        # Keep lookat at robot center (simple approach)
        self.camera_tracking_lookat_offset = np.array([0.0, 0.0, 0.0])

        logger.info(f"Captured camera offset: {self.camera_tracking_offset}")

    def next_task(self):
        pass

    # debug visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
        """Convenience wrapper"""
        draw_sphere(self, pos, radius, color=color, env_id=env_id, num_lats=20, num_longs=20)

    def draw_line(self, start_point, end_point, color, env_id):
        """Convenience wrapper"""
        draw_line(self, start_point, end_point, color=color, env_id=env_id)

    def write_state_updates(self):
        """See base class.

        IsaacGym-specific notes:
        - No-op implementation - state changes are applied immediately through tensor APIs
        """
        # IsaacGym applies state changes immediately, so no pending updates to write

    def _register_objects(self):
        """Finalize per-actor gym indices, then register via the shared registry path."""
        # Convert gym_object_indices lists to tensors (IsaacGym-specific index map).
        for object_name in self.gym_object_indices:
            indices_list = self.gym_object_indices[object_name]
            self.gym_object_indices[object_name] = torch.tensor(indices_list, device=self.device, dtype=torch.long)
            logger.debug(f"Finalized gym_object_indices for '{object_name}': {self.gym_object_indices[object_name]}")

        self._register_scene_assets()

    @property
    def scene_file_static_names(self) -> set[str]:
        """Names of scene-file (1->N) bodies the URDF welds static (file-driven). Empty
        when the scene has no files, so the loader may be absent; guard its presence."""
        loader = getattr(self, "urdf_scene_loader", None)
        return set(loader.scene_file_static_names) if loader else set()

    def _collect_spawned_actors(self):
        """Describe each spawned actor's WORLD initial pose for the registry. IsaacGym's live
        root-state tensor (and thus get_actor_states) is world-frame (env_origins are baked in
        at create_actor() time), so the registry stores WORLD poses too, matching
        MuJoCo/IsaacSim's "world coordinates, env_origins already applied". This adds env_origins
        to each object's per-env position here, and set_actor_states writes its (world) input
        straight through with NO origin re-add. Object poses come from the loader (a scene file
        expands 1->N into object_handles); a body is static per its standalone ``fixed`` flag or
        the file's joint structure (scene_file_static_names), the rest are free. The robot's
        config pose is registered WORLD too (origins added), so ``get_actor_initial_poses`` is
        world-frame for every actor including "robot".
        """
        pos = torch.tensor(self.robot_config.init_state.pos, device=self.device, dtype=torch.float32)
        rot = torch.tensor(self.robot_config.init_state.rot, device=self.device, dtype=torch.float32)
        robot_pose = torch.cat([pos, rot]).unsqueeze(0).expand(self.num_envs, 7).clone()
        robot_pose[:, :3] += self.env_origins

        static_names = {
            name for name, obj in self.scene_config.rigid_objects.items() if obj.fixed
        } | self.scene_file_static_names
        # object_handles is a superset of scene_config.rigid_objects: it also holds scene-file
        # (1->N) bodies, which have no standalone config. Map name -> config so each spawned
        # handle can look up its configured velocity (None for scene-file bodies; zero for
        # fixed bodies, enforced by RigidObjectConfig).
        cfg_by_name = dict(self.scene_config.rigid_objects)
        items = []
        for obj_name in self.object_handles:
            base_pose = (
                torch.tensor(
                    self.urdf_scene_loader.get_initial_pose(obj_name)[:7], device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(self.num_envs, 7)
                .clone()
            )
            # Register WORLD poses: add each env's origin to the position columns (orientation is
            # origin-invariant). This is what makes get_actor_initial_poses return world coords and
            # lets set_actor_states accept world input directly, uniform across all backends.
            base_pose[:, :3] += self.env_origins
            obj = cfg_by_name.get(obj_name)
            velocity = (
                torch.tensor([*obj.linear_velocity, *obj.angular_velocity], device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.num_envs, 6)
                if obj is not None
                else None
            )
            items.append((obj_name, obj_name in static_names, base_pose, velocity))
        return robot_pose, items

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """Sets the **robot** state -- backwards compatible method

        Does NOT apply env origins
        Detects if all_root_states tensor is passed
        """

        if root_states is self.all_root_states:
            # Use robot states view directly
            self.set_actor_root_state_tensor_robots(set_env_ids, self.robot_root_states[set_env_ids])
        else:
            # Otherwise, assume it's already robot states
            self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_actor_root_state_tensor_robots(self, env_ids=None, robot_root_states=None):
        """Set robot root states (position/orientation) following IsaacGym best practices

        Args:
            env_ids: Optional[torch.Tensor] - Which environments (None = all)
            root_states: Optional[torch.Tensor] - Robot states to set (None = use current robot_root_states)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if robot_root_states is None:
            robot_root_states = self.robot_root_states[env_ids]

        robot_indices = self.robot_indices[env_ids]
        self._set_actor_root_state_tensor_by_index(robot_indices, robot_root_states)

    def set_dof_state_tensor_robots(self, env_ids=None, dof_states=None):
        """Set robot DOF states (joint positions/velocities) - IsaacGym format.

        This method sets robot joint positions and velocities using IsaacGym's
        flattened tensor format.

        Parameters
        ----------
        env_ids : torch.Tensor | None, default=None
            Which environments to update, shape [num_envs], dtype torch.long.
            If None, updates all environments.
        dof_states : torch.Tensor | None, default=None
            DOF states to set, shape [num_envs * num_dofs, 2], dtype torch.float32.
            **IsaacGym flattened format**: environments and DOFs are combined in first dimension.
            Format: [:, 0] = joint positions, [:, 1] = joint velocities.
            If None, uses current dof_state.

        Examples
        --------
        >>> # IsaacGym format: flattened [num_envs * num_dofs, 2]
        >>> env_ids = torch.tensor([0, 1], device=device)
        >>> num_selected_envs = len(env_ids)
        >>> dof_states = torch.zeros(num_selected_envs * sim.num_dof, 2, device=device)
        >>>
        >>> # Set positions and velocities in flattened format
        >>> positions_2d = torch.zeros(num_selected_envs, sim.num_dof, device=device)  # [envs, dofs]
        >>> dof_states[:, 0] = positions_2d.flatten()  # Flatten to [envs*dofs]
        >>> dof_states[:, 1] = 0.0  # Zero velocities
        >>> sim.set_dof_state_tensor_robots(env_ids, dof_states)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if dof_states is None:
            dof_states = self.dof_state

        # Convert robot indices to int32 as required by IsaacGym
        robot_indices = self.robot_indices[env_ids].to(torch.int32)

        # Call IsaacGym API with full tensor (like the original set_dof_state_tensor method)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(dof_states), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )

    def set_actor_states(self, names: ActorNames, env_ids: EnvIds, states: ActorStates, write_updates: bool = True):
        """Set actor states by name using IsaacGym's indexed API.

        Parameters
        ----------
        names : ActorNames
            List of actor names to update
        env_ids : EnvIds
            Environment IDs to update
        states : ActorStates
            New actor states [len(names) * num_envs, 13], **WORLD frame** (pass world coords
            directly, env_origins are NOT re-added).
        write_updates : bool
            Ignored for IsaacGym, writes are always immediate

        Notes
        -----
        ``names`` may include ``"robot"``: the robot's live root state is world-frame
        (origins baked in at create_actor), so a world-frame robot state writes straight
        through like an object.
        """
        if len(names) == 0 or len(env_ids) == 0:
            return

        actor_indices = self._get_gym_object_indices(names, env_ids)

        # `states` is WORLD-frame and IsaacGym's root-state tensor is world-frame (origins baked
        # in at create_actor), so write straight through with no env_origins re-add. `states` and
        # `actor_indices` are both NAME-MAJOR / ENV-MINOR (all envs for names[0], then names[1],
        # ...), aligned row-for-row, so a non-contiguous env_ids subset addresses exactly its rows.
        self._root_states_raw[actor_indices] = states
        self._set_actor_root_state_tensor_by_index(actor_indices, states)

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """Get actor indices for specified objects across environments.

        Parameters
        ----------
        names : str | ActorNames
            Name(s) of actors to get indices for
        env_ids : EnvIds | None
            Environment IDs to get indices for, None for all environments

        Returns
        -------
        ActorIndices
            Global indices for specified actors across environments
        """
        if isinstance(names, str):
            names = [names]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # NOTE: use len(), not truthiness; a tensor like tensor([0]) is falsy yet valid.
        if len(names) == 0 or len(env_ids) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        return self._get_gym_object_indices(names, env_ids)

    def get_actor_states(self, names: ActorNames, env_ids: EnvIds) -> ActorStates:
        """Get actor states using IsaacGym's indexed tensor access.

        Parameters
        ----------
        names : ActorNames
            List of actor names to query
        env_ids : EnvIds
            Environment IDs to query

        Returns
        -------
        ActorStates
            Actor states tensor [num_objects * num_envs, 13] with pose and velocity data
        """
        if len(names) == 0 or len(env_ids) == 0:
            return torch.empty(0, 13, device=self.device)

        indices = self.get_actor_indices(names, env_ids)
        # Read the raw buffer directly (indexing all_root_states would route back here).
        return self._root_states_raw[indices]

    def _get_gym_object_indices(self, objects: list[str], env_ids: torch.Tensor) -> torch.Tensor:
        """Get IsaacGym actor indices for objects across environments.

        Parameters
        ----------
        objects : list[str]
            List of object names to get indices for
        env_ids : torch.Tensor
            Environment IDs to query

        Returns
        -------
        torch.Tensor
            Actual actor indices for specified objects and environments
        """
        all_indices: list[torch.Tensor] = []

        for obj_name in objects:
            if obj_name not in self.gym_object_indices:
                available = list(self.gym_object_indices.keys())
                raise KeyError(f"Object '{obj_name}' not found in gym_object_indices. Available: {available}")

            # Get the actual actor indices for this object (one per environment)
            obj_actor_indices: list[torch.Tensor] = self.gym_object_indices[obj_name]  # [num_envs]

            # Select only the requested environments
            selected_indices = obj_actor_indices[env_ids]  # [len(env_ids)]

            all_indices.append(selected_indices)

        # Concatenate all object indices
        result = torch.cat(all_indices) if len(all_indices) > 1 else all_indices[0]
        logger.debug(f"Final gym indices for {objects} in envs {env_ids}: {result}")
        return result

    def get_actor_initial_poses(self, names: ActorNames, env_ids: EnvIds | None = None) -> ActorPoses:
        """Get initial poses for actors using enhanced ObjectRegistry tensor interface.

        This method uses the new ObjectRegistry tensor-based interface where callers
        provide pre-calculated world coordinates during registration. This eliminates
        runtime coordinate transformations and provides consistent tensor lookup.

        Parameters
        ----------
        names : ActorNames
            List of actor names to get poses for
        env_ids : EnvIds | None, default=None
            Environment IDs to get poses for, None for all environments

        Returns
        -------
        ActorPoses
            Initial poses tensor [len(objects) * len(env_ids), 7] with position and quaternion
            Format: [x, y, z, qx, qy, qz, qw] per pose
        """
        if not names:
            return torch.empty(0, 7, device=self.device, dtype=torch.float32)

        # Determine which environments to use
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Use ObjectRegistry tensor-based interface for fast lookup
        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def get_actor_initial_velocities(self, names: ActorNames, env_ids: EnvIds | None = None) -> torch.Tensor:
        """Get initial velocities for actors (the sibling of get_actor_initial_poses).

        Returns [len(names) * len(env_ids), 6] world-frame [vx,vy,vz,wx,wy,wz], same row order as
        get_actor_initial_poses (so the two concatenate into the 13-vector set_actor_states wants).
        """
        if not names:
            return torch.empty(0, 6, device=self.device, dtype=torch.float32)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        return self.object_registry.get_initial_velocities_batch(names, env_ids)

    def apply_physics_properties_to_actor(self, env_ptr: int, actor_handle: int, object_name: str) -> None:
        """Apply physics properties to an actor.

        Works for both scene and individual objects by looking up physics
        configurations and applying them to the specified actor. This includes
        rigid shape properties (friction, restitution) and mass properties.

        Parameters
        ----------
        env_ptr : int
            Environment handle for the IsaacGym environment.
        actor_handle : int
            Handle to the actor to apply physics properties to.
        object_name : str
            Name of the object for configuration lookup and logging.

        Returns
        -------
        None
            Returns None if no physics configurations are available.
        """
        if not self.urdf_scene_loader.object_physics_configs:
            return

        # object_physics_configs holds live PhysicsConfig objects (always truthy), so test
        # for absence with `is None`, not falsiness.
        physics_config = self.urdf_scene_loader.object_physics_configs.get(object_name, None)
        if physics_config is None:
            return

        apply_rigid_shape_properties(self.gym, env_ptr, actor_handle, physics_config, object_name)
        apply_mass_from_config(self.gym, env_ptr, actor_handle, physics_config, object_name)
        logger.debug(f"Applied physics properties to '{object_name}'")

    def _set_actor_root_state_tensor_by_index(self, actor_indices, states):
        """Reset specific actors by their actual indices using IsaacGym's indexed API"""

        # Convert indices to int32 as required by IsaacGym
        actor_indices_int32 = actor_indices.to(torch.int32)

        # Pass the full raw tensor to IsaacGym - it will only update the specified indices
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states_raw),  # Full raw buffer
            gymtorch.unwrap_tensor(actor_indices_int32),
            len(actor_indices_int32),
        )

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """See base class.

        The raw root-state buffer spans all actors in world frame, so indexing it returns the
        13-vector state for any actor (robot or object).
        """
        if len(indices) == 0:
            return torch.empty(0, 13, device=self.device)
        return self._root_states_raw[indices]

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """See base class.

        Writes into the raw world-frame buffer and flushes via the indexed gym API.
        ``write_updates`` is ignored; IsaacGym writes are immediate.
        """
        if len(indices) == 0:
            return
        self._root_states_raw[indices] = states
        self._set_actor_root_state_tensor_by_index(indices, states)
