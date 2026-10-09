from __future__ import annotations

import builtins
import copy
import dataclasses
import xml.etree.ElementTree as ET
from typing import Any

import pathlib
import trimesh

from holosoma.config_types.full_sim import FullSimConfig
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaacsim.core.utils.stage as stage_utils
import omni.log
import torch
from pxr import Usd
from isaaclab.assets import Articulation
from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import EventManager, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, RayCaster, TiledCamera
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.sim.utils import bind_physics_material
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.timer import Timer
from loguru import logger
from omegaconf import DictConfig

from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.config_types.simulator import SimulatorInitConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.isaacsim.event_cfg import EventCfg
from holosoma.simulator.isaacsim.events import randomize_body_com, randomize_rigid_body_inertia
from holosoma.simulator.isaacsim import light_setup, sensor_setup
from holosoma.simulator.isaacsim.viewer_input import ViewerInputController
from holosoma.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import (
    build_contact_sensor_cfg,
    build_robot_articulation_cfg,
)
from holosoma.simulator.isaacsim.object_spawner import (
    build_standalone_rigid_object,
    expand_scene_file,
    physics_material_cfg,
)
from holosoma.simulator.isaacsim.scene_fixtures import build_terrain_fixtures
from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
from holosoma.simulator.shared.dof_limits import build_dof_limits_from_config
from holosoma.simulator.shared.external_wrench import WrenchTarget
from holosoma.simulator.shared.root_states_view import UnifiedRootStatesView
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter
from holosoma.simulator.isaacsim.prim_utils import log_robot_properties, print_prim_tree
from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder
from holosoma.simulator.shared.virtual_gantry import (
    VirtualGantry,
    create_virtual_gantry,
    GantryCommand,
    GantryCommandData,
)

from holosoma.simulator.types import ActorNames, ActorIndices, EnvIds, ActorStates, ActorPoses


class IsaacSim(BaseSimulator):
    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str):
        super().__init__(tyro_config, terrain_manager, device)

        # Public interface attribute read across backends (video_recorder, virtual_gantry,
        # simulator_bridge). For IsaacSim it always equals self.sim_device; internal tensor
        # allocations below use self.sim_device directly.
        self.device = device

        # Names (prefixed {file}_{body}) of scene-file bodies spawned static (kinematic),
        # populated by _load_scene_files; read in _collect_spawned_actors to classify.
        self.scene_file_static_names: set[str] = set()

        # Cameras here are read out of the periodic render pass, which draws when the substep counter
        # reaches render_interval. A driving loop closes its frame on the control step's last substep,
        # so only a render_interval dividing control_decimation puts the draw and the frame's state on
        # the same substep; anything else pairs an image with a pose from a different one. Checked
        # before the SimulationContext exists, so a rejected config leaves no global context behind.
        decimation = self.simulator_config.sim.control_decimation_steps
        render_interval = self.simulator_config.sim.render_interval_steps
        cameras = sorted(n for n, c in self.sensor_config.items() if isinstance(c, CameraSensorConfig))
        if cameras and decimation % render_interval:
            raise ValueError(
                f"render_interval ({render_interval}) must divide control_decimation ({decimation}) "
                f"when cameras are configured ({cameras}); set it to a divisor, or to "
                f"control_decimation to render once per control step."
            )

        sim_config: SimulationCfg = SimulationCfg(
            dt=1.0 / self.simulator_config.sim.fps,
            render_interval=self.simulator_config.sim.render_interval_steps,
            device=self.sim_device,
            physx=PhysxCfg(
                bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                solver_type=self.simulator_config.sim.physx.solver_type,
                max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
            # Global physics material, can be overridden by the individual articulation
            # Can be inspected by:
            # materials = self._robot.root_physx_view.get_material_properties()
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # default is 0.5
                dynamic_friction=1.0,  # default is 0.5
                restitution=0.0,
            ),
        )

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1.0 / self.simulator_config.sim.fps}")
        logger.info(
            f"\tRendering step-size   : {1.0 / self.simulator_config.sim.fps * self.simulator_config.sim.substeps}"
        )

        if self.simulator_config.sim.render_interval_steps < self.simulator_config.sim.control_decimation_steps:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval_steps}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation_steps}). Multiple render calls will happen for each "
                "environment step. If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.training_config.num_envs,
            env_spacing=self.scene_config.env_spacing,
            replicate_physics=self.scene_config.replicate_physics,
        )
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # Narrow the type from the base class's SceneInterface (which declares only
            # env_origins) to IsaacLab's InteractiveScene, what self.scene actually is on this
            # backend, so the rich .rigid_objects/.articulations/.sensors/... accesses below
            # type-check. The base protocol stays correct for MuJoCo/IsaacGym, whose scenes
            # implement only env_origins.
            self.scene: InteractiveScene = InteractiveScene(scene_config)
            # Per-camera TiledCamera map, keyed by sensor name; populated by
            # sensor_setup.build_cameras (called from _setup_scene, pre-clone), read at render.
            self.tiled_cameras: dict[str, TiledCamera] = {}
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        viewer_config: ViewerCfg
        if self.simulator_config.viewer.enable_tracking:
            viewer_config = ViewerCfg(origin_type="asset_root", asset_name="robot", eye=(0.0, -1.5, 1.5))
        else:
            viewer_config = ViewerCfg()

        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller: ViewportCameraController | None = ViewportCameraController(
                self, viewer_config
            )
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore[attr-defined]
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()

        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.training_config.num_envs, 3), dtype=torch.float, device="cpu")

        self.events_cfg = EventCfg()

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)

        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None

        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of
        # the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation

        self._sim_step_counter = 0

        # Set True by the keyboard controller's push-robots key; polled by external tooling.
        self.push_requested: bool = False

        if self.video_config.enabled:
            self.video_recorder = IsaacSimVideoRecorder(self.video_config, self)

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()

        # print the environment information

        logger.info("Completed setting up the environment...")

    def _bind_robot_link_material(self) -> None:
        """Bind the robot's ``link_physics.isaacsim`` as a ``RigidBodyMaterialCfg`` on every link.

        Same mechanism as a scene object's material bind (``object_spawner.physics_material_cfg``
        with ``bind_physics_material``), applied to the robot articulation: spawn one material prim
        under the env_0 robot and bind it to the robot prim. ``bind_physics_material`` is
        ``@apply_nested``, so the single bind reaches every link's collider. Run before
        ``clone_environments`` (the clone copies the authored env_0 prim, material binding included),
        so all envs inherit it.

        A material prim is used rather than the per-shape ``root_physx_view.set_material_properties``
        tensor because the friction/restitution combine modes live on the PhysX material prim
        (``physxMaterial:frictionCombineMode``), which the per-shape float table cannot carry. Binding
        the material authors the combine modes too, so the robot honors ``friction_combine_mode`` and
        ``restitution_combine_mode`` as an object does. No-op when ``link_physics.isaacsim`` is unset
        (the links keep the asset's authored or global-default material).
        """
        link_physics = self.robot_config.asset.link_physics
        mat_cfg = physics_material_cfg(link_physics) if link_physics is not None else None
        if mat_cfg is None:
            return
        robot_prim_path = "/World/envs/env_0/Robot"
        material_path = f"{robot_prim_path}/linkPhysicsMaterial"

        # The URDF-converted robot authors its colliders as instanceable prims; bind_physics_material
        # (and its @apply_nested walk) no-ops on instance proxies, so the material never reaches the
        # real collider and PhysX keeps the default per-shape material (the failure objects avoid via
        # CustomUsdFileCfg.disable_instanceable). Clear instanceable across the robot subtree first so
        # the bind reaches the actual collider prims.
        stage = stage_utils.get_current_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        mat_cfg.func(
            material_path, mat_cfg
        )  # spawn_rigid_body_material: authors friction/restitution and combine modes
        bind_physics_material(robot_prim_path, material_path)  # @apply_nested over every link collider
        logger.info(
            f"Bound robot link material: static_friction={mat_cfg.static_friction} "
            f"dynamic_friction={mat_cfg.dynamic_friction} restitution={mat_cfg.restitution} "
            f"friction_combine={mat_cfg.friction_combine_mode} restitution_combine={mat_cfg.restitution_combine_mode}"
        )

    def _setup_scene(self) -> None:
        self._load_scene_config()

        robot_articulation_config = build_robot_articulation_cfg(self.robot_config)
        contact_sensor_config = build_contact_sensor_cfg(
            self.simulator_config.contact_sensor_history_length, self.debug_viz_enabled
        )

        self._robot = Articulation(robot_articulation_config)

        # The robot must be spawned before fixtures and mounted sensors resolve imported link prims.
        # URDF conversion may preserve a dummy root link between /Robot and the configured bodies.
        global_collision_prims, height_scanner_config, self.terrain = build_terrain_fixtures(
            self.terrain_manager, self.scene, self.robot_config.body_names[0]
        )

        # Bind the robot's link_physics friction/restitution material to every link, before clone so
        # all envs inherit it (the clone copies env_0's authored prim). Authors the combine modes too
        # (see _bind_robot_link_material). No-op unless link_physics.isaacsim is set.
        self._bind_robot_link_material()

        print_prim_tree("/World/envs/env_0/Robot")
        log_robot_properties("/World/envs/env_0/Robot", "*")

        self.scene.articulations["robot"] = self._robot

        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        if height_scanner_config:
            self._height_scanner = RayCaster(height_scanner_config)
            self.scene.sensors["height_scanner"] = self._height_scanner

        # Perception cameras: one TiledCamera per configured camera, as a child prim of its
        # mount body so it follows that body, created before clone so it replicates per env.
        sensor_setup.build_sensors(self)

        # Author the light rig under env_0 before cloning so each environment gets its own lights; an
        # empty rig authors nothing, leaving only the scene asset's baked lights.
        light_setup.build_lights(self)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        self.scene.filter_collisions(global_prim_paths=global_collision_prims)

        # Register the TiledCameras (built pre-clone above) into the shared SensorManager. Done
        # here at the end of scene build (IsaacSim builds its scene in __init__, not load_assets).
        sensor_setup.create_sensors(self)

    # ----- Camera sensors (TiledCamera; child prim of the mount body, auto-follow) -----

    def render_sensors(self) -> None:
        """Capture all due mounted sensors into their cached buffers."""
        sensor_setup.render_sensors(self)

    def _get_base_body_name(self, preference_order: list[str]) -> str:
        """Get the base body name with fallback logic.

        Args:
            preference_order: List of body names to try in order

        Returns:
            The first body name found in the robot's body list

        Raises:
            ValueError: If none of the preferred body names are found
        """
        _, body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        for preferred_name in preference_order:
            if preferred_name in body_names:
                return preferred_name

        raise ValueError(
            f"None of the preferred base body names {preference_order} found in robot body names: {body_names}"
        )

    def get_supported_scene_formats(self) -> list[str]:
        """See base class.

        IsaacSim-specific notes:
        - Supports USD as the preferred format.
        - Also supports URDF, which isaacsim will internally translate to USD.

        Returns
        -------
        List[str]
            ["usd", "urdf"]
        """
        return ["usd", "urdf"]

    def set_headless(self, headless: bool) -> None:
        # call super
        super().set_headless(headless)
        if not self.headless:
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def _load_scene_config(self) -> None:
        """Load scene configuration with proper separation of concerns.

        Handles both scene files (collections) and individual rigid objects.
        Replaces the previous _load_scene_usd method with a more flexible approach
        that supports multiple scene file formats and individual object loading.
        """
        if self.scene_config is None:
            return  # type: ignore[unreachable]

        # Load scene files (USD/URDF scene files as collections)
        self._load_scene_files(self.scene_config)

        # Load individual rigid objects
        self._load_rigid_objects(self.scene_config)

    def _load_scene_files(self, scene_config: SceneConfig) -> None:
        """Load multi-body scene files (1->N) by expanding each into per-body RigidObjects.

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration containing scene files and asset root path
        """
        for scene_file_name, scene_file in scene_config.scene_files.items():
            objects, static_names = expand_scene_file(
                scene_file_name, scene_file, self.get_supported_scene_formats(), scene_config.asset_root
            )
            self.scene.rigid_objects.update(objects)
            self.scene_file_static_names |= static_names

    def _load_rigid_objects(self, scene_config: SceneConfig) -> None:
        """Load standalone rigid objects (1->1), USD or URDF, through the unified spawner.

        Format is chosen per object by ``select_asset_format`` (USD preferred, URDF fallback;
        an object providing neither fails loud); both formats build the identical RigidObject
        via object_spawner; the only difference is the spawn cfg.
        """
        for obj_name, obj in scene_config.rigid_objects.items():
            self.scene.rigid_objects[obj_name] = build_standalone_rigid_object(
                obj_name, obj, self.get_supported_scene_formats(), scene_config.asset_root
            )

    def setup(self) -> None:
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self) -> None:
        pass

    def load_assets(self) -> None:
        """
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        """

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        # isaacsim only support matching joint names without "joint" postfix

        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 1.05),
        #     joint_pos={
        #         ".*_hip_yaw": 0.0,
        #         ".*_hip_roll": 0.0,
        #         ".*_hip_pitch": -0.28,  # -16 degrees
        #         ".*_knee": 0.79,  # 45 degrees
        #         ".*_ankle": -0.52,  # -30 degrees
        #         "torso": 0.0,
        #         ".*_shoulder_pitch": 0.28,
        #         ".*_shoulder_roll": 0.0,
        #         ".*_shoulder_yaw": 0.0,
        #         ".*_elbow": 0.52,
        #     },
        #     joint_vel={".*": 0.0},
        # ),

        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        #     ),
        # ),

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, holosoma config)
        # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        """
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19],
        ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link',
        'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',
        'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link',
        'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link',
        'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link',
        'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link',
        'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link',
        'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link',
        'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        """

        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning(
                "The order of the joint_names in the robot_config does not match the "
                "order of the joint_ids in IsaacSim."
            )

        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), (
            "Number of bodies must be equal to number of body names"
        )
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

        self._contact_to_robot_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(body_name) for body_name in self.body_names],
            device=self.sim_device,
        )

        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(  # type: ignore[override]
        self, num_envs: int, env_origins: torch.Tensor, base_init_state: torch.Tensor
    ) -> tuple[InteractiveScene, Articulation]:
        self.num_envs = num_envs
        # IsaacSim does NOT honor the passed env_origins: InteractiveScene clones the envs on its own
        # env_spacing grid (built in __init__), and every internal placement (robot/object poses,
        # terrain) uses self.scene.env_origins, not the argument. Storing the passed value here left
        # self.env_origins disagreeing with where envs actually are (unlike mujoco/isaacgym, where the
        # passed origins ARE the placement) — so callers reading sim.env_origins (e.g. a multi-env
        # camera harness pinning a robot relative to its env) landed off from the grid-placed scene.
        # Reconcile to the real grid so sim.env_origins means the same thing on every backend.
        self.env_origins = self.scene.env_origins
        self.base_init_state = base_init_state

        return self.scene, self._robot

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            self.hard_dof_pos_limits,
            self.dof_pos_limits,
            self.dof_vel_limits,
            self.torque_limits,
        ) = build_dof_limits_from_config(self.robot_config, self.num_dof, self.sim_device)
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order

        Raises
        ------
        RuntimeError
            If the body name is not found, or matches more than one body.
        """
        raw_indices, names = self._robot.find_bodies(body_name)
        indices: list[int] = [self.body_ids.index(i) for i in raw_indices]
        if len(indices) == 0:
            raise RuntimeError(f"Body '{body_name}' not found in robot bodies.")
        if len(indices) > 1:
            raise RuntimeError(f"Multiple bodies found for '{body_name}': {names}.")
        return indices[0]

    def _collect_spawned_actors(
        self,
    ) -> tuple[torch.Tensor, list[tuple[str, bool, torch.Tensor, torch.Tensor | None]]]:
        """IsaacSim describes WORLD poses (env_origins added). Every spawned body lives under its
        own name in ``scene.rigid_objects``: both standalone objects AND multi-body scene-file
        bodies (1->N, named ``{file}_{body}``). A body is static if its standalone ``fixed`` flag
        is set or it is in ``scene_file_static_names`` (the file marked it kinematic); otherwise
        free.
        """
        # Robot: config base pose with env_origins added (world coordinates).
        base = torch.cat(
            [
                torch.tensor(self.robot_config.init_state.pos, device=self.sim_device, dtype=torch.float32),
                torch.tensor(self.robot_config.init_state.rot, device=self.sim_device, dtype=torch.float32),
            ]
        )
        robot_pose = base.unsqueeze(0).repeat(self.num_envs, 1)
        robot_pose[:, :3] += self.scene.env_origins

        static_names = {
            name for name, obj in self.scene_config.rigid_objects.items() if obj.fixed
        } | self.scene_file_static_names
        # scene.rigid_objects is a superset of scene_config.rigid_objects: it also holds
        # scene-file (1->N) bodies, which have no standalone config. Map name -> config so each
        # spawned body can look up its configured velocity (None for scene-file bodies; zero for
        # fixed bodies, enforced by RigidObjectConfig).
        cfg_by_name = dict(self.scene_config.rigid_objects)
        names = list(self.scene.rigid_objects.keys())  # InteractiveScene always exposes this dict (empty if none)
        items = []
        if names:
            # Per-env world poses for every object, reshaped to [n, num_envs, 7] (rows are
            # obj-major then env, matching get_actor_initial_poses' ordering). Registering the full
            # per-env block (not env 0 repeated) keeps a static body's registry pose at its true
            # per-env world position, so its read-back matches where the clone actually sits.
            per_env = self.get_actor_initial_poses(names).view(len(names), self.num_envs, 7)
            for name, poses in zip(names, per_env):
                obj = cfg_by_name.get(name)
                velocity = (
                    torch.tensor(
                        [*obj.linear_velocity, *obj.angular_velocity], device=self.sim_device, dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .repeat(self.num_envs, 1)
                    if obj is not None
                    else None
                )
                items.append((name, name in static_names, poses, velocity))
        return robot_pose, items

    def prepare_sim(self) -> None:
        # Wait until play so rigid object collections are initialized
        self._register_scene_assets()

        # Create before state adapter, needs a reference
        self.robot_root_states = RootStatesProxy(self._robot.data.root_state_w)  # (num_envs, 13)

        # Create state adapter after object registry and robot root states are set
        self._state_adapter = IsaacSimStateAdapter(
            device=self.sim_device,
            object_registry=self.object_registry,
            scene=self.scene,
            robot=self._robot,
            robot_states=self.robot_root_states,
        )

        # Unified all-actors view: routes [indices] reads/writes through
        # get/set_actor_states_by_index, which delegate to the state adapter.
        self.all_root_states = UnifiedRootStatesView(self)

        self.contact_forces_history = torch.zeros(
            self.num_envs,
            self.simulator_config.contact_sensor_history_length,
            self.num_bodies,
            3,
            device=self.sim_device,
        )

        # Initialize virtual gantry system after object registry setup
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

        # Setup video recording after scene is ready
        if self.video_recorder:
            self.video_recorder.register_hooks(self.hooks)
            self.video_recorder.setup_recording()

        # Initialize robot tensors
        self.refresh_sim_tensors()

        # Initialize acceleration tensors ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            logger.info("Bridge enabled: initializing acceleration computation tensors")
            self.dof_acc = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)
            self.prev_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)
            self.base_linear_acc = torch.zeros(self.num_envs, 3, device=self.sim_device)
            self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.sim_device)
        else:
            logger.debug("Bridge disabled: skipping acceleration computation tensors")

        # Apply each free object's configured initial velocity now that the scene is played and
        # the state adapter is live. Pose is already set at spawn (InitialStateCfg), so re-writing
        # it (merged with the velocity) is a cheap round-trip; set_actor_states wants the full 13-vector.
        free_names = self.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
        if free_names and self.num_envs > 0:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)
            poses = self.get_actor_initial_poses(free_names, env_ids)  # [n*num_envs, 7]
            vels = self.get_actor_initial_velocities(free_names, env_ids)  # [n*num_envs, 6]
            self.set_actor_states(free_names, env_ids, torch.cat([poses, vels], dim=1))

    @property
    def dof_state(self) -> torch.Tensor:
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self) -> None:
        # Apply reset to recache new wyxz -> xyzw tensor
        self.robot_root_states.reset(self._robot.data.root_state_w)  # (num_envs, 13)

        self.base_quat = self.robot_root_states[:, 3:7]  # (num_envs, 4), xyzw
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # The body ordering of contact_sensor is different from the body ordering of the robot.
        self.contact_forces = self.contact_sensor.data.net_forces_w[
            :, self._contact_to_robot_body_ids
        ]  # (num_envs, num_bodies, 3)

        # Issue: data.net_forces_w_history is not cleared after a reset.
        # Solution: We only read the most recent decimation_factor steps.
        control_decimation = self.simulator_config.sim.control_decimation_steps
        effective_history_length = min(control_decimation, self.simulator_config.contact_sensor_history_length)
        self.contact_forces_history[:, :effective_history_length, :, :] = self.contact_sensor.data.net_forces_w_history[
            :, :effective_history_length, self._contact_to_robot_body_ids
        ]  # (num_envs, history_length, num_bodies, 3), the first index is the most recent

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][
            :, :, [1, 2, 3, 0]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        # Match rigid_body_pos_w's link-frame origin. IsaacLab's legacy body_lin_vel_w alias is
        # COM velocity, while body_link_lin_vel_w applies the required omega x (-COM offset).
        self._rigid_body_vel = self._robot.data.body_link_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        # set_joint_effort_target writes only the given joint_ids without clobbering the rest, so a
        # subset (dof_indices) composes natively with a co-controller on the other DOFs.
        if dof_indices is None:
            joint_ids = self.dof_ids
        else:
            joint_ids = [self.dof_ids[i] for i in dof_indices]
        self._robot.set_joint_effort_target(torques, joint_ids=joint_ids)

    def draw_debug_viz(self) -> None:
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def _step_dynamics(self) -> None:
        self._sim_step_counter += 1
        # Only render if actively recording (not just if video recorder exists)
        has_video_recording = self.video_recorder is not None and self.video_recorder.is_recording
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors() or has_video_recording

        # Buffers external forces onto the articulation / object handles; write_data_to_sim() below
        # flushes them into PhysX before sim.step().
        self.flush_external_wrench()

        self.scene.write_data_to_sim()

        render_now = is_rendering and self._sim_step_counter % self.simulator_config.sim.render_interval_steps == 0

        # Physics: exactly one step, never through the renderer. sim.step(render=True) falls
        # through to a bare app.update() in isaacsim.core, which also runs the physics engine on
        # the app clock — sim time ran ~3x too fast whenever a camera/GUI was active.
        self.sim.step(render=False)

        if render_now:
            self.render()

        # update buffers at sim
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

        # Need to update these tensors after each step, since they are used in `_apply_force_in_physics_step`
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # Update accelerations ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            # Update DOF acceleration using numerical differentiation
            self.dof_acc = (self.dof_vel - self.prev_dof_vel) / self.sim_dt
            self.prev_dof_vel = self.dof_vel.clone()

            # Update base linear acceleration using numerical differentiation
            current_base_vel = self.robot_root_states[:, 7:10]
            self.base_linear_acc = (current_base_vel - self.prev_base_lin_vel) / self.sim_dt
            self.prev_base_lin_vel = current_base_vel.clone()

    def forward_kinematics(self) -> None:
        """Flush pending state writes and update articulation kinematics, fabric, and the render
        products — without stepping physics. ``sim.current_time`` does not advance."""
        self._sim_step_counter += 1
        has_video_recording = self.video_recorder is not None and self.video_recorder.is_recording
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors() or has_video_recording

        self.scene.write_data_to_sim()
        # sim.forward() runs physics_sim_view.update_articulations_kinematic() and flushes
        # fabric, so body/camera prims match the written joint/root state without integrating.
        self.sim.forward()

        # Same render gate as _step_dynamics: RTX sensors (TiledCameras) only track the new
        # poses after a render tick.
        render_now = is_rendering and self._sim_step_counter % self.simulator_config.sim.render_interval_steps == 0
        if render_now:
            self.sim.render()
            if self.debug_viz_enabled:
                self.clear_lines()
                self.draw_debug_viz()

        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

    def _write_external_wrench_native(self, targets: list[WrenchTarget]) -> None:
        """Set world-frame wrenches on the robot articulation and object RigidObjects.

        set_external_force_and_torque applies in the BODY-LOCAL frame (IsaacLab hardcodes/defaults
        is_global=False), so world wrenches are rotated per body via quat_apply_inverse first.
        Only write_cols is written: the composer SETs per body_id, so touching every body would
        clobber forces another writer applied to bodies we don't own. scene.write_data_to_sim()
        (in simulate_at_each_physics_step) flushes into PhysX.
        """
        from isaaclab.utils.math import quat_apply_inverse

        dev = self.sim_device
        # Runs every physics substep with a stable body set: cache the resolved id lists/tensors.
        if not hasattr(self, "_wrench_id_cache"):
            self._wrench_env_ids_all = torch.arange(self.num_envs, device=dev)
            self._wrench_id_cache: dict[tuple[str, tuple[int, ...]], tuple[list[int], torch.Tensor]] = {}
        for target in targets:
            # Robot: the articulation, holosoma body columns mapped to raw isaaclab ids.
            # Object: its single-body RigidObject handle, body 0 (body_ids is ignored by IsaacLab).
            handle = self._robot if target.is_robot else self.scene.rigid_objects[target.actor_name]
            cols = target.write_cols
            key = (target.actor_name, tuple(cols))
            if key not in self._wrench_id_cache:
                ids = [self.body_ids[col] for col in cols] if target.is_robot else [0]
                self._wrench_id_cache[key] = (ids, torch.tensor(ids, device=dev))
            ids, ids_t = self._wrench_id_cache[key]
            wrench = target.wrench.to(device=dev, dtype=torch.float32)[:, cols, :]  # [num_envs, n_cols, 6]
            body_quats = handle.data.body_quat_w[:, ids, :]  # wxyz, [num_envs, n_cols, 4]
            handle.set_external_force_and_torque(
                forces=quat_apply_inverse(body_quats, wrench[..., 0:3]),
                torques=quat_apply_inverse(body_quats, wrench[..., 3:6]),
                env_ids=self._wrench_env_ids_all,
                body_ids=ids_t,
            )

    def setup_viewer(self) -> None:
        self.viewer = self.viewport_camera_controller

        # Initialize commands tensor if not already done
        if not hasattr(self, "commands"):
            self.commands = torch.zeros((self.training_config.num_envs, 12), device=self.sim_device)

        # Set up viewer input (owns the carb.input subscription for its lifetime)
        if self.viewport_camera_controller is not None:
            self.viewer_input = ViewerInputController(self)

    def render(self, sync_frame_time: bool = True) -> None:
        self.sim.render()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    # debug visualization - delegate to draw adapter
    def clear_lines(self) -> None:
        """Delegate to draw adapter."""
        from holosoma.utils.draw import clear_lines

        clear_lines(self)

    def draw_sphere(
        self, pos: torch.Tensor, radius: float, color: torch.Tensor, env_id: int, pos_id: int | None
    ) -> None:
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_sphere

        draw_sphere(self, pos, radius, color, env_id, pos_id)

    def draw_line(self, start_point: torch.Tensor, end_point: torch.Tensor, color: torch.Tensor, env_id: int) -> None:
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_line

        draw_line(self, start_point, end_point, color, env_id)

    def set_actor_root_state_tensor_robots(
        self, env_ids: EnvIds | None = None, root_states: torch.Tensor | None = None
    ) -> None:
        """See base class.

        IsaacSim-specific notes:
        - Quaternions converted from (x,y,z,w) to (w,x,y,z) format for IsaacSim compatibility
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if root_states is None:
            robot_root_states = self.robot_root_states
        elif root_states is self.all_root_states:
            # Wholesale-pass of the unified view: write the robot only.
            robot_root_states = self.robot_root_states
        elif isinstance(root_states, RootStatesProxy):
            # assumes the user passed in robot_root_states directly
            robot_root_states = root_states
        else:
            raise ValueError(f"Unexpected root states type: {type(root_states)}")

        self._robot.write_root_pose_to_sim(robot_root_states._get_wxyz(env_ids)[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(robot_root_states._get_wxyz(env_ids)[:, 7:], env_ids)

    def set_dof_state_tensor_robots(
        self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None
    ) -> None:
        """See base class.

        IsaacSim-specific notes:
        - Tensor format: 3D [num_envs, num_dofs, 2] (differs from IsaacGym's flattened format)

        Examples
        --------
        >>> # IsaacSim format: 3D [num_envs, num_dofs, 2]
        >>> env_ids = torch.tensor([0, 1], device=device)
        >>> dof_states = torch.zeros(len(env_ids), sim.num_dof, 2, device=device)
        >>> dof_states[:, :, 0] = default_joint_positions  # 2D positions [envs, dofs]
        >>> dof_states[:, :, 1] = 0.0  # Zero velocities
        >>> sim.set_dof_state_tensor_robots(env_ids, dof_states)
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state

        dof_pos, dof_vel = dof_states[env_ids, :, 0], dof_states[env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, env_ids)

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """See base class."""
        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """See base class.

        Delegates to the state adapter (resolve indices -> per-object read -> wxyz->xyzw ->
        concatenated [N, 13]). Does not index ``all_root_states``, which would recurse.
        """
        return self._state_adapter.get_states_by_index(indices)

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """See base class.

        Delegates to the state adapter (xyzw->wxyz, per-object pose/velocity write, sets the dirty
        flag). ``write_updates=True`` (default) flushes via :meth:`write_state_updates`
        (``scene.write_data_to_sim``); ``False`` defers for batching.
        """
        self._state_adapter.write_states_by_index(indices, states)
        if write_updates:
            self.write_state_updates()

    def get_actor_states(self, names: ActorNames, env_ids: EnvIds) -> ActorStates:
        """See base class."""
        if not names or (env_ids is not None and len(env_ids) == 0):
            return torch.empty(0, 13, device=self.sim_device)
        return self.get_actor_states_by_index(self.get_actor_indices(names, env_ids))

    def set_actor_states(
        self, names: ActorNames, env_ids: EnvIds, states: ActorStates, write_updates: bool = True
    ) -> None:
        """See base class.

        ``write_updates=True`` (default) syncs immediately; pass ``False`` to batch several writes
        and flush once with :meth:`write_state_updates`.
        """
        if not names or (env_ids is not None and len(env_ids) == 0):
            return  # uniform no-op on empty input, matching get_actor_states / IsaacGym / MuJoCo
        self.set_actor_states_by_index(self.get_actor_indices(names, env_ids), states, write_updates)

    def get_actor_initial_poses(self, names: ActorNames, env_ids: EnvIds | None = None) -> ActorPoses:
        """See base class."""
        if not names:
            return torch.empty(0, 7, device=self.sim_device, dtype=torch.float32)

        # Determine which environments to use
        if env_ids is None:
            num_envs = getattr(self, "num_envs", self.scene.num_envs)
            env_ids = torch.arange(num_envs, device=self.sim_device)

        # Get base poses for each object (one per object)
        base_poses = []
        for obj_name in names:
            if obj_name == "robot":
                # Get robot base pose from configuration
                pos = torch.tensor(self.robot_config.init_state.pos, device=self.sim_device, dtype=torch.float32)
                rot = torch.tensor(self.robot_config.init_state.rot, device=self.sim_device, dtype=torch.float32)
                pose = torch.cat([pos, rot])  # [7] - [x,y,z,qx,qy,qz,qw]
                base_poses.append(pose)

            elif obj_name in self.scene.rigid_objects:
                # Get object pose from its rigid object (scene-file bodies included).
                rigid_object = self.scene.rigid_objects[obj_name]
                default_state = rigid_object.data.default_root_state[0]  # [13]
                pose = default_state[[0, 1, 2, 4, 5, 6, 3]]  # [x,y,z,qx,qy,qz,qw] reorder from wxyz to xyzw
                base_poses.append(pose)

            else:
                available_objects = ["robot"] + list(self.scene.rigid_objects.keys())
                raise KeyError(f"Object '{obj_name}' not found. Available: {available_objects}")

        base_poses_tensor = torch.stack(base_poses)  # [n, 7], env-independent config pose

        # Expand to per-env world poses, adding each env's origin to the position so the rows are
        # true world coordinates the registry stores. IsaacLab's default_root_state carries
        # the config pose without env origins; the spread comes from scene.env_origins (the
        # terrain/cloner grid every other placement path uses). Row order matches the registry:
        # [obj0_env0, obj0_env1, ..., obj1_env0, ...].
        env_origins = self.scene.env_origins[env_ids]  # [len(env_ids), 3]
        poses = base_poses_tensor.repeat_interleave(len(env_ids), dim=0)  # [n*len(env_ids), 7]
        poses[:, :3] += env_origins.repeat(len(names), 1)
        return poses

    def get_actor_initial_velocities(self, names: ActorNames, env_ids: EnvIds | None = None) -> torch.Tensor:
        """Get initial velocities for actors (the sibling of get_actor_initial_poses).

        Returns [len(names) * len(env_ids), 6] world-frame [vx,vy,vz,wx,wy,wz], same row order as
        get_actor_initial_poses (so the two concatenate into the 13-vector set_actor_states wants).
        """
        if not names:
            return torch.empty(0, 6, device=self.sim_device, dtype=torch.float32)

        if env_ids is None:
            num_envs = getattr(self, "num_envs", self.scene.num_envs)
            env_ids = torch.arange(num_envs, device=self.sim_device)

        return self.object_registry.get_initial_velocities_batch(names, env_ids)

    def _get_object_states(self, object_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states for any object type - delegates to state adapter.

        Parameters
        ----------
        object_name : str
            Name of the object to query
        env_ids : torch.Tensor
            Environment IDs to query, shape [num_envs], dtype torch.long

        Returns
        -------
        torch.Tensor
            Object states [len(env_ids), 13] containing position, quaternion, and velocities
            in xyzw format (converted by state adapter)
        """
        return self._state_adapter.get_object_states(object_name, env_ids)

    def _write_object_state_unified(self, object_name: str, states: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write object states for any object type - delegates to state adapter."""
        self._state_adapter.write_object_states(object_name, states, env_ids)

    def _physics_time(self) -> float:
        """The IsaacSim simulation clock in seconds (used by ``time()`` outside kinematic mode)."""
        return float(self.sim.current_time)

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        """Get DOF forces for a specific environment.

        This method provides access to measured joint forces. For IsaacSim,
        joint forces are computed from applied torques since direct force
        sensing is not available in the same way as IsaacGym.

        Args:
            env_id: Environment index (default: 0)

        Returns:
            torch.Tensor: Tensor of shape [num_dof] with computed joint forces

        Note:
            IsaacSim doesn't have the same DOF force sensor infrastructure as IsaacGym.
            This implementation returns the applied torques as an approximation.
            For actual force sensing, consider using contact sensors or force/torque sensors.
        """
        # IsaacSim doesn't have direct DOF force sensors like IsaacGym
        # Return the applied torques (which are the commanded forces)
        # This matches the bridge's usage pattern where forces are used for feedback
        if not hasattr(self._robot, "data") or not hasattr(self._robot.data, "applied_torque"):
            logger.warning(
                "DOF forces not directly available in IsaacSim. "
                "Returning zeros. For force feedback, the bridge will use commanded torques."
            )
            return torch.zeros(self.num_dof, device=self.sim_device)

        # Get applied torques which represent the forces being applied to joints
        applied_torques = self._robot.data.applied_torque[env_id, self.dof_ids]
        return applied_torques

    def write_state_updates(self) -> None:
        """See base class.

        IsaacSim-specific notes:
        - Uses IsaacLab's scene.write_data_to_sim() for efficient batch synchronization
        - Only performs sync if state adapter indicates dirty state (performance optimization)
        """
        if not self._state_adapter.is_dirty():
            logger.debug("No object state changes to sync")
            return

        logger.debug("Syncing object state changes to simulation")

        # Single call to sync all object state changes
        self.scene.write_data_to_sim()

        # Clear dirty flag via state adapter
        self._state_adapter.clear_dirty()

        logger.debug("All object state changes synced to simulation")
