"""MuJoCo scene manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger

from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig, SceneFileConfig
from holosoma.config_types.simulator import MujocoXMLFilterCfg, SimulatorConfig
from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.simulator.shared.asset_format import select_asset_format
from holosoma.utils.path import resolve_asset_path


@dataclass
class ActorSpecMeta:
    """Element inventory of a single actor, captured from its isolated ``MjSpec``.

    Recorded BEFORE the spec is attached/merged, when it holds only this actor's
    elements, so membership is unambiguous. Disambiguating in the merged model by
    prefix would be unsafe: an object prefix like ``"robot_decoy_"`` startswith-collides
    with the robot prefix ``"robot_"``, and object joints are indistinguishable from
    robot DOFs by name alone.

    Names are the actor's *clean* (unprefixed) names; the compiled-model name is
    ``prefix + name``. The root freejoint is excluded (it may be unnamed, e.g. t1's)
    and is resolved structurally by ``_resolve_freejoint_addrs``.
    """

    prefix: str
    root_body: str
    # Robot/actor bodies in spec (DFS) order, excluding the implicit worldbody.
    body_names: list[str] = field(default_factory=list)
    # Named, non-free joints only (the actor's actuated DOFs).
    dof_joint_names: list[str] = field(default_factory=list)
    # Named actuators.
    actuator_names: list[str] = field(default_factory=list)


class MujocoSceneManager:
    """Compositional world builder using MjSpec for MuJoCo simulations.

    This class provides a compositional approach to building MuJoCo simulation worlds
    by combining terrain, lighting, materials, and robots using the MjSpec API.
    It handles terrain generation, collision configuration, and robot integration
    while maintaining proper scene composition order.

    The scene manager supports multiple terrain types (plane, heightfield, trimesh)
    and provides automatic collision configuration based on robot self-collision settings.
    """

    def __init__(self, simulator_config: SimulatorConfig) -> None:
        """Initialize the scene manager with simulator configuration.

        Parameters
        ----------
        simulator_config : SimulatorConfig
            Simulator configuration containing physics and rendering parameters.
        """
        self.world_spec = mujoco.MjSpec()
        self.world_spec.copy_during_attach = True
        self._setup_world_options(simulator_config)
        self.robot_config: RobotConfig | None = None  # Set when adding robot
        # Actor name -> prefixed root-body name, for resolving each actor's freejoint
        # by exact body (avoids prefix-startswith collisions like "a_" vs "a_b_").
        self.robot_root_body: str | None = None  # Set when adding robot
        self.rigid_object_root_bodies: dict[str, str] = {}  # name -> prefixed root body name
        # Scene-file (1->N) bodies: registered name -> (prefixed root body, is_static).
        # A multi-body file is attached once (preserving inter-body relative poses); each
        # top-level body becomes its own actor, free (has freejoint) or static (jointless).
        self.scene_file_bodies: dict[str, tuple[str, bool]] = {}
        # Robot element inventory captured from its isolated spec pre-attach.
        # Single source of truth for robot element membership; see ActorSpecMeta.
        self.robot_spec_meta: ActorSpecMeta | None = None  # Set when adding robot

    def _setup_world_options(self, simulator_config: SimulatorConfig) -> None:
        """Configure world specification options from simulator config.

        Parameters
        ----------
        simulator_config : SimulatorConfig
            Simulator configuration containing physics parameters.
        """
        # TODO: expose to Mujoco-specific config
        self.world_spec.option.gravity = [0, 0, -9.81]
        self.world_spec.option.timestep = 1.0 / simulator_config.sim.fps  # type: ignore[attr-defined]

    def add_materials(self) -> None:
        """Add standard materials and textures to the world specification.

        Creates a chequered texture and grid material that can be applied
        to terrain and other geometric elements for visual enhancement.
        """

        self.world_spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            width=512,
            height=3072,
            rgb1=[0.3, 0.5, 0.7],  # Light blue
            rgb2=[0.0, 0.0, 0.0],  # Black
        )

        # Add chequered texture
        self.world_spec.add_texture(
            name="chequered",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
        )

        grid_material = self.world_spec.add_material(name="grid", texrepeat=[5, 5], reflectance=0.2)
        grid_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chequered"

        # Add a solid gray material with moderate specular response for meshes without textures
        self.world_spec.add_material(
            name="solid_gray",
            rgba=[0.3, 0.3, 0.3, 1.0],
            specular=0.2,
            reflectance=0.2,
            shininess=0.2,
            metallic=0.1,
            emission=1.0,
        )

    def add_lighting(self, lighting_config: Any | None = None) -> None:
        """Add lighting configuration to the world specification.

        Parameters
        ----------
        lighting_config : Any | None
            Lighting configuration parameters (currently unused, uses defaults).
        """
        # Arbitrary headlight ambient lighting
        self.world_spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        self.world_spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self.world_spec.visual.headlight.specular = [0.0, 0.0, 0.0]

        # Add global lighting orientation
        self.world_spec.visual.global_.azimuth = -130
        self.world_spec.visual.global_.elevation = -20

        # Match our existing scene files
        self.world_spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]

        # Uncomment to increase to reduce shadow pixelation for larger terrain.
        # Slows down rendering dramatically...
        # self.world_spec.visual.quality.shadowsize = 1024

        # Arbitrary lights (offset XY to avoid gantry shadows)
        self.world_spec.worldbody.add_light(
            pos=[2, 0, 5.0],
            dir=[0, 0, -1],
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
            # castshadow=True,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

        # Second light for extra shadows, commented out a little experience performance.
        # self.world_spec.worldbody.add_light(
        #    pos=[-2, 0, 4.0], dir=[0, 0, -1],
        #    diffuse=[0.6, 0.6, 0.6],
        #    specular=[0.2, 0.2, 0.2],
        #    castshadow=True,
        #    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        # )

    def add_terrain(self, terrain_state: TerrainTermBase, num_envs: int) -> None:
        """Add terrain to the world specification with extensible dispatch.

        Creates terrain using the TerrainTermBase class and converts it to the
        appropriate MuJoCo representation (plane, heightfield, or trimesh).
        Automatically configures collision properties for robot interaction.

        Parameters
        ----------
        cfg : TerrainConfig
            Terrain configuration specifying mesh type, dimensions, and properties.
        num_envs : int
            Number of environments (affects terrain layout planning).
        """

        geom: mujoco.MjSpec.Geom | None = None
        if terrain_state.mesh_type == "plane":
            geom = self._create_ground_plane(terrain_state)
        elif terrain_state.mesh_type in ["trimesh"]:
            # Use heightfield to reduce penetrations (vs. trimesh/geom mesh)
            geom = self._create_hfield(terrain_state)
        elif terrain_state.mesh_type in ["load_obj"]:
            geom = self._create_trimesh(terrain_state)
        elif terrain_state.mesh_type is None:
            logger.info("Terrain is none")
        else:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        if geom is not None:
            # Monkey-patch Mujoco geom into our terrain manager for convenience
            terrain_state.geom = geom  # type: ignore[attr-defined]

            # Set environment collision properties so robot self_collision flag works
            # Environment collision class
            terrain_state.geom.contype = 2  # type: ignore[attr-defined]
            # Only collide with robot (class 1)
            terrain_state.geom.conaffinity = 1  # type: ignore[attr-defined]

            # Hide the terrain visual while keeping its collider: zero-alpha rgba stops the geom from
            # drawing but leaves contype/conaffinity untouched. The geom's rgba overrides any
            # material/default color.
            if terrain_state.hide_visual:
                terrain_state.geom.rgba = [0.0, 0.0, 0.0, 0.0]  # type: ignore[attr-defined]

    def _create_ground_plane(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create a ground plane terrain geometry.

        Returns
        -------
        mujoco.MjSpec.Geom
            Ground plane geometry with configured physics properties.
        """
        # Create ground plane with hardcoded parameters and physics properties
        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            # Size=0 is rendered infinitely. Collision plane is always infinite.
            # Note: size.z is actually the rendered spacing betweeh the grid
            #       subdivisions (to improve lighting, shadows).
            size=[0, 0, 0.05],
            pos=[0, 0, 0],
            material="grid",
            friction=[
                # Ignore terrain config until we expose Mujoco-specific parameters
                0.7,  # reasonable default
                0.005,  # reasonable default
                0.001,  # reasonable default
            ],  # [sliding, torsional, rolling]
            solimp=[0.99, 0.99, 0.01, 0.5, 2],  # 5 elements: [dmin, dmax, width, midpoint, power]
            solref=[0.001, 1],  # 2 elements: [timeconst, dampratio]
        )

    def _create_trimesh(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create MuJoCo mesh terrain matching shared Terrain class behavior."""

        if terrain_state.mesh is None:
            raise ValueError("Terrain mesh data is required when using trimesh terrain type.")

        vertices = np.asarray(terrain_state.mesh.vertices, dtype=np.float32)
        faces = np.asarray(terrain_state.mesh.faces, dtype=np.int32)

        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Terrain mesh is empty and cannot be used to create a mesh geom.")

        mesh_spec = self.world_spec.add_mesh(name="terrain")
        mesh_spec.uservert = vertices.flatten(order="C")
        mesh_spec.userface = faces.flatten(order="C")
        mesh_spec.smoothnormal = False

        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_spec.name,
            pos=[0.0, 0.0, 0.0],
            material="solid_gray",
            friction=[
                # Ignore terrain config until we expose Mujoco-specific parameters
                0.7,  # reasonable default
                0.005,  # reasonable default
                0.001,  # reasonable default
            ],  # [sliding, torsional, rolling]
            solimp=[0.99, 0.99, 0.01, 0.5, 2],
            solref=[0.001, 1],
        )

    def _create_hfield(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create MuJoCo heightfield terrain from procedural terrain data.

        Converts the heightfield data from the terrain generator into a MuJoCo
        heightfield asset and geom. This avoids the convex hull simplification
        that occurs with trimesh terrain.

        Returns
        -------
        mujoco.MjSpec.Geom
            Heightfield geometry with configured physics properties.
        """
        terrain = terrain_state.terrain
        if not hasattr(terrain, "_height_field_raw"):
            raise ValueError("Terrain does not have heightfield data")

        # Get heightfield parameters from terrain
        height_data = np.asarray(terrain._height_field_raw, dtype=np.float32)
        vertical_scale = terrain._vertical_scale
        border_size = terrain._border_size
        total_length = terrain._total_length
        total_width = terrain._total_width

        # Apply vertical scaling to height data (convert from int16 indices to meters)
        height_data_scaled = height_data * vertical_scale

        # Handle negative heights: shift to make non-negative (MuJoCo requirement)
        min_height = height_data_scaled.min()
        z_offset = 0.0
        if min_height < 0:
            height_data_scaled = height_data_scaled - min_height + 1e-9
            z_offset = min_height
            logger.info(f"Shifted heightfield by {-min_height:.3f}m to ensure non-negative heights")

        max_height = height_data_scaled.max()
        min_height_final = height_data_scaled.min()

        # Calculate size parameters for MuJoCo hfield
        # size = [x_half, y_half, HEIGHT_RANGE, z_baseline]
        # Note: nrow/ncol are swapped for correct orientation
        height_range = max_height - min_height_final

        # Create heightfield asset
        hfield_spec = self.world_spec.add_hfield(name="terrain")
        hfield_spec.nrow = height_data.shape[1]  # swap: cols become rows
        hfield_spec.ncol = height_data.shape[0]  # swap: rows become cols
        hfield_spec.size = [0.5 * total_length, 0.5 * total_width, height_range, min_height_final]
        # MuJoCo expects raw elevation data in column-major (Fortran) order
        hfield_spec.userdata = height_data_scaled.flatten(order="F").tolist()

        logger.info(
            f"Created heightfield: {hfield_spec.nrow}x{hfield_spec.ncol},"
            f" size=[{0.5 * total_length:.2f}, {0.5 * total_width:.2f}, {height_range:.3f}, {min_height_final:.3f}]"
        )

        # Create heightfield geom, positioned to match terrain coordinate system
        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hfield_spec.name,
            pos=[
                0.5 * total_length - border_size,
                0.5 * total_width - border_size,
                z_offset if z_offset < 0 else 0.0,
            ],
            friction=[
                # Ignore terrain config until we expose Mujoco-specific parameters
                0.7,  # reasonable default
                0.005,  # reasonable default
                0.001,  # reasonable default
            ],  # [sliding, torsional, rolling]
            solimp=[0.99, 0.99, 0.01, 0.5, 2],
            solref=[0.001, 1],
        )

    def add_robot(
        self,
        terrain_state: TerrainTermBase,
        robot_config: RobotConfig,
        xml_filter: MujocoXMLFilterCfg | None = None,
        prefix: str = "robot_",
    ) -> None:
        """Add robot from XML file with namespace prefix and optional filtering.

        Loads a robot from its XML specification, applies optional filtering to
        remove scene elements (lights, ground), configures collision settings,
        and attaches it to the world with a namespace prefix.

        Parameters
        ----------
        robot_config : RobotConfig
            Robot configuration containing asset path and collision settings.
        xml_filter : MujocoXMLFilterCfg | None
            Optional XML filtering configuration to remove unwanted elements.
        prefix : str
            Namespace prefix for robot elements (default: "robot_").
        """
        robot_xml_path = resolve_asset_path(robot_config.asset.xml_file, robot_config.asset.asset_root)

        logger.info(f"Adding robot from: {robot_xml_path} with prefix: {prefix}")
        self.robot_model_path = robot_xml_path
        robot_spec = mujoco.MjSpec.from_file(robot_xml_path)

        if xml_filter and getattr(xml_filter, "enable", False):
            # Remove worldbody lights and ground|floor|plane geoms because they're added dynamically
            robot_spec = self._filter_worldbody(robot_spec, xml_filter)

        if hasattr(terrain_state, "geom") and terrain_state.geom:
            # Apply collision settings based on unified self_collisions flag in config
            # Only modifies collision groups if we have programmatically added terrain, otherwise
            # assumes the robot XML knows what it's doing
            self._apply_collision_settings(robot_spec, robot_config)

        # Require a single free body; capture its root-body name for addressing.
        root_body = self._ensure_single_root_body(robot_spec, robot_config.asset.robot_type, add_freejoint=True)

        # Apply the robot's shared link_physics (geom material/density) to every link body via the
        # same seam objects use, before attach/compile so it lands in root_model before put_model.
        # No-op unless link_physics/overrides are configured.
        self._apply_link_physics_to_robot(robot_spec, robot_config)

        # Capture the robot's element inventory before attach (see ActorSpecMeta).
        self.robot_spec_meta = self._capture_spec_meta(robot_spec, prefix, root_body)

        # Attach at a unit frame; the real initial pose is set later via qpos.
        unit_frame = self.world_spec.worldbody.add_frame()
        self.world_spec.attach(robot_spec, frame=unit_frame, prefix=prefix)

        # Store the prefixed root-body name for the simulator's freejoint resolution. (Robot
        # element membership / name maps come from robot_spec_meta, captured above, not a prefix.)
        self.robot_root_body = f"{prefix}{root_body}"

    def add_rigid_object(
        self,
        name: str,
        obj_config: RigidObjectConfig,
        supported_formats: list[str],
        xml_filter: MujocoXMLFilterCfg | None = None,
        scene_asset_root: str | None = None,
    ) -> None:
        """Add a rigid object to the world (before compile).

        A free object (``fixed=False``) gets a free joint so it can move and is
        addressed via its qpos slice. A static object (``fixed=True``) is attached to
        the world with no joint, so it holds its pose; it has no qpos slice and its
        pose is read from xpos. The asset format is chosen by ``supported_formats``
        preference order via ``select_asset_format``.

        Parameters
        ----------
        name : str
            Actor name (the ObjectRegistry key); used as the attach prefix.
        obj_config : RigidObjectConfig
            Object configuration: asset file(s), pose, and ``fixed`` flag.
        supported_formats : list[str]
            Asset formats this backend loads, in preference order (the simulator's
            ``get_supported_scene_formats()``).
        xml_filter : MujocoXMLFilterCfg | None
            Optional worldbody filtering (lights/ground).
        scene_asset_root : str | None
            Scene-level asset root fallback for relative paths.
        """
        _fmt, asset_rel = select_asset_format(obj_config, supported_formats)
        # Resolve under the per-object asset_root, falling back to the scene's.
        asset_path = resolve_asset_path(asset_rel, obj_config.asset_root or scene_asset_root)
        prefix = f"{name}_"

        logger.info(f"Adding rigid object '{name}' from: {asset_path} with prefix: {prefix}")
        obj_spec = mujoco.MjSpec.from_file(asset_path)

        if xml_filter and getattr(xml_filter, "enable", False):
            # Asset files may carry their own lights/ground; strip them like the robot.
            obj_spec = self._filter_worldbody(obj_spec, xml_filter)

        # Single top-level body; add a free joint only for free (non-fixed) objects.
        root_body = self._ensure_single_root_body(obj_spec, name, add_freejoint=not obj_config.fixed)

        # Apply the per-object physics override (shared PhysicsConfig) to the body before attach,
        # while the spec still holds only this object's elements.
        self._apply_physics_to_body(obj_spec.worldbody.bodies[0], obj_config.physics, name)

        # Free objects attach at a unit (identity) frame; their world pose is set later
        # via the freejoint qpos. A static object has no qpos, so its pose must be baked
        # into the attach frame here (config orientation is wxyz, which MjSpec frames use).
        frame = self.world_spec.worldbody.add_frame()
        if obj_config.fixed:
            frame.pos = list(obj_config.position)
            frame.quat = list(obj_config.orientation)  # [w, x, y, z]
        self.world_spec.attach(obj_spec, frame=frame, prefix=prefix)

        # Store the prefixed root-body name for addressing (freejoint or static pose).
        self.rigid_object_root_bodies[name] = f"{prefix}{root_body}"

    def add_scene_file(
        self,
        scene_file_name: str,
        scene_file: SceneFileConfig,
        supported_formats: list[str],
        xml_filter: MujocoXMLFilterCfg | None = None,
        scene_asset_root: str | None = None,
    ) -> None:
        """Add a multi-body scene file to the world (before compile), 1->N.

        The whole file is attached once at the file's configured world pose, so MjSpec
        composes that pose with each body's authored relative pose (preserving inter-body
        offsets) into the compiled model. Each top-level worldbody body becomes its own
        actor named ``{scene_file_name}_{body}``: a body with a free joint registers free
        (its composed world pose lives in its freejoint qpos0), a jointless body registers
        static (pose read from xpos). The per-body classification comes from the file's own
        structure, never from config.

        Parameters
        ----------
        scene_file_name : str
            File-scope namespace (the SceneConfig.scene_files key); registered bodies are
            ``{scene_file_name}_{body}``.
        scene_file : SceneFileConfig
            Scene-file config: asset path(s) and world pose.
        supported_formats : list[str]
            Asset formats this backend loads, in preference order (the simulator's
            ``get_supported_scene_formats()``).
        xml_filter : MujocoXMLFilterCfg | None
            Optional worldbody filtering (lights/ground).
        scene_asset_root : str | None
            Scene-level asset root fallback for relative paths.
        """
        fmt, asset_rel = select_asset_format(scene_file, supported_formats)
        asset_path = resolve_asset_path(asset_rel, scene_file.asset_root or scene_asset_root)
        prefix = f"{scene_file_name}_"

        logger.info(f"Adding scene file '{scene_file_name}' from: {asset_path} ({fmt}) with prefix: {prefix}")
        spec = mujoco.MjSpec.from_file(asset_path)
        if xml_filter and getattr(xml_filter, "enable", False):
            spec = self._filter_worldbody(spec, xml_filter)

        bodies = list(spec.worldbody.bodies)
        if not bodies:
            raise ValueError(f"Scene file '{scene_file_name}' ({asset_path}) has no top-level bodies.")

        # Apply the include/exclude subset filter (shared SceneFileConfig.should_include, identical
        # on every backend). A dropped body is deleted from the spec before attach so it never
        # reaches the compiled model (the whole file is attached at once, so there's no per-body
        # skip after attach). Bodies that pass continue to classification below.
        included = [b for b in bodies if scene_file.should_include(b.name)]
        for body in bodies:
            if body not in included:
                spec.delete(body)
        if not included:
            raise ValueError(
                f"Scene file '{scene_file_name}' ({asset_path}): include/exclude patterns selected "
                f"no bodies (include={scene_file.include_patterns}, "
                f"exclude={scene_file.exclude_patterns}; candidates={[b.name for b in bodies]})."
            )

        # Classify each top-level body: free joint => free, jointless => static (the file's
        # structure), with any per-object config override applied. The body is then made to
        # match (a freejoint dropped from a forced-static body, or added to a forced-free
        # one) so its joint state and registered type always agree.
        body_is_static: dict[str, bool] = {}
        for body in included:
            if body.name in body_is_static:
                raise ValueError(
                    f"Scene file '{scene_file_name}' has duplicate top-level body '{body.name}'; "
                    "body names must be unique within a file."
                )
            freejoint = next((j for j in body.joints if j.type == mujoco.mjtJoint.mjJNT_FREE), None)
            is_static = scene_file.resolve_fixed(body.name, structural_default=freejoint is None)
            if is_static and freejoint is not None:
                spec.delete(freejoint)  # override: forced static -> drop its freejoint
            elif not is_static and freejoint is None:
                body.add_freejoint()  # override: forced free -> give it a freejoint
            # Per-body physics override (shared resolve_physics rule, same as IsaacGym/IsaacSim),
            # applied to this body before the file is attached.
            self._apply_physics_to_body(body, scene_file.resolve_physics(body.name), f"{scene_file_name}_{body.name}")
            # Per-body world-frame position offset. body.pos is relative to the attach frame, so
            # rotate the world offset into the frame's local space (conjugate of the file
            # orientation) before adding. Identity orientation => a plain add.
            offset = scene_file.resolve_position_offset(body.name)
            if offset is not None:
                local_offset = np.zeros(3)
                conj_quat = np.array(scene_file.orientation, dtype=float)  # wxyz
                mujoco.mju_negQuat(conj_quat, conj_quat)
                mujoco.mju_rotVecQuat(local_offset, np.array(offset, dtype=float), conj_quat)
                body.pos = [body.pos[0] + local_offset[0], body.pos[1] + local_offset[1], body.pos[2] + local_offset[2]]
            body_is_static[body.name] = is_static

        # Attach the whole file once at its world pose; MjSpec composes the frame pose
        # with each body's relative pose (free bodies' composed pose lands in qpos0).
        frame = self.world_spec.worldbody.add_frame()
        frame.pos = list(scene_file.position)
        frame.quat = list(scene_file.orientation)  # [w, x, y, z]
        self.world_spec.attach(spec, frame=frame, prefix=prefix)

        for body_name, is_static in body_is_static.items():
            actor_name = f"{scene_file_name}_{body_name}"
            self.scene_file_bodies[actor_name] = (f"{prefix}{body_name}", is_static)

    def _apply_physics_to_body(
        self, body: mujoco.MjSpec.Body, physics: PhysicsConfig | None, name: str, *, apply_mass: bool = True
    ) -> None:
        """Apply a ``PhysicsConfig`` onto an MjSpec body before compile. Shared by scene objects
        and robot links.

        Applies the cross-backend core (``density`` kg/m³, and ``mass`` kg when ``apply_mass``) plus
        MuJoCo's own geom-level sub-config (``physics.mujoco``: friction/solref/solimp/condim). The
        other engines' sub-configs (``physx``/``isaacgym``/``isaacsim``) do not apply to MuJoCo and
        are ignored.

        ``mass`` sets the body's explicit mass (``explicitinertial`` so MuJoCo keeps it rather than
        recomputing from geom density); ``density`` and the geom props are applied to every geom of
        the body. ``None`` on any field keeps the asset's authored value.

        ``apply_mass=False`` skips the ``mass`` write. It is passed when applying the whole-robot
        ``link_physics`` default to every link, where one mass on every link is meaningless (the
        config validator forbids ``mass`` on that default; this is the matching guard at the apply
        site). A per-link override that sets ``mass`` is applied with the default ``apply_mass=True``
        on its single target body.
        """
        if physics is None:
            return

        if apply_mass and physics.mass is not None:
            body.mass = physics.mass
            body.explicitinertial = True
            logger.debug(f"Set explicit mass={physics.mass} on '{name}'")

        mj = physics.mujoco
        for geom in body.geoms:
            if physics.density is not None:
                geom.density = physics.density
            if mj is not None:
                if mj.friction is not None:
                    geom.friction = list(mj.friction)
                if mj.solref is not None:
                    geom.solref = list(mj.solref)
                if mj.solimp is not None:
                    geom.solimp = list(mj.solimp)
                if mj.condim is not None:
                    geom.condim = mj.condim

    def _apply_link_physics_to_robot(self, robot_spec: mujoco.MjSpec, robot_config: RobotConfig) -> None:
        """Apply the robot's ``link_physics`` to every robot link body, before attach/compile.

        Walks the robot spec's bodies (the same all-links traversal ``_configure_robot_collisions``
        uses) and routes each through the shared :meth:`_apply_physics_to_body`, giving robot links
        the same MuJoCo geom-material path scene objects get.

        ``link_physics`` is the whole-robot ``body_names='.*'`` default, so it is applied with
        ``apply_mass=False``: one mass on every link is meaningless (the config validator forbids
        ``mass`` here; this is the matching guard at the apply site). No-op when ``link_physics`` is
        unset.

        Runs pre-compile, so this is baked into ``root_model`` before ``WarpBackend.put_model`` and
        the captured CUDA step graph, avoiding the graph-orphan concern that affects runtime DR.
        """
        asset = robot_config.asset
        if asset.link_physics is None:
            return
        worldbody_name = robot_spec.worldbody.name
        for body in robot_spec.bodies:
            if not body.name or body.name == worldbody_name:
                continue
            self._apply_physics_to_body(body, asset.link_physics, f"{asset.robot_type}/{body.name}", apply_mass=False)

    def _ensure_single_root_body(self, spec: mujoco.MjSpec, name: str, add_freejoint: bool = True) -> str:
        """Require a single top-level body; return its root-body name.

        Every actor (robot or rigid object) is modelled as a single top-level body.
        Raises if the asset has multiple. When ``add_freejoint`` is True (free bodies),
        reuses an existing freejoint or adds one; when False (static SCENE bodies), the
        body is left jointless so it stays welded to the world.

        Parameters
        ----------
        spec : mujoco.MjSpec
            The asset spec (mutated: a freejoint is added if requested and absent).
        name : str
            Actor name, for error messages.
        add_freejoint : bool
            Add a free joint (free body) vs. leave jointless (static body).

        Returns
        -------
        str
            The root body's name (unprefixed, as defined in the asset).
        """
        root_bodies = list(spec.worldbody.bodies)
        if len(root_bodies) != 1:
            raise ValueError(f"Actor '{name}' asset has {len(root_bodies)} top-level bodies; exactly one is supported.")
        root_body = root_bodies[0]
        if add_freejoint and not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root_body.joints):
            root_body.add_freejoint()
        return root_body.name

    def _capture_spec_meta(self, spec: mujoco.MjSpec, prefix: str, root_body: str) -> ActorSpecMeta:
        """Record an actor's named element inventory from its isolated (pre-attach) spec.

        Excludes the implicit worldbody and the root freejoint; see ``ActorSpecMeta``
        for why capture happens before attach. ``spec`` is the actor's own MjSpec,
        ``prefix`` its attach-time namespace, ``root_body`` its unprefixed root body.
        """
        worldbody_name = spec.worldbody.name
        body_names = [b.name for b in spec.bodies if b.name and b.name != worldbody_name]
        dof_joint_names = [j.name for j in spec.joints if j.name and j.type != mujoco.mjtJoint.mjJNT_FREE]
        actuator_names = [a.name for a in spec.actuators if a.name]
        return ActorSpecMeta(
            prefix=prefix,
            root_body=root_body,
            body_names=body_names,
            dof_joint_names=dof_joint_names,
            actuator_names=actuator_names,
        )

    def _apply_collision_settings(self, robot_spec: mujoco.MjSpec, robot_config: RobotConfig) -> None:
        """Apply collision settings based on unified self_collisions configuration.

        This matches IsaacGym/IsaacSim behavior programmatically by configuring
        MuJoCo collision classes based on the robot's self_collisions setting.

        Parameters
        ----------
        robot_spec : mujoco.MjSpec
            Robot specification to modify collision settings for.
        robot_config : RobotConfig
            Robot configuration containing self_collisions setting.
        """
        self._configure_robot_collisions(robot_spec, robot_config.asset.enable_self_collisions)

    def _configure_robot_collisions(self, robot_spec: mujoco.MjSpec, enable_self_collisions: bool) -> None:
        """Configure robot collision behavior using MuJoCo collision classes.

        Parameters
        ----------
        robot_spec : mujoco.MjSpec
            Robot specification to configure collisions for.
        enable_self_collisions : bool
            If True, robot parts collide with each other + environment.
            If False, robot parts only collide with environment.

        Notes
        -----
        Collision class system:
        - Robot parts: contype=1
        - Environment: contype=2, conaffinity=1
        - Robot conaffinity: 3 (both) if self_collisions, 2 (env only) if not
        """
        if enable_self_collisions:
            robot_conaffinity = 3  # Collide with robot (1) + environment (2) = 3
            collision_mode = "self + environment"
        else:
            robot_conaffinity = 2  # Only collide with environment (2)
            collision_mode = "environment only"

        bodies_processed = 0
        geoms_processed = 0

        # Apply collision settings to all robot bodies
        for body in robot_spec.bodies:
            if not body.name:
                # Skip unnamed bodies
                continue

            bodies_processed += 1
            for geom in body.geoms:
                # Skip geoms that have been explicitly configured away from defaults
                # Visual meshes typically have contype=0, conaffinity=0
                if geom.contype == 0 or geom.conaffinity == 0:
                    continue  # Skip visual/disabled collision geoms

                # Apply collision settings to geoms using default collision behavior
                # (contype=1, conaffinity=1 are MuJoCo defaults)
                if geom.contype == 1 and geom.conaffinity == 1:
                    geom.conaffinity = robot_conaffinity  # Configurable based on self_collisions
                    geoms_processed += 1
                    logger.debug(f"Set {body.name} geom: contype=1, conaffinity={robot_conaffinity} ({collision_mode})")

        logger.info(f"Applied collision settings to {geoms_processed} geoms across {bodies_processed} bodies")

    def _filter_worldbody(self, spec: mujoco.MjSpec, cfg: MujocoXMLFilterCfg) -> mujoco.MjSpec:
        """Remove lights and ground elements from an asset spec's worldbody.

        Helper work-around while asset XMLs (robot or object) contain scene
        elements that should be managed by the scene manager instead.

        Parameters
        ----------
        spec : mujoco.MjSpec
            Asset specification to filter (robot or rigid object).
        cfg : MujocoXMLFilterCfg
            Filtering configuration specifying what to remove.

        Returns
        -------
        mujoco.MjSpec
            Filtered specification.
        """
        # Remove lights if configured
        if cfg.remove_lights:
            for light in spec.worldbody.lights:
                spec.delete(light)

        # Remove ground geoms if configured
        if cfg.remove_ground:
            for geom in spec.worldbody.geoms:
                if self._is_ground_geom(geom, cfg.ground_names):
                    spec.delete(geom)

        return spec

    def _is_ground_geom(self, geom: mujoco.MjSpec.Geom, ground_names: List[str]) -> bool:
        """Determine if a geometry represents ground/floor.

        Parameters
        ----------
        geom : mujoco.MjSpec.Geom
            Geometry to check.
        ground_names : List[str]
            List of names that indicate ground geometries.

        Returns
        -------
        bool
            True if the geometry represents ground/floor.
        """
        # Check by name
        if geom.name and any(name in geom.name.lower() for name in ground_names):
            return True

        return geom.type == mujoco.mjtGeom.mjGEOM_PLANE

    def compile(self) -> mujoco.MjModel:
        """Compile the final world model from the specification.

        Returns
        -------
        mujoco.MjModel
            Compiled MuJoCo model ready for simulation.
        """
        logger.info("Compiling world model using MjSpec")
        return self.world_spec.compile()
