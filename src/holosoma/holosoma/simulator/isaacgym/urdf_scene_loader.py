"""
URDF Scene Loader for IsaacGym

This module handles URDF scene loading logic using the unified configuration system.

## Overview

The URDFSceneLoader processes two types of URDF objects:
1. **Scene Files**: Complex URDF files containing multiple objects (e.g., shelves with items)
2. **Rigid Objects**: Individual URDF objects with specific poses and physics properties

## Key Concepts

- **URDF Specs**: Standardized internal representation of objects to be loaded
- **Asset Configuration**: IsaacGym-specific loading parameters (physics, rendering options)
- **Physics Configuration**: Post-loading physics properties (damping, mass, etc.)
- **Shared Resolvers**: resolve_asset_path (asset paths), resolve_physics (physics config),
  and resolve_fixed (static vs free link classification) shared across backends

## Processing Flow

1. **Validation**: Check scene config structure and log key information
2. **Scene Processing**: Parse scene URDF files, compute each body's world pose by composing
   the file world transform with the body's authored relative pose, classify links
   static/free, and extract individual objects
3. **Rigid Processing**: Process individual rigid objects with their poses and physics
4. **Asset Loading**: Load IsaacGym assets from URDF specs with proper configurations
5. **Storage**: Store loaded assets and initial states for environment creation

## Architecture

The loader uses a pipeline approach:
- Scene Config → URDF Specs → Loaded Assets → Environment Creation
- Each stage has focused responsibilities and clear data contracts
- Physics, asset paths, and link fixity are resolved via the shared resolvers
  (resolve_physics / resolve_asset_path / resolve_fixed) rather than in-place
  scene-graph mutation or pattern matching
- Physics configs are stored separately for post-creation application
"""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import trimesh.transformations as tra
import yourdfpy  # type:ignore[import-untyped]
from isaacgym import gymapi
from loguru import logger
from yourdfpy.urdf import Robot  # type:ignore[import-untyped]

from holosoma.config_types.scene import (
    PhysicsConfig,
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)
from holosoma.simulator.isaacgym.physics import apply_physx_asset_options
from holosoma.simulator.shared.asset_format import select_asset_format
from holosoma.utils.path import resolve_asset_path

# Type aliases compatible with Python 3.8
Pose = List[float]  # [x, y, z, qx, qy, qz, qw]
AssetDict = Dict[str, gymapi.Asset]
PoseDict = Dict[str, Pose]


# Cannot freeze because we convert dynamically on load
@dataclass(frozen=False)
class AssetConfig:
    """Default configuration for IsaacGym asset loading.

    This dataclass maps directly to gymapi.AssetOptions parameters.
    Each field corresponds to an option that can be set on AssetOptions
    when loading assets through IsaacGym's load_asset() method.
    """

    default_dof_drive_mode: Any = None
    collapse_fixed_joints: bool = True
    replace_cylinder_with_capsule: bool = False
    flip_visual_attachments: bool = False
    fix_base_link: bool = False
    density: float = 1000.0
    angular_damping: float = 0.1
    linear_damping: float = 0.1
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.001
    disable_gravity: bool = False


@dataclass(frozen=True)
class URDFSpec:
    """Unified URDF specification for scene loading.

    This dataclass represents a standardized specification for URDF objects
    that need to be loaded into the simulation environment. It supports both
    pre-loaded assets (for scene objects) and URDF file paths (for rigid objects).

    Parameters
    ----------
    name : str
        Unique identifier for the URDF object.
    pose : Pose
        Object pose as [x, y, z, qx, qy, qz, qw].
    asset : gymapi.Asset | None
        Pre-loaded IsaacGym asset for scene objects, by default None.
    urdf_path : str | None
        Path to URDF file for rigid objects, by default None.
    asset_config : AssetConfig | None
        IsaacGym asset configuration parameters, by default None.
    physics : PhysicsConfig | None
        Per-object physics override, applied post-creation by
        ``IsaacGym.apply_physics_properties_to_actor``. ``None`` keeps the asset's authored physics.
    """

    name: str
    pose: Pose
    asset: gymapi.Asset | None = None  # Pre-loaded asset (for scene objects)
    urdf_path: str | None = None  # Path to URDF file (for rigid objects)
    asset_config: AssetConfig | None = None
    physics: PhysicsConfig | None = None

    @property
    def is_preloaded(self) -> bool:
        """Check if asset is pre-loaded."""
        return self.asset is not None

    @property
    def needs_loading(self) -> bool:
        """Check if asset needs to be loaded from file."""
        return self.asset is None and self.urdf_path is not None


class URDFSceneLoader:
    """URDF scene loader working directly with typed dataclasses.

    This class handles loading URDF scenes using the unified configuration system.
    It processes both scene files (complex URDF files with multiple objects) and
    rigid objects (individual URDF objects with specific poses and physics).

    The loader uses a pipeline approach where scene configurations are converted
    to standardized URDF specs, then loaded as IsaacGym assets with proper
    physics configurations stored separately for post-creation application.

    Parameters
    ----------
    gym_instance : gymapi.Gym
        IsaacGym instance for asset loading operations.
    sim : gymapi.Sim
        IsaacGym simulation handle.
    device : str
        Device identifier for tensor operations.

    Attributes
    ----------
    gym : gymapi.Gym
        IsaacGym instance reference.
    sim : gymapi.Sim
        IsaacGym simulation handle.
    device : str
        Device identifier.
    object_physics_configs : dict
        Physics configurations for post-creation application.
    loaded_assets : dict
        Loaded IsaacGym assets indexed by object name.
    loaded_initial_states : dict
        Initial poses for loaded objects.
    """

    def __init__(self, gym_instance: gymapi.Gym, sim: gymapi.Sim, device: str) -> None:
        self.gym: gymapi.Gym = gym_instance
        self.sim: gymapi.Sim = sim
        self.device: str = device
        # Per-object physics overrides (PhysicsConfig), keyed by actor name, for post-creation
        # application in IsaacGym.apply_physics_properties_to_actor.
        self.object_physics_configs: dict[str, PhysicsConfig] = {}
        # Names (prefixed) of scene-file bodies classified STATIC by the file's joint
        # structure (fixed-to-world). Read by IsaacGym._collect_spawned_actors to
        # register the right ObjectType.
        self.scene_file_static_names: set[str] = set()
        # Store loaded data for access during environment creation
        self.loaded_assets: AssetDict = {}
        self.loaded_initial_states: PoseDict = {}

    # === PUBLIC INTERFACE ===

    def load_scene_files(self, scene_config: SceneConfig) -> tuple[AssetDict, PoseDict]:
        """Load URDF scenes using unified configuration structure.

        Main entry point for loading URDF scenes. Processes both scene files
        (complex URDF files with multiple objects) and rigid objects (individual
        URDF objects with specific poses and physics properties).

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration dataclass containing scene_files and rigid_objects.

        Returns
        -------
        tuple[AssetDict, PoseDict]
            Tuple containing (object_assets, object_initial_state) where:
            - object_assets: dict mapping object names to IsaacGym assets
            - object_initial_state: dict mapping object names to poses [x,y,z,qx,qy,qz,qw]

        Raises
        ------
        ValueError
            If URDF scene loading fails due to configuration or file issues.
        """
        try:
            logger.debug(f"scene_config type: {type(scene_config)}")
            logger.debug(f"scene_config.scene_files: {getattr(scene_config, 'scene_files', None)}")
            logger.debug(f"scene_config.rigid_objects: {getattr(scene_config, 'rigid_objects', None)}")

            # Process different source types
            scene_specs = self._process_scene_files(scene_config)
            rigid_specs = self._process_rigid_objects(scene_config)

            # Combine and load assets
            all_specs = scene_specs + rigid_specs
            return self._load_assets_from_specs(all_specs)

        except Exception as e:
            logger.error(f"URDF scene loading failed: {e}")
            raise ValueError("URDF scene loading failed") from e

    # === SCENE PROCESSING ===

    def _process_scene_files(self, scene_config: SceneConfig) -> list[URDFSpec]:
        """Expand every scene file (1->N) into URDFSpecs, accumulated across all files.

        Each file's format is chosen by the shared ``select_asset_format`` (IsaacGym loads URDF
        only), so a usd/xml-only scene file fails loud with the standard matrix error. Each file
        resolves its own mesh paths locally, so multiple files do not interfere.
        """
        if not (hasattr(scene_config, "scene_files") and scene_config.scene_files):
            return []

        specs: list[URDFSpec] = []
        for scene_file_name, scene_file in scene_config.scene_files.items():
            # IsaacGym loads URDF only; fail loud on usd/xml-only scene files.
            select_asset_format(scene_file, ["urdf"])
            specs.extend(self._expand_scene_file(scene_file_name, scene_file, scene_config))
        return specs

    def _process_rigid_objects(self, scene_config) -> list[URDFSpec]:
        """Process rigid objects and return URDFSpec objects"""
        if not (hasattr(scene_config, "rigid_objects") and scene_config.rigid_objects):
            return []

        return self._build_rigid_object_specs(scene_config.rigid_objects, scene_config.asset_root)

    def _load_assets_from_specs(self, specs: list[URDFSpec]) -> tuple[AssetDict, PoseDict]:
        """Load assets from standardized URDFSpec objects"""
        if not specs:
            logger.warning("No URDF objects to load")
            return {}, {}

        object_assets = {}
        object_initial_state = {}

        for spec in specs:
            if spec.is_preloaded:
                # Use pre-loaded asset
                object_assets[spec.name] = spec.asset
                object_initial_state[spec.name] = spec.pose
                logger.debug(f"Added pre-loaded '{spec.name}' asset")
            elif spec.needs_loading:
                # Load asset from file
                asset = self._load_and_store_asset(spec.name, spec.urdf_path, spec.asset_config)
                object_assets[spec.name] = asset
                object_initial_state[spec.name] = spec.pose
                logger.debug(f"Loaded '{spec.name}' from {spec.urdf_path}")
            else:
                logger.error(f"Invalid URDFSpec: {spec}")
                raise ValueError(f"URDFSpec missing both asset and urdf_path: {spec.name}")

            # Record the per-object physics override for post-creation application
            # (None keeps the asset's authored physics).
            if spec.physics is not None:
                self.object_physics_configs[spec.name] = spec.physics

        # Store the loaded data for access during environment creation
        self.loaded_assets = object_assets
        self.loaded_initial_states = object_initial_state

        logger.info(f"Successfully loaded {len(object_assets)} URDF objects")
        return object_assets, object_initial_state

    def _load_and_store_asset(self, name: str, urdf_path: str | None, asset_config: AssetConfig | None):
        """Load the IsaacGym asset from a URDF file.

        Physics overrides are carried on the ``URDFSpec`` and stored by
        ``_load_assets_from_specs`` — this method only loads the asset.
        """
        try:
            assert urdf_path is not None
            assert asset_config is not None
            asset = self._load_asset_from_urdf(urdf_path, asset_config)
            if asset is None:
                error_msg = f"Failed to load '{name}': IsaacGym returned None asset from {urdf_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            return asset
        except Exception as e:
            error_msg = f"Failed to load '{name}' from {urdf_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _expand_scene_file(self, scene_file_name, scene_file, scene_config):
        """
        Parse a URDF as a "scene" file and return standardized URDF specs.

        This is a helper because Isaacgym doesn't directly support loading URDF files
        into multiple actors. Instead, we parse top-level links and create actors
        for each.

        Scene URDF Processing Flow:
        1. Parse URDF with yourdfpy to build scene graph
        2. Compute the file world transform (config position + orientation)
        3. Classify each link static (welded) vs free from its joint structure
        4. For each link, compose its world pose (file world transform @ relative pose)
        5. Create a temporary URDF file per link and load the IsaacGym asset immediately
        6. Return specs with pre-loaded assets
        """
        # Resolve via the shared resolver (handles "holosoma/..." package paths and
        # asset_root joins; scene_config.asset_root may be None).
        urdf_path = resolve_asset_path(scene_file.urdf_path, scene_file.asset_root or scene_config.asset_root)
        logger.debug(f"Loading scene URDF file: {urdf_path}")

        # The per-link tempfiles resolve their meshes relative to THIS file's directory, threaded
        # through _create_individual_link_urdf_tempfile (which derives it from urdf_path) — no
        # shared loader state, so looping multiple scene files is safe.

        # Step 1: Parse URDF with scene graph for transform extraction
        urdf = self._parse_scene_urdf_with_yourdfpy(urdf_path, scene_file)

        # Step 2: file world transform (config position + wxyz orientation). Each body's
        # world pose is this composed with the body's authored relative pose (computed in
        # _extract_world_transform_from_scene) — yourdfpy does NOT propagate a transform
        # applied to the 'world' node down to its joint children, so we compose explicitly.
        file_world_transform = tra.translation_matrix(scene_file.position) @ tra.quaternion_matrix(
            scene_file.orientation
        )

        # Classify each link free vs static from the file's own joint structure: a link
        # whose joint to its parent is "fixed" is welded => STATIC; "floating" => FREE.
        static_by_link = self._classify_scene_links(urdf)

        # Step 3: Process each link as an individual object, sorted by name for
        # backend-independent registration order; names are file-prefixed for collision
        # safety ({scene_file_name}_{link}).
        prefix = f"{scene_file_name}_"
        urdf_specs = []
        for link in sorted(urdf.robot.links, key=lambda link_obj: link_obj.name):
            if link.name == "world":
                continue

            # Apply pattern-based filtering
            if not self._should_load_object(link.name, scene_file):
                logger.debug(f"Skipping '{link.name}' due to scene file patterns")
                continue

            is_static = scene_file.resolve_fixed(link.name, structural_default=static_by_link.get(link.name, True))
            actor_name = f"{prefix}{link.name}"

            # Get IsaacGym asset configuration (fixed base iff the file welds this link)
            asset_config = self._get_asset_config_for_scene_object(link.name, scene_file, is_static=is_static)
            if is_static:
                self.scene_file_static_names.add(actor_name)

            # World pose = file world transform composed with the body's relative pose.
            if urdf.scene is None:
                logger.warning(f"No scene graph for '{link.name}', using default pose")
                pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                pose = self._extract_world_transform_from_scene(urdf, link.name, file_world_transform)

            # Per-body world-frame position offset added on top of the composed world position.
            # None => unchanged.
            offset = scene_file.resolve_position_offset(link.name)
            if offset is not None:
                pose = [pose[0] + offset[0], pose[1] + offset[1], pose[2] + offset[2], *pose[3:]]

            # Step 4 & 5: Create temp URDF and load asset immediately
            with self._create_individual_link_urdf_tempfile(link, link.name, urdf_path) as temp_urdf_file:
                asset = self._load_asset_from_urdf(temp_urdf_file.name, asset_config)
                if asset is None:
                    error_msg = (
                        f"Failed to load '{actor_name}': IsaacGym returned None asset from {temp_urdf_file.name}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Step 6: Create URDFSpec with pre-loaded asset. The live per-body physics
                # override (shared resolve_physics rule) rides on the spec to the apply site.
                urdf_spec = URDFSpec(
                    name=actor_name,
                    pose=pose,
                    asset=asset,  # Pre-loaded
                    urdf_path=None,  # Not needed for scene objects
                    asset_config=asset_config,
                    physics=scene_file.resolve_physics(link.name),
                )
                urdf_specs.append(urdf_spec)
                logger.debug(f"Generated URDFSpec for '{actor_name}' (static={is_static}) at pose {pose[:3]}")

        logger.debug(f"Generated {len(urdf_specs)} URDFSpec objects from scene URDF file")
        return urdf_specs

    def _build_rigid_object_specs(self, rigid_objects, asset_root):
        """Generate URDF specs from the rigid_objects dict (name -> config)."""
        logger.debug("Loading rigid objects")

        if not rigid_objects:
            logger.warning("No rigid_objects configuration found")
            return []

        urdf_specs = []

        # Any failure here (missing file, unsupported format, bad config) is fatal by design,
        # matching select_asset_format and the scene-file path: never silently drop an object.
        for object_name, obj in rigid_objects.items():
            # IsaacGym loads URDF only; fail loud on usd/xml-only objects.
            _fmt, urdf_rel = select_asset_format(obj, ["urdf"])

            # Resolve under the per-object asset_root, falling back to the scene's
            # (handles package "holosoma/..." paths, not just asset_root joins).
            urdf_path = resolve_asset_path(urdf_rel, obj.asset_root or asset_root)

            if not Path(urdf_path).exists():
                raise ValueError(f"URDF file not found for object '{object_name}': {urdf_path}")

            # Convert orientation from [w, x, y, z] to [x, y, z, w] for IsaacGym
            pose = obj.position + [obj.orientation[1], obj.orientation[2], obj.orientation[3], obj.orientation[0]]

            # Get asset configuration from physics config
            asset_config = self._get_asset_config_for_rigid_object(object_name, obj)

            # Create URDFSpec for rigid object; its physics override rides on the spec to the
            # post-creation apply site.
            urdf_specs.append(
                URDFSpec(
                    name=object_name,
                    pose=pose,
                    asset=None,  # Needs loading
                    urdf_path=urdf_path,
                    asset_config=asset_config,
                    physics=obj.physics,
                )
            )
            logger.debug(f"Generated URDFSpec for '{object_name}' at {urdf_path}")

        logger.debug(f"Generated {len(urdf_specs)} URDFSpec objects for rigid objects")
        return urdf_specs

    # === UTILITIES ===

    def _should_load_object(self, object_name: str, scene_file: SceneFileConfig) -> bool:
        """Whether a scene-file body loads, per its include/exclude patterns.

        Delegates to the shared ``SceneFileConfig.should_include`` so the include/exclude
        SUBSET rule is identical across MuJoCo / IsaacGym / IsaacSim — a scene file's body set
        is the same regardless of which backend loads it.
        """
        return scene_file.should_include(object_name)

    # === ASSET CONFIGURATION ===

    def _classify_scene_links(self, urdf) -> dict[str, bool]:
        """Map each scene-file link name -> is_static, from the URDF joint structure.

        A link welded to its parent by a ``fixed`` joint is static; a link on a
        ``floating`` joint is a free body. A link with no parent joint (e.g. a lone root)
        defaults to static (welded to world). This is the file-driven free/static rule,
        matching MuJoCo's freejoint test and IsaacSim's kinematic flag.
        """
        joint_type_by_child = {j.child: j.type for j in urdf.robot.joints}
        # Only fixed/floating are meaningful in a scene file: any other joint type (revolute,
        # prismatic, ...) would silently weld an articulated body, so reject it up front.
        unsupported = {c: t for c, t in joint_type_by_child.items() if t not in ("fixed", "floating")}
        if unsupported:
            raise ValueError(
                f"Scene URDF has unsupported joint types {unsupported}; scene files support only "
                f"'fixed' (static) and 'floating' (free) joints (articulations are not scene files)."
            )
        return {
            link.name: joint_type_by_child.get(link.name, "fixed") != "floating"
            for link in urdf.robot.links
            if link.name != "world"
        }

    def _get_asset_config_for_scene_object(
        self, object_name: str, scene_file: SceneFileConfig, is_static: bool = True
    ) -> AssetConfig:
        """Get asset configuration for a scene-file body.

        ``is_static`` (from the file's joint structure) maps to ``fix_base_link``: a
        welded link is fixed-base, a floating link keeps the movable default.
        """
        defaults = self._get_default_asset_config(is_scene_object=is_static)
        # Shared per-body physics resolution (same rule every backend uses — see
        # SceneFileConfig.resolve_physics), so IsaacGym and IsaacSim apply identical overrides.
        physics_config = scene_file.resolve_physics(object_name)
        return self._apply_physics_config(defaults, physics_config, object_name)

    def _get_asset_config_for_rigid_object(self, object_name: str, obj: RigidObjectConfig) -> AssetConfig:
        """Get asset configuration for a rigid object.

        A ``fixed`` object is static (welded to the world) — IsaacGym expresses that as
        ``fix_base_link=True`` on the asset; a free object keeps the movable default.
        """
        defaults = self._get_default_asset_config(is_scene_object=obj.fixed)
        physics_config = obj.physics
        return self._apply_physics_config(defaults, physics_config, object_name)

    def _get_default_asset_config(self, is_scene_object: bool) -> AssetConfig:
        """Create default asset config with type-appropriate defaults"""
        return AssetConfig(
            fix_base_link=is_scene_object,  # Scene objects fixed, rigid objects movable
            angular_damping=0.0 if is_scene_object else 0.1,
            linear_damping=0.0 if is_scene_object else 0.1,
        )

    def _apply_physics_config(self, asset_config, physics_config, object_name: str):
        """Apply a ``PhysicsConfig`` override onto the IsaacGym ``AssetConfig`` (load-time options).

        Static-vs-free is NOT taken from here — it comes from ``fixed`` via
        ``_get_default_asset_config(is_scene_object=...)`` (the single source of truth), so this
        only layers physics overrides: ``mass`` is post-creation (see
        ``apply_physics_properties_to_actor``), ``density`` is a load-time AssetOption, and the
        PhysX solver knobs (damping / velocity limits) come from the shared ``physics.physx``
        sub-config. ``None`` on ``physx`` keeps the AssetConfig defaults.
        """
        if not physics_config:
            return asset_config

        logger.debug(f"Applying physics config for '{object_name}'")

        # density + the PhysX solver knobs (damping / velocity limits) go through the same helper the
        # robot path uses (physics.apply_physx_asset_options), so an object and a robot link map these
        # load-time options identically. AssetConfig shares gymapi.AssetOptions' field names, so the
        # helper writes onto it directly. None on density/physx keeps AssetConfig defaults.
        apply_physx_asset_options(asset_config, physics_config)

        return asset_config

    # === ASSET LOADING ===

    def _load_asset_from_urdf(self, urdf_path: str | None, asset_config: AssetConfig):
        """Load asset from URDF file"""
        if not urdf_path:
            raise ValueError("Missing URDF path")

        # Each URDF (standalone object or per-link scene tempfile) resolves its meshes relative
        # to its own directory — no shared loader-wide "original dir" state (which would
        # cross-contaminate across multiple scene files).
        asset_root = str(Path(urdf_path).parent)
        asset_file = Path(urdf_path).name

        logger.debug(f"Loading URDF asset: asset_root='{asset_root}', asset_file='{asset_file}'")

        return self._load_gym_asset(asset_root, asset_file, asset_config)

    def _load_gym_asset(self, asset_root, asset_file, asset_cfg):
        """Load object asset using IsaacGym"""
        asset_path = str(Path(asset_root) / asset_file)
        gym_asset_root = str(Path(asset_path).parent)
        gym_asset_file = Path(asset_path).name

        asset_options = gymapi.AssetOptions()
        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
        ]

        # Apply asset configuration options
        for option in asset_config_options:
            if hasattr(asset_cfg, option):
                value = getattr(asset_cfg, option)
                if value is not None:
                    setattr(asset_options, option, value)

        object_asset = self.gym.load_asset(self.sim, gym_asset_root, gym_asset_file, asset_options)

        if object_asset is None:
            logger.error(f"IsaacGym returned None asset for '{gym_asset_file}' - check URDF file and paths")
        else:
            logger.debug(f"Successfully loaded asset '{gym_asset_file}' with density={asset_options.density}")

        return object_asset

    # === URDF PARSING & TRANSFORMATION ===

    def _parse_scene_urdf_with_yourdfpy(self, scene_urdf_path, source):
        """Parse scene URDF using yourdfpy"""
        logger.debug(f"Parsing scene URDF with yourdfpy: {scene_urdf_path}")

        # Use proper filename handler to resolve mesh paths relative to URDF file
        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=scene_urdf_path)

        # Load with scene graph for transform extraction
        urdf = yourdfpy.URDF.load(
            scene_urdf_path,
            load_meshes=False,  # Skip mesh loading for faster parsing
            build_scene_graph=True,  # Build scene graph for transforms
            filename_handler=filename_handler,  # Proper path resolution
        )

        logger.debug(
            f"Parsed URDF: {urdf.robot.name} with {len(urdf.robot.links)} links, {len(urdf.robot.joints)} joints"
        )
        logger.debug(f"Scene graph built: {urdf.scene is not None}")

        return urdf

    @contextmanager
    def _create_individual_link_urdf_tempfile(self, link, link_name, scene_urdf_path):
        """Create individual URDF file for a single link"""
        # Create minimal robot with just this link
        minimal_robot = Robot(name=link_name)
        minimal_robot.links = [link]
        minimal_robot.joints = []  # No joints for single link
        minimal_robot.materials = []  # Materials embedded in link visuals

        # Use proper filename handler
        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=scene_urdf_path)

        minimal_urdf = yourdfpy.URDF(
            robot=minimal_robot, build_scene_graph=False, load_meshes=False, filename_handler=filename_handler
        )

        # Generate XML string
        xml_string_bytes = minimal_urdf.write_xml_string()

        # Convert bytes to string if needed
        if isinstance(xml_string_bytes, bytes):
            xml_string = xml_string_bytes.decode("utf-8")
        else:
            xml_string = xml_string_bytes

        logger.debug(f"Generated URDF for '{link_name}' ({len(xml_string)} chars)")

        # Resolve meshes relative to THIS scene file's own directory (threaded in, not shared
        # loader state) so looping multiple scene files cannot cross-contaminate mesh paths.
        urdf_dir = str(Path(scene_urdf_path).parent)

        # Fix mesh paths to absolute paths
        xml_string = self._fix_mesh_paths_in_urdf(xml_string, link_name, urdf_dir)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", prefix=f"{link_name}_", dir=urdf_dir, delete=False
        ) as temp_file:
            temp_file.write(xml_string)
            temp_file.close()

            logger.debug(f"Created temporary URDF for '{link_name}': {temp_file.name}")

            try:
                # Return a simple object with the name attribute
                yield type("TempFile", (), {"name": temp_file.name})()
            finally:
                # Clean up the temporary file
                try:
                    Path(temp_file.name).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file.name}: {e}")

    def _fix_mesh_paths_in_urdf(self, xml_string, link_name, urdf_dir):
        """Fix relative mesh paths to absolute paths in URDF XML, relative to ``urdf_dir``.

        ``urdf_dir`` is the directory of the scene file the link came from, threaded in by the
        caller so it never relies on shared loader state (which would break multi-file loads).
        """
        # Find all mesh filename references in the XML
        mesh_matches = re.findall(r'filename="([^"]*)"', xml_string)

        if not mesh_matches:
            logger.debug(f"No mesh references found in URDF for '{link_name}'")
            return xml_string

        logger.debug(f"Found {len(mesh_matches)} mesh references in '{link_name}': {mesh_matches}")

        fixed_xml = xml_string
        fixes_applied = 0

        for mesh_path in mesh_matches:
            if not Path(mesh_path).is_absolute():  # Only fix relative paths
                # Convert relative path to absolute path based on the scene file's directory
                absolute_mesh_path = str(Path(urdf_dir) / mesh_path)

                logger.debug(f"Fixing mesh path: '{mesh_path}' -> '{absolute_mesh_path}'")

                # Apply the fix regardless of file existence (IsaacGym will handle missing files)
                fixed_xml = fixed_xml.replace(f'filename="{mesh_path}"', f'filename="{absolute_mesh_path}"')
                fixes_applied += 1
            else:
                logger.debug(f"Already absolute path: '{mesh_path}'")

        if fixes_applied > 0:
            logger.debug(f"Applied {fixes_applied} mesh path fixes for '{link_name}'")

        return fixed_xml

    def _extract_world_transform_from_scene(self, urdf, link_name, file_world_transform):
        """World pose of a link = file world transform ∘ the link's relative transform.

        ``file_world_transform`` is the 4x4 placing the whole file in the world; the
        link's relative transform (within the file) comes from the URDF scene graph.
        Composing them preserves inter-body relative poses while placing the file at its
        configured world pose. Returns ``[x,y,z, qx,qy,qz,qw]`` (xyzw) for IsaacGym.
        """
        if urdf.scene is None:
            logger.error(f"No scene graph available for link '{link_name}'")
            raise RuntimeError(f"No scene graph available for link '{link_name}'")

        # Relative transform of the link within the file, then compose with the file pose.
        relative_transform = urdf.get_transform(frame_to=link_name, frame_from="world")
        world_transform = file_world_transform @ relative_transform

        # Extract translation and rotation
        translation = tra.translation_from_matrix(world_transform)
        quaternion_wxyz = tra.quaternion_from_matrix(world_transform)

        # Convert from [w, x, y, z] to [x, y, z, w] format for IsaacGym
        quaternion_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]

        pose = translation.tolist() + quaternion_xyzw
        logger.debug(f"Extracted world pose for '{link_name}': {pose[:3]} (translation)")

        return pose

    def get_initial_pose(self, object_name: str) -> list[float]:
        """Get initial pose for an object by name

        Args:
            object_name: Name of the object to get pose for

        Returns:
            Initial pose [x, y, z, qx, qy, qz, qw]

        Raises:
            KeyError: If object not found in loaded initial states
        """
        if object_name not in self.loaded_initial_states:
            available = list(self.loaded_initial_states.keys())
            raise KeyError(f"Object '{object_name}' not found in loaded initial states. Available: {available}")
        return self.loaded_initial_states[object_name]
