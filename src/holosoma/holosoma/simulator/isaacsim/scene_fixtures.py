"""Terrain and height-scanner prim setup for the IsaacSim scene build.

Authors the ground prim (plane via ``TerrainImporterCfg`` or trimesh via ``create_prim_from_mesh``)
and, for non-flat terrain, the robot-mounted height-scanner RayCaster cfg. Returns the collision
prim paths to filter against plus the built cfg/terrain, leaving the caller to instantiate the
sensor and wire it into the scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from pxr import Usd, UsdGeom

from holosoma.simulator.isaacsim.prim_utils import resolve_robot_link_prim_expression

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from holosoma.managers.terrain import TerrainManager

TERRAIN_PRIM_PATH = "/World/ground"


class TerrainFixtures(NamedTuple):
    """Result of :func:`build_terrain_fixtures`."""

    global_collision_prims: list[str]
    height_scanner_cfg: RayCasterCfg | None
    terrain: Any  # the trimesh Terrain handle, or None for a plane / fake terrain


def _hide_prim_subtree(stage: Usd.Stage, prim_path: str) -> None:
    """Make the geometry under ``prim_path`` invisible (rendering only; physics untouched).

    Authors ``visibility = invisible`` on every Imageable prim in the subtree. The terrain root
    (e.g. ``/World/ground``) is typeless, so setting visibility only there is a no-op; the visible
    meshes are on Imageable descendants. Colliders are unaffected, so the body stays collidable.
    """
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()


def build_terrain_fixtures(
    terrain_manager: TerrainManager, scene: InteractiveScene, robot_base_body_name: str
) -> TerrainFixtures:
    """Author the terrain ground prim + optional height scanner cfg.

    Parameters
    ----------
    terrain_manager : TerrainManager
        Source of the ``locomotion_terrain`` state (mesh type, friction, visibility).
    scene : InteractiveScene
        The scene being built; its ``cfg.num_envs`` / ``cfg.env_spacing`` size the plane importer.
    robot_base_body_name : str
        Robot base-link body name the height scanner attaches to.

    Returns
    -------
    TerrainFixtures
        Collision prim paths to filter against, the height-scanner cfg (``None`` for flat/fake
        terrain), and the trimesh ``Terrain`` handle (``None`` unless a mesh terrain was created).

    Raises
    ------
    ValueError
        If the terrain mesh type is unsupported.
    """
    terrain_state = terrain_manager.get_state("locomotion_terrain")

    height_scanner_cfg: RayCasterCfg | None = None
    if terrain_state.mesh_type not in ["fake", None]:
        # Add a height scanner to the torso to detect the height of the terrain mesh
        # TODO: Scene USD files need ground mapping
        height_scanner_cfg = RayCasterCfg(
            prim_path=resolve_robot_link_prim_expression(robot_base_body_name),
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=[TERRAIN_PRIM_PATH],
        )

    global_collision_prims: list[str] = []
    terrain = None
    if terrain_state.mesh_type == "plane":
        terrain_config = TerrainImporterCfg(
            prim_path=TERRAIN_PRIM_PATH,
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=terrain_state.static_friction,
                dynamic_friction=terrain_state.dynamic_friction,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        terrain_config.num_envs = scene.cfg.num_envs
        terrain_config.env_spacing = scene.cfg.env_spacing
        terrain_config.class_type(terrain_config)
        global_collision_prims.append(terrain_config.prim_path)
        # Hide the ground plane's visual while keeping its collider; avoids z-fighting a scene
        # USD's own floor. Applied to the whole /World/ground subtree.
        if terrain_state.hide_visual:
            _hide_prim_subtree(stage_utils.get_current_stage(), TERRAIN_PRIM_PATH)
    elif terrain_state.mesh_type in ["trimesh", "load_obj"]:
        terrain = terrain_state.terrain
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=terrain_state.static_friction,
            dynamic_friction=terrain_state.dynamic_friction,
            restitution=terrain_state.restitution,
        )
        create_prim_from_mesh(
            TERRAIN_PRIM_PATH,
            terrain.mesh,
            visual_material=visual_material,
            physics_material=physics_material,
            translation=(0.0, 0.0, 0.0),
        )
        global_collision_prims.append(TERRAIN_PRIM_PATH)
        print("[INFO] Successfully created custom terrain mesh")
    else:
        raise ValueError(f"Unsupported terrain mesh type: {terrain_state.mesh_type}")

    return TerrainFixtures(global_collision_prims, height_scanner_cfg, terrain)
