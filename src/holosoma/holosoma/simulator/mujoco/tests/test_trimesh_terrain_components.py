"""MuJoCo collides a mesh geom as its CONVEX HULL, so concave terrain needs one geom per component.

A stepped/stair course is concave. Emitted as a single mesh geom, its collision surface is the hull
over every vertex — a smooth ramp spanning the whole course that sits *above* the true ground near
the start, so a robot spawned there begins embedded in the terrain. Nothing errors: the mesh loads,
the geom is created, and the robot simply sinks.

These tests work on the shipped terrain mesh directly (no simulator needed) and assert the property
``MujocoSceneManager._create_trimesh`` relies on: per-component hulls reproduce the true surface,
while a single hull does not.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import trimesh

from holosoma.utils.path import resolve_data_file_path

# No simulator is needed, but this module sits inside the holosoma.simulator.mujoco package, whose
# __init__ imports mujoco, so it can only run where mujoco is installed: the mujoco job.
pytestmark = pytest.mark.mujoco

TERRAIN_OBJ = "holosoma/data/terrains/terrain.obj"
# The spawn point run_sim uses (env origin is the world origin; robot z comes from init_state).
SPAWN_XY = (0.0, 0.0)


def _load_terrain() -> trimesh.Trimesh:
    mesh = trimesh.load(resolve_data_file_path(TERRAIN_OBJ), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = cast("trimesh.Trimesh", mesh.dump(concatenate=True))
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def _surface_z(meshes: list[trimesh.Trimesh], x: float, y: float) -> float | None:
    """Highest downward ray hit at (x, y) across ``meshes`` — the collision surface height."""
    best: float | None = None
    for mesh in meshes:
        hits, _, _ = mesh.ray.intersects_location(
            ray_origins=np.array([[x, y, 50.0]]), ray_directions=np.array([[0.0, 0.0, -1.0]])
        )
        if len(hits):
            top = float(hits[:, 2].max())
            best = top if best is None else max(best, top)
    return best


def test_terrain_mesh_is_concave_and_splits_into_components() -> None:
    """The premise: the shipped course is multi-body, so splitting it is meaningful."""
    mesh = _load_terrain()
    components = [c for c in mesh.split() if len(c.vertices) and len(c.faces)]
    assert len(components) > 1, "a single-component mesh would make the per-component split a no-op"


def test_single_convex_hull_would_bury_the_spawn_point() -> None:
    """Documents the failure this split exists to prevent — the regression is silent otherwise."""
    mesh = _load_terrain()
    true_z = _surface_z([mesh], *SPAWN_XY)
    hull_z = _surface_z([mesh.convex_hull], *SPAWN_XY)

    assert true_z is not None and hull_z is not None
    # The hull lifts the ground at the start of the course well above the real surface.
    assert hull_z > true_z + 0.1, (
        f"expected the whole-mesh hull to sit above the true ground at {SPAWN_XY}; got hull={hull_z} vs true={true_z}"
    )


@pytest.mark.parametrize("x", [0.0, 2.0, 5.0, 10.0, 20.0, 70.0])
def test_per_component_hulls_reproduce_the_true_surface(x: float) -> None:
    """Each component is convex, so per-component hulls collide as the real geometry does."""
    mesh = _load_terrain()
    hulls = [c.convex_hull for c in mesh.split() if len(c.vertices) and len(c.faces)]

    true_z = _surface_z([mesh], x, SPAWN_XY[1])
    hull_z = _surface_z(hulls, x, SPAWN_XY[1])

    assert true_z is not None and hull_z is not None
    assert hull_z == pytest.approx(true_z, abs=1e-6), f"collision surface differs from the mesh at x={x}"


def test_spawn_height_clears_the_collision_surface() -> None:
    """The robot's ``init_state`` z must sit above the terrain under the spawn point."""
    from holosoma.config_values.robot import ROBOT_REGISTRY

    mesh = _load_terrain()
    hulls = [c.convex_hull for c in mesh.split() if len(c.vertices) and len(c.faces)]
    surface_z = _surface_z(hulls, *SPAWN_XY)
    spawn_z = ROBOT_REGISTRY["g1_29dof"].init_state.pos[2]

    assert surface_z is not None
    # Pelvis spawn height, less standing leg length, still has to clear the ground.
    assert spawn_z > surface_z, f"g1_29dof spawns at z={spawn_z} but the terrain there is z={surface_z}"


def _terrain_geom_meshes(mesh: trimesh.Trimesh) -> list[npt.NDArray[np.float64]]:
    """Run ``MujocoSceneManager._create_trimesh`` on ``mesh``; return each emitted geom's vertices."""
    import types

    import mujoco

    from holosoma.managers.terrain.base import TerrainTermBase
    from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager

    stub = types.SimpleNamespace(world_spec=mujoco.MjSpec())
    terrain_state = types.SimpleNamespace(mesh=mesh, name="terrain", hide_visual=False)
    MujocoSceneManager._create_trimesh(cast("MujocoSceneManager", stub), cast("TerrainTermBase", terrain_state))
    return [np.asarray(m.uservert, dtype=np.float64).reshape(-1, 3) for m in stub.world_spec.meshes]


def test_open_ground_plane_next_to_closed_box_is_kept() -> None:
    """An open quad ground must not be dropped because a closed obstacle exists."""
    plane = trimesh.Trimesh(
        vertices=[[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    box = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    box.apply_translation([2.0, 0.0, 0.1])

    geom_verts = _terrain_geom_meshes(cast("trimesh.Trimesh", trimesh.util.concatenate([plane, box])))

    assert len(geom_verts) == 2
    assert any(len(v) == 4 and np.allclose(np.abs(v[:, :2]), 5.0) and np.allclose(v[:, 2], 0.0) for v in geom_verts)
    assert any(len(v) == 8 and v[:, 2].max() == pytest.approx(0.2) for v in geom_verts)


def test_flat_shaded_obj_stays_one_geom() -> None:
    """A triangle soup (no shared vertices, as a flat-shaded OBJ loads) must not split per triangle."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    soup = trimesh.Trimesh(
        vertices=box.vertices[box.faces].reshape(-1, 3),
        faces=np.arange(3 * len(box.faces)).reshape(-1, 3),
        process=False,
    )

    geom_verts = _terrain_geom_meshes(soup)

    assert len(geom_verts) == 1
    assert len(geom_verts[0]) == 3 * len(box.faces)


def test_shipped_terrain_emits_one_geom_per_block() -> None:
    mesh = _load_terrain()

    assert len(_terrain_geom_meshes(mesh)) == len(mesh.split())
