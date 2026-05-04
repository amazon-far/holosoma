"""Generate one IsaacLab pyramid-stair tile + matching kinematic motion .npz.

Drop-in alternative to ``generate_synthetic_stair_motions.py`` that uses
the same pyramid-stair geometry IsaacLab's
``MeshPyramidStairsTerrainCfg`` / ``MeshInvertedPyramidStairsTerrainCfg``
build (and which PHUMA2's ``STAIRS_TERRAINS_CFG`` is configured around)
for the tile mesh, instead of a hand-built linear stack of step boxes.

The mesh-builder + border helpers are inlined here so we can run from
the holosoma `hssim` env without pulling in IsaacLab's package
``__init__`` (which transitively imports ``omni.log``). The geometry is
byte-equivalent to the IsaacLab functions
``pyramid_stairs_terrain`` / ``inverted_pyramid_stairs_terrain``
(`isaaclab/terrains/trimesh/mesh_terrains.py`).

Output .npz follows the synth_stair schema so ``rollout_single_policy.py``
+ ``visualize.py`` can consume it unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

# Reuse the standing-pose helpers from the existing synth-stair generator.
sys.path.append(str(Path(__file__).parent))
from generate_synthetic_stair_motions import (  # noqa: E402
    _STANDING_DOF,
    _measure_pelvis_above_foot,
    _setup_default_qpos,
)


# ---------------------------------------------------------------------------
# Geometry — copied verbatim (whitespace + variable names) from
# isaaclab/terrains/trimesh/{utils,mesh_terrains}.py to avoid having to
# import the IsaacLab package (its __init__ requires the omni runtime).
# Keep this in sync if IsaacLab changes the geometry.
# ---------------------------------------------------------------------------


def _make_border(size, inner_size, height, position):
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    box_dims = (size[0], thickness_y, height)
    box_pos = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_pos = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    bot = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_dims = (thickness_x, inner_size[1], height)
    box_pos = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    box_pos = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    return [left, right, top, bot]


def _pyramid_stairs_terrain(*, size, border_width, step_height, step_width,
                            platform_width, holes=False):
    num_steps_x = (size[0] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps_y = (size[1] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    meshes_list = []
    if border_width > 0.0 and not holes:
        border_center = [0.5 * size[0], 0.5 * size[1], -step_height / 2]
        border_inner = (size[0] - 2 * border_width, size[1] - 2 * border_width)
        meshes_list += _make_border(size, border_inner, step_height, border_center)

    terrain_center = [0.5 * size[0], 0.5 * size[1], 0.0]
    terrain_size = (size[0] - 2 * border_width, size[1] - 2 * border_width)
    for k in range(num_steps):
        if holes:
            box_size = (platform_width, platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = (k + 2) * step_height
        box_dims = (box_size[0], step_width, box_height)
        for sign in (+1, -1):
            box_pos = (terrain_center[0], terrain_center[1] + sign * (terrain_size[1] / 2.0 - box_offset), box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
        if holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        for sign in (+1, -1):
            box_pos = (terrain_center[0] + sign * (terrain_size[0] / 2.0 - box_offset), terrain_center[1], box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
    return meshes_list, origin


def _inverted_pyramid_stairs_terrain(*, size, border_width, step_height, step_width,
                                     platform_width, holes=False):
    num_steps_x = (size[0] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps_y = (size[1] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    meshes_list = []
    if border_width > 0.0 and not holes:
        border_center = [0.5 * size[0], 0.5 * size[1], -0.5 * step_height]
        border_inner = (size[0] - 2 * border_width, size[1] - 2 * border_width)
        meshes_list += _make_border(size, border_inner, step_height, border_center)

    terrain_center = [0.5 * size[0], 0.5 * size[1], 0.0]
    terrain_size = (size[0] - 2 * border_width, size[1] - 2 * border_width)
    for k in range(num_steps):
        if holes:
            box_size = (platform_width, platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = total_height - (k + 1) * step_height
        box_dims = (box_size[0], step_width, box_height)
        for sign in (+1, -1):
            box_pos = (terrain_center[0], terrain_center[1] + sign * (terrain_size[1] / 2.0 - box_offset), box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
        if holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        for sign in (+1, -1):
            box_pos = (terrain_center[0] + sign * (terrain_size[0] / 2.0 - box_offset), terrain_center[1], box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
    return meshes_list, origin


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def _build_isaaclab_stair_mesh(
    *, kind: str, step_height: float, step_width: float,
    tile_size: float, border_width: float, platform_width: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return ``(combined_mesh, origin)`` for a single IsaacLab stair tile."""
    common = dict(
        size=(tile_size, tile_size),
        border_width=border_width,
        step_height=step_height,
        step_width=step_width,
        platform_width=platform_width,
        holes=False,
    )
    if kind == "pyramid":
        boxes, origin = _pyramid_stairs_terrain(**common)
    elif kind == "inv_pyramid":
        boxes, origin = _inverted_pyramid_stairs_terrain(**common)
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'pyramid' or 'inv_pyramid')")
    # IsaacLab returns a list of axis-aligned boxes (one per step strip + the
    # central platform + 4 border strips). Concatenating with merge_vertices
    # gives a tile mesh with crisp vertical risers and tiny face count
    # (~85 boxes × 12 faces = ~1k faces per tile), so 90 tiles in one PhysX
    # scene stays well under the collision-buffer ceiling. process(validate=
    # True) removes the duplicate faces that show up where adjacent step
    # boxes share a wall — without it, those internal walls trigger
    # ambiguous collision resolution and feet get stuck inside the higher
    # step.
    combined = trimesh.util.concatenate(boxes)
    combined.merge_vertices()
    combined.process(validate=True)
    return combined, np.asarray(origin, dtype=np.float64)


def _surface_mesh_from_boxes(
    boxes: list[trimesh.Trimesh], *, tile_size: float, samples_per_meter: int = 20,
) -> trimesh.Trimesh:
    """Approximate the upper envelope of a set of axis-aligned boxes as a
    heightmap-style surface mesh.

    For each grid sample ``(x, y)`` we compute the maximum z that lies
    inside any box (= "ground height under this point"). The result is then
    triangulated as a regular grid.
    """
    # Gather each box's axis-aligned bounds (n_boxes × 6 = (x_lo, y_lo, z_lo, x_hi, y_hi, z_hi)).
    bbs = np.stack([b.bounds.flatten()[[0, 1, 2, 3, 4, 5]] for b in boxes], axis=0)
    bb_lo, bb_hi = bbs[:, :3], bbs[:, 3:]

    # Use the union of all box bounds as the sampling extent — covers
    # both the inner stair region and the surrounding border ring.
    x_lo = float(bb_lo[:, 0].min()); x_hi = float(bb_hi[:, 0].max())
    y_lo = float(bb_lo[:, 1].min()); y_hi = float(bb_hi[:, 1].max())
    z_floor = float(bb_lo[:, 2].min())

    nx = int(round((x_hi - x_lo) * samples_per_meter)) + 1
    ny = int(round((y_hi - y_lo) * samples_per_meter)) + 1
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

    # Heightmap: max z_hi over boxes that contain (x, y) in their xy footprint.
    # Vectorised: for each box, mask cells inside its xy bbox and take max.
    z_grid = np.full_like(gx, z_floor, dtype=np.float64)
    for (lo_x, lo_y, _, hi_x, hi_y, hi_z) in bbs:
        m = (gx >= lo_x) & (gx <= hi_x) & (gy >= lo_y) & (gy <= hi_y)
        if m.any():
            z_grid[m] = np.maximum(z_grid[m], hi_z)

    # Build vertices (top surface) and faces (two triangles per cell).
    verts_top = np.stack([gx.ravel(), gy.ravel(), z_grid.ravel()], axis=-1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[1:, 1:]; d = idx[:-1, 1:]
    faces_top = np.stack(
        [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)], axis=0
    ).reshape(-1, 3)

    # Add a flat bottom slab at z_floor so the mesh is closed and PhysX has
    # a back face. Skip if z_floor already coincides with z_grid.min() (no
    # gap → no need for a back face).
    if z_grid.min() > z_floor + 1e-6:
        verts_bot = np.stack(
            [gx.ravel(), gy.ravel(), np.full(nx * ny, z_floor)], axis=-1
        )
        offset = len(verts_top)
        faces_bot = (faces_top + offset)[:, ::-1]  # flip winding so normals face down
        verts = np.concatenate([verts_top, verts_bot], axis=0)
        faces = np.concatenate([faces_top, faces_bot], axis=0)
    else:
        verts = verts_top
        faces = faces_top

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh


def _height_profile(mesh: trimesh.Trimesh, xs: np.ndarray, y: float) -> np.ndarray:
    """For each x, return the topmost mesh-z at (x, y) via downward ray cast."""
    z_high = float(mesh.bounds[1, 2]) + 5.0
    z_floor = float(mesh.bounds[0, 2]) - 5.0
    n = xs.shape[0]
    origins = np.stack(
        [xs, np.full(n, y, dtype=np.float64), np.full(n, z_high, dtype=np.float64)],
        axis=-1,
    )
    directions = np.zeros_like(origins)
    directions[:, 2] = -1.0
    locations, ray_idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True,
    )
    out = np.full(n, z_floor, dtype=np.float64)
    for loc, ri in zip(locations, ray_idx):
        if loc[2] > out[ri]:
            out[ri] = loc[2]
    return out


def main() -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_basename", default=None,
                   help="Output .npz basename (without extension). Default = "
                   "isaaclab_<kind>_h<height_cm>_w<width_cm>_v<vel_cm_s>.")
    p.add_argument("--mujoco_xml",
                   default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
    p.add_argument("--kind", choices=["pyramid", "inv_pyramid"], default="inv_pyramid",
                   help="pyramid = peak at center (walk +x → descend); "
                   "inv_pyramid = pit at center (walk +x → ascend).")
    p.add_argument("--step_height", type=float, default=0.15)
    p.add_argument("--step_width", type=float, default=0.5)
    p.add_argument("--tile_size", type=float, default=8.0)
    p.add_argument("--border_width", type=float, default=1.0)
    p.add_argument("--platform_width", type=float, default=3.0)
    p.add_argument("--lin_vel_x", type=float, default=0.6)
    p.add_argument("--motion_seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, origin = _build_isaaclab_stair_mesh(
        kind=args.kind,
        step_height=args.step_height,
        step_width=args.step_width,
        tile_size=args.tile_size,
        border_width=args.border_width,
        platform_width=args.platform_width,
    )
    # Translate so the IsaacLab spawn origin sits at world (0, 0, 0). Robot
    # then starts at (0, 0, 0) and walks +x — same convention as the existing
    # synth_stair generator, so rollout_single_policy / visualize work unchanged.
    mesh.apply_translation(-origin)
    print(
        f"[INFO] mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces; "
        f"x ∈ [{mesh.bounds[0, 0]:.2f}, {mesh.bounds[1, 0]:.2f}]m, "
        f"z ∈ [{mesh.bounds[0, 2]:.2f}, {mesh.bounds[1, 2]:.2f}]m"
    )

    # Mujoco model for FK on each frame.
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    joint_names_motion = list(_STANDING_DOF.keys())
    base_qpos = _setup_default_qpos(model)
    pelvis_above_ground = _measure_pelvis_above_foot(model, base_qpos)
    print(f"[INFO] pelvis_above_foot (FK) = {pelvis_above_ground:.4f} m")

    dt = 1.0 / args.fps
    T = max(int(round(args.motion_seconds * args.fps)), 30)

    # Constant +x walk; pelvis z follows the stair top under each x. Clamp
    # xs slightly inside the mesh +x bound so the robot stops at the tile
    # edge instead of teleporting into empty air (where the ray cast misses
    # the mesh and we'd fall back to z_floor).
    xs_max = float(mesh.bounds[1, 0]) - 0.1
    xs_raw = np.arange(T) * dt * args.lin_vel_x
    xs_path = np.minimum(xs_raw, xs_max)
    if xs_raw.max() > xs_max:
        n_clamped = int((xs_raw > xs_max).sum())
        print(
            f"[WARN] motion length exceeds tile +x bound ({xs_max:.2f} m); "
            f"clamping last {n_clamped}/{T} frames to the edge."
        )
    zs_top = _height_profile(mesh, xs_path, y=0.0)
    zs_path = zs_top + pelvis_above_ground

    # FK per frame (root teleported, joints fixed at standing pose).
    data = mujoco.MjData(model)
    body_pos = np.zeros((T, len(body_names), 3), dtype=np.float64)
    body_quat = np.zeros((T, len(body_names), 4), dtype=np.float64)
    joint_pos = np.zeros((T, model.nq), dtype=np.float32)
    for t in range(T):
        qpos = base_qpos.copy()
        qpos[0] = float(xs_path[t])
        qpos[1] = 0.0
        qpos[2] = float(zs_path[t])
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos
        body_quat[t] = data.xquat
        joint_pos[t] = qpos

    # Body / joint velocities by finite difference (boundaries one-sided so
    # frame-0 pelvis vel is non-zero, matching the synth_stair convention).
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    if T >= 2:
        body_lin_vel[0] = (body_pos[1] - body_pos[0]) / dt
        body_lin_vel[-1] = (body_pos[-1] - body_pos[-2]) / dt
    if T >= 3:
        body_lin_vel[1:-1] = (body_pos[2:] - body_pos[:-2]) / (2 * dt)

    joint_vel = np.zeros((T, model.nv), dtype=np.float32)
    if T >= 2:
        joint_vel[0, :3] = (joint_pos[1, :3] - joint_pos[0, :3]) / dt
        joint_vel[-1, :3] = (joint_pos[-1, :3] - joint_pos[-2, :3]) / dt
    if T >= 3:
        joint_vel[1:-1, :3] = (joint_pos[2:, :3] - joint_pos[:-2, :3]) / (2 * dt)

    base = (
        args.out_basename
        or f"isaaclab_{args.kind}_h{int(round(args.step_height * 100)):03d}_"
           f"w{int(round(args.step_width * 100)):03d}_"
           f"v{int(round(args.lin_vel_x * 100)):03d}"
    )
    out_path = out_dir / f"{base}.npz"
    payload = {
        "fps": np.array([args.fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel.astype(np.float64),
        "body_ang_vel_w": body_ang_vel.astype(np.float64),
        "joint_names": np.array(
            joint_names_motion,
            dtype=f"<U{max(len(n) for n in joint_names_motion)}",
        ),
        "body_names": np.array(
            body_names, dtype=f"<U{max(len(n) for n in body_names)}"
        ),
        "mesh_vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "mesh_faces": np.asarray(mesh.faces, dtype=np.int32),
    }
    np.savez(out_path, **payload)
    print(
        f"[OK] wrote {out_path}  (T={T} frames, "
        f"x ∈ [{xs_path.min():.2f}, {xs_path.max():.2f}], "
        f"z ∈ [{zs_path.min():.2f}, {zs_path.max():.2f}])"
    )
    return out_path


if __name__ == "__main__":
    main()
