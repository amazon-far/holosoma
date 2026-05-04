"""Generate (h, w, direction) sweep of IsaacLab-style stair tiles + matching
motions, then aggregate SR per height for an "Ascending vs Descending" bar
chart (matches the figure layout the user shared).

Usage:
    # Phase 1: generate motion library
    python scripts/eval/unreal_engine/sweep_isaaclab_stair_sr.py generate \
        --out_dir /tmp/dagger_sweep/sweep_isaaclab_h15-35

    # Phase 2: caller runs eval_success_rate.py on the generated dir.
    # Phase 3: aggregate + plot
    python scripts/eval/unreal_engine/sweep_isaaclab_stair_sr.py plot \
        --motion_dir /tmp/dagger_sweep/sweep_isaaclab_h15-35 \
        --per_motion_tsv /tmp/dagger_sweep/sweep_per_motion.tsv \
        --out /home/kyungminlee/work/holosoma/videos/.../stair_sr_sweep.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Reuse the IsaacLab stair generator's mesh/motion code by importing main()
# arguments. We invoke its private helpers directly.
sys.path.append(str(Path(__file__).parent))
from generate_isaaclab_stair_motion import (  # noqa: E402
    _build_isaaclab_stair_mesh,
    _height_profile,
)
import trimesh  # noqa: E402
from generate_synthetic_stair_motions import (  # noqa: E402
    _STANDING_DOF,
    _measure_pelvis_above_foot,
    _setup_default_qpos,
)

import mujoco  # noqa: E402

HEIGHTS_CM = [15, 20, 25, 30, 35]
WIDTH_MAX_CM = 70
WIDTH_STEP_CM = 5
NUM_STEPS = 6
LIN_VEL_X = 0.6
FPS = 50
PLATFORM_WIDTH = 3.0
BORDER_WIDTH = 1.0
PELVIS_HEIGHT_DEFAULT = 0.78


def _build_one_motion(*, kind, h_m, w_m, num_steps, model, base_qpos,
                       body_names, joint_names_motion, pelvis_above_ground,
                       motion_seconds, lin_vel_x, samples_per_meter=10):
    # tile_size such that IsaacLab's pyramid_stairs_terrain generates
    # EXACTLY num_steps. The formula it uses internally is
    #   num_steps = (size - 2*border - platform) // (2*w) + 1
    # so we invert to get size:
    #   size = 2*(num_steps - 1)*w + 2*border + platform
    # We rely on IsaacLab's own border ring (1 m wide) for plateau room
    # past the outermost step — adding extra outer padding here would
    # make IsaacLab fill it with more stairs and break the 6-step count.
    tile_size = 2 * (num_steps - 1) * w_m + 2 * BORDER_WIDTH + PLATFORM_WIDTH
    # Use the box-stack mesh (the fixed builder in generate_isaaclab_stair_
    # motion._build_isaaclab_stair_mesh: concatenate IsaacLab boxes, then
    # merge_vertices + process(validate=True) to dedupe coincident faces
    # and produce a single low-face-count tile mesh that PhysX can scale
    # to 90 tiles without overflowing collision buffers).
    mesh, origin = _build_isaaclab_stair_mesh(
        kind=kind, step_height=h_m, step_width=w_m,
        tile_size=tile_size, border_width=BORDER_WIDTH,
        platform_width=PLATFORM_WIDTH,
    )
    mesh.apply_translation(-origin)

    dt = 1.0 / FPS
    T = max(int(round(motion_seconds * FPS)), 30)

    # Set motion goal on the BORDER RING (the flat plain just past the
    # stairs — for inv_pyramid the top plain at z=(num_steps+1)*h, for
    # pyramid the bottom plain at z=-(num_steps+1)*h). Earlier we stopped
    # one step short on the outermost stair top, but that left the ghost
    # frozen at the stair edge instead of stepping onto the natural flat
    # ground beyond. xs_max sits ~0.4 m inside the outer tile bound to
    # keep the ghost away from the inter-tile gap where the robot would
    # fall off.
    xs_max = float(mesh.bounds[1, 0]) - 0.4
    xs_raw = np.arange(T) * dt * lin_vel_x
    xs_path = np.minimum(xs_raw, xs_max)
    zs_top = _height_profile(mesh, xs_path, y=0.0)
    zs_path = zs_top + pelvis_above_ground

    data = mujoco.MjData(model)
    body_pos = np.zeros((T, len(body_names), 3), dtype=np.float64)
    body_quat = np.zeros((T, len(body_names), 4), dtype=np.float64)
    joint_pos = np.zeros((T, model.nq), dtype=np.float32)
    for t in range(T):
        qpos = base_qpos.copy()
        qpos[0] = float(xs_path[t]); qpos[1] = 0.0; qpos[2] = float(zs_path[t])
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[:] = qpos; data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos
        body_quat[t] = data.xquat
        joint_pos[t] = qpos

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

    return {
        "fps": np.array([FPS], dtype=np.int64),
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


def cmd_generate(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    joint_names_motion = list(_STANDING_DOF.keys())
    base_qpos = _setup_default_qpos(model)
    pelvis_above_ground = _measure_pelvis_above_foot(model, base_qpos)
    print(f"[INFO] pelvis_above_foot = {pelvis_above_ground:.4f} m")

    manifest = []
    for kind, dir_label in [("inv_pyramid", "ascend"), ("pyramid", "descend")]:
        for h_cm in HEIGHTS_CM:
            for w_cm in range(h_cm + 5, WIDTH_MAX_CM + 1, WIDTH_STEP_CM):
                h_m, w_m = h_cm / 100.0, w_cm / 100.0
                # Motion length: cover 6 steps + extension onto border ring +
                # a few seconds buffer. xs_max ≈ tile_half - 0.4, and the ghost
                # walks at lin_vel_x, so ensure we have enough frames to reach
                # the border-ring goal before clamping kicks in.
                tile_size_m = 2 * (NUM_STEPS - 1) * w_m + 2 * BORDER_WIDTH + PLATFORM_WIDTH
                xs_max_m = tile_size_m / 2.0 - 0.4
                motion_seconds = max(8.0, xs_max_m / LIN_VEL_X + 3.0)
                payload = _build_one_motion(
                    kind=kind, h_m=h_m, w_m=w_m, num_steps=NUM_STEPS,
                    model=model, base_qpos=base_qpos,
                    body_names=body_names, joint_names_motion=joint_names_motion,
                    pelvis_above_ground=pelvis_above_ground,
                    motion_seconds=motion_seconds, lin_vel_x=LIN_VEL_X,
                    samples_per_meter=args.samples_per_meter,
                )
                # Filename encodes h, w, direction so we can parse later.
                base = f"isaaclab_{dir_label}_h{h_cm:02d}_w{w_cm:02d}_n{NUM_STEPS:02d}"
                out_path = out_dir / f"{base}.npz"
                np.savez(out_path, **payload)
                manifest.append({
                    "name": base, "kind": kind, "direction": dir_label,
                    "h_cm": h_cm, "w_cm": w_cm, "n_steps": NUM_STEPS,
                    "motion_seconds": motion_seconds,
                })
                print(f"  wrote {base}.npz  ({payload['joint_pos'].shape[0]} frames)")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote {len(manifest)} motions to {out_dir}")


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build labels[name] = ok|fail
    labels = {}
    for line in open(args.per_motion_tsv):
        s, p = line.strip().split("\t")
        labels[Path(p).stem] = s == "OK"

    # Aggregate per (direction, h_cm)
    rows = {("ascend", h): [0, 0] for h in HEIGHTS_CM}
    rows.update({("descend", h): [0, 0] for h in HEIGHTS_CM})
    pat = re.compile(r"isaaclab_(ascend|descend)_h(\d+)_w(\d+)_")
    for name, ok in labels.items():
        m = pat.match(name)
        if not m: continue
        d, h, _w = m.group(1), int(m.group(2)), int(m.group(3))
        if (d, h) not in rows: continue
        rows[(d, h)][0] += int(ok)
        rows[(d, h)][1] += 1

    sr = {key: (ok / n if n else 0.0) for key, (ok, n) in rows.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, direction, title in [(axes[0], "ascend", "Ascending Stairs"),
                                  (axes[1], "descend", "Descending Stairs")]:
        xs = np.arange(len(HEIGHTS_CM))
        ys = [sr[(direction, h)] for h in HEIGHTS_CM]
        ns = [rows[(direction, h)][1] for h in HEIGHTS_CM]
        bars = ax.bar(xs, ys, color="tab:blue", width=0.6)
        for x, y, n in zip(xs, ys, ns):
            ax.text(x, y + 0.02, f"{y:.0%}\n(n={n})", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{h} cm" for h in HEIGHTS_CM])
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(title)
        if direction == "ascend":
            ax.set_ylabel("Success rate (root 0.5m, avg over widths)")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[OK] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out_dir", required=True)
    g.add_argument("--mujoco_xml",
                   default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
    g.add_argument("--samples_per_meter", type=int, default=10,
                   help="Heightmap mesh grid density. Default 10 (= 0.1 m grid). "
                   "Lower = fewer mesh faces; with 90 tiles concatenated, dense meshes "
                   "(samples_per_meter=20 → 9M faces total) overwhelm PhysX collision "
                   "and the robot falls through the floor.")
    g.set_defaults(fn=cmd_generate)
    pl = sub.add_parser("plot")
    pl.add_argument("--motion_dir", required=True)
    pl.add_argument("--per_motion_tsv", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_plot)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
