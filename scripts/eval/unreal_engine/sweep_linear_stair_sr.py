"""Generate (h, w, direction) sweep of LINEAR stair tiles using the well-tested
generate_synthetic_stair_motions._generate_motion path. Linear stairs include
1m plateaus on each side of the stair zone so the robot has safe walking room
and doesn't fall off the tile edge (which was an issue with pyramid stairs at
small step widths).

Then aggregate per-height success rate using the "did the robot reach the
goal x" criterion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.append(str(Path(__file__).parent))
from generate_synthetic_stair_motions import (  # noqa: E402
    _STANDING_DOF,
    _generate_motion,
    _measure_pelvis_above_foot,
    _setup_default_qpos,
)

HEIGHTS_CM = [15, 20, 25, 30, 35]
WIDTH_MAX_CM = 70
WIDTH_STEP_CM = 5
NUM_STEPS = 6
LIN_VEL_X = 0.6
FPS = 50


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

    rng = np.random.default_rng(0)
    manifest = []
    for direction in ["up", "down"]:
        ascending = direction == "up"
        for h_cm in HEIGHTS_CM:
            for w_cm in range(h_cm + 5, WIDTH_MAX_CM + 1, WIDTH_STEP_CM):
                payload, meta = _generate_motion(
                    model, base_qpos, body_names, joint_names_motion, rng,
                    fps=FPS, lin_vel_x=LIN_VEL_X,
                    fixed_h=h_cm / 100.0, fixed_w=w_cm / 100.0,
                    fixed_n_steps=NUM_STEPS, fixed_ascending=ascending,
                    plateau_pre=1.5, plateau_post=2.0,  # extra room
                    pelvis_above_ground=pelvis_above_ground,
                )
                # Merge coincident vertices at adjacent box boundaries (the
                # raw mesh has duplicate verts where step i's right face
                # touches step i+1's left face). DO NOT call
                # process(validate=True) here — it removes back-to-back
                # interior wall faces with opposite winding, which leaves
                # PhysX with no collision between boxes and the robot
                # falls through the side gap.
                import trimesh
                _mesh = trimesh.Trimesh(
                    vertices=payload["mesh_vertices"].astype(np.float64),
                    faces=payload["mesh_faces"].astype(np.int64),
                    process=False,
                )
                _mesh.merge_vertices()
                payload["mesh_vertices"] = np.asarray(_mesh.vertices, dtype=np.float32)
                payload["mesh_faces"] = np.asarray(_mesh.faces, dtype=np.int32)

                base = f"linear_{direction}_h{h_cm:02d}_w{w_cm:02d}_n{NUM_STEPS:02d}"
                np.savez(out_dir / f"{base}.npz", **payload)
                manifest.append({"name": base, "direction": direction,
                                 "h_cm": h_cm, "w_cm": w_cm,
                                 "T": int(payload['joint_pos'].shape[0])})
                print(f"  wrote {base}.npz  T={payload['joint_pos'].shape[0]}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote {len(manifest)} motions to {out_dir}")


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    pat = re.compile(r"linear_(up|down)_h(\d+)_w(\d+)_")
    motion_dir = Path(args.motion_dir)
    rollout_dir = Path(args.rollout_dir)

    fell = 0
    agg = defaultdict(lambda: [0, 0])
    details = []
    for npy in sorted(rollout_dir.glob("policy_*.npy")):
        name = npy.stem.replace("policy_", "")
        m = pat.match(name)
        if not m: continue
        direction, h, w = m.group(1), int(m.group(2)), int(m.group(3))
        mn = np.load(motion_dir / f"{name}.npz", allow_pickle=True)
        pol = np.load(npy, allow_pickle=True).item()['root_trans']
        mot = np.asarray(mn['body_pos_w'][:, 1, :])
        # Fall-through filter
        if pol[:, 2].min() < -2.0:
            fell += 1
            continue
        # Reach criterion: progress = (final_x - start_x) / (goal_x - start_x)
        start_x, goal_x = float(mot[0, 0]), float(mot[-1, 0])
        span = goal_x - start_x
        if abs(span) < 1e-6:
            continue
        progress = (float(pol[-1, 0]) - start_x) / span
        ok = progress >= 0.9
        agg[(direction, h)][0] += int(ok); agg[(direction, h)][1] += 1
        details.append({"name": name, "direction": direction, "h": h, "w": w,
                        "progress": progress, "ok": ok})

    print(f"fall-through: {fell} (excluded from SR)")
    print(f"\n{'dir':>5} | {'h':>2} | OK/n  SR")
    for d in ["up", "down"]:
        for h in HEIGHTS_CM:
            ok, n = agg[(d, h)]
            sr = ok/n*100 if n else 0
            print(f"{d:>5} | {h:>2} | {ok:>2}/{n:>2}  {sr:>5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    xs = np.arange(len(HEIGHTS_CM))
    for ax, direction, title in [(axes[0], "up", "Ascending Stairs"),
                                  (axes[1], "down", "Descending Stairs")]:
        sr = [agg[(direction, h)][0] / max(agg[(direction, h)][1], 1) for h in HEIGHTS_CM]
        ns = [agg[(direction, h)][1] for h in HEIGHTS_CM]
        ax.bar(xs, sr, 0.6, color="tab:blue")
        for x, s, n in zip(xs, sr, ns):
            ax.text(x, s + 0.02, f"{s:.0%}\n(n={n})", ha="center", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels([f"{h} cm" for h in HEIGHTS_CM])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(title)
        if direction == "up":
            ax.set_ylabel("Success rate\n(reach ≥ 90% of goal x)")
    fig.suptitle(
        f"xy_yaw model_16000 on linear stairs (n=6 steps, w sweep; {fell} fall-throughs excluded)",
        fontsize=10
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out_dir", required=True)
    g.add_argument("--mujoco_xml",
                   default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
    g.set_defaults(fn=cmd_generate)
    pl = sub.add_parser("plot")
    pl.add_argument("--motion_dir", required=True)
    pl.add_argument("--rollout_dir", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_plot)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
