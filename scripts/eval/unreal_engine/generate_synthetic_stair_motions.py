"""Generate procedural stair tiles + matching kinematic motion .npz files.

Each output .npz mirrors the schema of the
``Unreal_Engine_stair_subset200_aligned`` motions but is built from scratch:

  * ``mesh_vertices`` / ``mesh_faces``: a stack of boxes forming an
    ascending or descending staircase with random ``(h, w)`` ∈
    (``[0.10, 0.30]``, ``[0.30, 0.80]``) m and a random number of steps.
  * ``joint_pos`` / ``joint_vel``: root pose marching at ``lin_vel_x`` along
    the stair (with the pelvis z snapped to the step height profile + a
    standing pelvis offset of ~0.78 m), joint angles held at the holosoma
    default standing pose for every frame.
  * ``body_pos_w`` / ``body_quat_w`` / ``body_lin_vel_w`` /
    ``body_ang_vel_w``: forward-kinematic results for every body in the
    G1 mujoco XML (33 bodies including ``world``), so MotionLibrary's
    ``_body_indexes`` mapping resolves cleanly. Body 0 (``world``) is held
    at identity; remaining bodies are computed from ``mj_forward`` per
    frame.

The point of these motions is to give the trained student a "go straight
+x at v m/s on a random stair" command without re-implementing the
locomotion task — the student's existing observation pipeline (which reads
``motion_ref_pos_b`` / ``motion_ref_ori_b`` from the active motion) gets a
synthetic target and we can measure how far it carries the rollout.

Usage:
    python scripts/eval/unreal_engine/generate_synthetic_stair_motions.py \
        --out_dir src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/synth_stairs_h010-030_w030-080 \
        --num_motions 100 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


# Default standing pose (g1_29dof) — must match
# ``config_values/robot.py:341`` (RobotConfig.init_state.default_joint_angles).
_STANDING_DOF = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
_DEFAULT_PELVIS_HEIGHT = 0.78  # holosoma init_state.pos[2] approximates this


def _build_step_box(x_lo: float, x_hi: float, y_half: float, z_lo: float, z_hi: float):
    """Return (vertices, faces) of an axis-aligned box."""
    v = np.array(
        [
            [x_lo, -y_half, z_lo],
            [x_hi, -y_half, z_lo],
            [x_hi, y_half, z_lo],
            [x_lo, y_half, z_lo],
            [x_lo, -y_half, z_hi],
            [x_hi, -y_half, z_hi],
            [x_hi, y_half, z_hi],
            [x_lo, y_half, z_hi],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # -y face
            [2, 6, 7], [2, 7, 3],  # +y face
            [1, 5, 6], [1, 6, 2],  # +x face
            [3, 7, 4], [3, 4, 0],  # -x face
        ],
        dtype=np.int32,
    )
    return v, f


def _build_stair_mesh(h: float, w: float, n_steps: int, ascending: bool, y_half: float = 1.5,
                      plateau_pre: float = 1.0, plateau_post: float = 1.0):
    """Return (mesh_vertices, mesh_faces, step_top_z_at_x).

    Ascending: stair rises in +x. Descending: stair falls in +x.
    Layout: [pre-plateau at z=0] -> [stair rising/falling] -> [post-plateau at end z].
    """
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    voff = 0

    def _add(v, f):
        nonlocal voff
        verts.append(v)
        faces.append(f + voff)
        voff += v.shape[0]

    total_h = n_steps * h
    end_z_top = total_h if ascending else 0.0

    # Pre-plateau: from x = -plateau_pre to x = 0, z ∈ [0, top_z_pre]
    pre_top = 0.0 if ascending else total_h
    if pre_top > 0:
        v, f = _build_step_box(-plateau_pre, 0.0, y_half, 0.0, pre_top)
        _add(v, f)
    else:
        # ground-level pre-plateau still needs a thin slab so the robot has
        # something to stand on (and visualization shows ground continuity).
        v, f = _build_step_box(-plateau_pre, 0.0, y_half, -0.05, 0.0)
        _add(v, f)

    # Stair: each step i has top at height step_z[i], spanning x ∈ [i*w, (i+1)*w].
    # Step boxes are filled to ground (z=0) so neighbouring steps don't have
    # phantom vertical gaps.
    for i in range(n_steps):
        x_lo = i * w
        x_hi = (i + 1) * w
        if ascending:
            z_top = (i + 1) * h
        else:
            z_top = (n_steps - 1 - i) * h
            # We still want the descending stair to be tall on its high side
            # (low x) and short on the low side (high x). The first step sits
            # at the FULL height; last step is just one step above ground.
            z_top = (n_steps - i) * h
        v, f = _build_step_box(x_lo, x_hi, y_half, 0.0, z_top)
        _add(v, f)

    # Post-plateau: x ∈ [n*w, n*w + plateau_post], z ∈ [0, end_z_top]
    post_top = total_h if ascending else 0.0
    x_lo = n_steps * w
    x_hi = x_lo + plateau_post
    if post_top > 0:
        v, f = _build_step_box(x_lo, x_hi, y_half, 0.0, post_top)
        _add(v, f)
    else:
        v, f = _build_step_box(x_lo, x_hi, y_half, -0.05, 0.0)
        _add(v, f)

    mesh_v = np.concatenate(verts, axis=0).astype(np.float32)
    mesh_f = np.concatenate(faces, axis=0).astype(np.int32)

    def step_z_at_x(x: float) -> float:
        if x < 0:
            return pre_top
        if x >= n_steps * w:
            return post_top
        i = int(x / w)
        return (i + 1) * h if ascending else (n_steps - i) * h

    return mesh_v, mesh_f, step_z_at_x


def _setup_default_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Build (nq,) qpos with root at identity + standing-pose joint angles."""
    qpos = np.zeros(model.nq, dtype=np.float64)
    # Free root: [x, y, z, qw, qx, qy, qz]
    qpos[:3] = [0.0, 0.0, 0.0]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    # Joints — look up by name to be robust to ordering differences.
    for j_name, angle in _STANDING_DOF.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id < 0:
            raise ValueError(f"Joint {j_name!r} not found in model")
        qpos_adr = model.jnt_qposadr[j_id]
        qpos[qpos_adr] = angle
    return qpos


def _measure_pelvis_above_foot(model: mujoco.MjModel, base_qpos: np.ndarray) -> float:
    """FK the robot at base_qpos with root at identity and return the pelvis-z
    minus the lower foot-z, i.e. the height the pelvis sits above the ground
    when both feet are at ground level. This depends on the joint pose: a
    mid-stride walking pose has different pelvis-above-foot than a default
    standing pose. Without this, synth motions teleport the robot's pelvis to
    `_DEFAULT_PELVIS_HEIGHT` regardless of pose, which puts the feet either
    above or below the ground at spawn.
    """
    data = mujoco.MjData(model)
    qpos = base_qpos.copy()
    qpos[0:3] = 0.0
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    foot_candidates = [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_foot", "right_foot",
        "left_ankle_link", "right_ankle_link",
    ]
    foot_zs = []
    for fname in foot_candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fname)
        if bid >= 0:
            foot_zs.append(float(data.xpos[bid, 2]))
    if not foot_zs:
        raise RuntimeError(f"No foot bodies found among {foot_candidates}")
    pelvis_z = float(data.xpos[pelvis_id, 2])
    return pelvis_z - min(foot_zs)


def _generate_motion(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    body_names: list[str],
    joint_names_motion_order: list[str],
    rng: np.random.Generator,
    fps: int = 50,
    lin_vel_x: float = 0.6,
    fixed_h: float | None = None,
    fixed_w: float | None = None,
    fixed_n_steps: int | None = None,
    fixed_ascending: bool | None = None,
    plateau_pre: float = 1.0,
    plateau_post: float = 1.0,
    frame0_joint_dof_vel_by_name: dict[str, float] | None = None,
    pelvis_above_ground: float = _DEFAULT_PELVIS_HEIGHT,
):
    """Sample one stair + walking trajectory and return the .npz payload dict."""
    h = float(rng.uniform(0.10, 0.30)) if fixed_h is None else fixed_h
    w = float(rng.uniform(0.30, 0.80)) if fixed_w is None else fixed_w
    n_steps = int(rng.integers(5, 16)) if fixed_n_steps is None else fixed_n_steps
    ascending = bool(rng.integers(0, 2)) if fixed_ascending is None else fixed_ascending

    mesh_v, mesh_f, step_z_at_x = _build_stair_mesh(
        h, w, n_steps, ascending,
        plateau_pre=plateau_pre, plateau_post=plateau_post,
    )

    # Spend a bit longer on the pre-plateau before crossing onto stair, so a
    # walking-init robot has room to take a few "warm-up" steps. Scale with
    # plateau_pre.
    pre_walk_distance = max(0.5, plateau_pre - 0.5)  # leave a small margin
    pre_pad_s = pre_walk_distance / lin_vel_x
    stair_len = n_steps * w
    cross_s = stair_len / lin_vel_x
    post_pad_s = max(0.5, (plateau_post - 0.5)) / lin_vel_x
    total_s = pre_pad_s + cross_s + post_pad_s
    T = max(int(round(total_s * fps)), 30)
    dt = 1.0 / fps

    data = mujoco.MjData(model)
    body_pos = np.zeros((T, len(body_names), 3), dtype=np.float64)
    body_quat = np.zeros((T, len(body_names), 4), dtype=np.float64)
    joint_pos = np.zeros((T, model.nq), dtype=np.float32)

    # Robot starts in the middle of the pre-plateau so it has room on both
    # sides — useful when comparing standing-init vs walking-init runs (the
    # robot has time to either start walking forward or just stand still
    # before reaching the first step).
    start_x = -plateau_pre / 2.0
    for t in range(T):
        seconds = t * dt
        x = start_x + lin_vel_x * seconds
        z = step_z_at_x(x) + pelvis_above_ground
        qpos = base_qpos.copy()
        qpos[0] = x
        qpos[1] = 0.0
        qpos[2] = z
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # face +x, upright
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos
        body_quat[t] = data.xquat  # mujoco stores wxyz
        joint_pos[t] = qpos

    # Body lin/ang velocities by finite difference. Boundaries use forward /
    # backward diff (NOT zero) — rollout_single_policy.py teleports the robot
    # with body_lin_vel[0] from the motion library, and a zero pelvis velocity
    # at frame 0 puts the student in a "stand still" state it never saw at
    # training time (real motions have non-zero pelvis vel mid-walk).
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    if T >= 2:
        body_lin_vel[0] = (body_pos[1] - body_pos[0]) / dt
        body_lin_vel[-1] = (body_pos[-1] - body_pos[-2]) / dt
    if T >= 3:
        body_lin_vel[1:-1] = (body_pos[2:] - body_pos[:-2]) / (2 * dt)
    # Angular velocity stays ~0 since orientation is upright + constant.

    # joint_vel: nv values per frame. nv layout = [root_lin (3), root_ang (3),
    # 29 hinge joints in mujoco joint order]. Root linear velocity follows
    # from finite diff of joint_pos[:, :3]. Root angular velocity stays 0
    # (identity quat). Hinge joint velocities are 0 except at frame 0 when
    # `frame0_joint_dof_vel_by_name` is supplied (walking-init mode) — there
    # we splice in the real motion's frame-0 dof vel so the teleported robot
    # starts mid-stride in BOTH dof_pos and dof_vel.
    joint_vel = np.zeros((T, model.nv), dtype=np.float32)
    if T >= 2:
        joint_vel[0, :3] = (joint_pos[1, :3] - joint_pos[0, :3]) / dt
        joint_vel[-1, :3] = (joint_pos[-1, :3] - joint_pos[-2, :3]) / dt
    if T >= 3:
        joint_vel[1:-1, :3] = (joint_pos[2:, :3] - joint_pos[:-2, :3]) / (2 * dt)
    if frame0_joint_dof_vel_by_name:
        for j_name, vel in frame0_joint_dof_vel_by_name.items():
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id < 0:
                continue
            dof_adr = model.jnt_dofadr[j_id]
            joint_vel[0, dof_adr] = float(vel)

    return {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel.astype(np.float64),
        "body_ang_vel_w": body_ang_vel.astype(np.float64),
        "joint_names": np.array(joint_names_motion_order, dtype=f"<U{max(len(n) for n in joint_names_motion_order)}"),
        "body_names": np.array(body_names, dtype=f"<U{max(len(n) for n in body_names)}"),
        "mesh_vertices": mesh_v,
        "mesh_faces": mesh_f,
    }, dict(h=h, w=w, n_steps=n_steps, ascending=ascending, T=T, lin_vel_x=lin_vel_x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_motions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mujoco_xml",
        default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml",
        help="Path to the G1 mujoco XML used for FK.",
    )
    parser.add_argument("--lin_vel_x", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=50)
    # Optional fixed-stair params (useful for "regenerate this exact stair").
    parser.add_argument("--fixed_h", type=float, default=None)
    parser.add_argument("--fixed_w", type=float, default=None)
    parser.add_argument("--fixed_n_steps", type=int, default=None)
    parser.add_argument(
        "--fixed_direction", choices=["up", "down"], default=None,
        help="Force ascending or descending; default = random.",
    )
    parser.add_argument(
        "--init_from_real_motion", default=None,
        help="Path to a real motion .npz. When set, the synthetic motion uses that "
        "real motion's frame-0 joint angles (29 dof) as the standing pose for ALL "
        "frames instead of the holosoma default standing pose. The robot then "
        "init's into a walking-style pose at frame 0, which the trained student "
        "is much more familiar with than 'two feet together' default.",
    )
    parser.add_argument(
        "--plateau_pre", type=float, default=1.0,
        help="Length (m) of the flat plateau before the stair. Default 1.0m. "
        "Increase to give a walking-init robot room to take warm-up steps.",
    )
    parser.add_argument(
        "--plateau_post", type=float, default=1.0,
        help="Length (m) of the flat plateau after the stair. Default 1.0m.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    # Joint name list in motion-data order (29 hinge joints, root excluded).
    joint_names_motion = list(_STANDING_DOF.keys())
    base_qpos = _setup_default_qpos(model)
    print(f"[INFO] Mujoco model: {model.nbody} bodies, {model.nq} qpos dofs")
    print(f"[INFO] body_names[:5] = {body_names[:5]}")

    # If a real motion is provided, override base_qpos's joint angles with that
    # motion's frame-0 joint pose AND collect frame-0 dof velocities. The base
    # position / orientation stays identity (we still spawn the synthetic stair
    # at the origin), only the 29 joint angles change.
    frame0_joint_dof_vel_by_name: dict[str, float] = {}
    if args.init_from_real_motion is not None:
        real = np.load(args.init_from_real_motion, allow_pickle=True)
        # joint_pos in motion data: (T, 36) where [:, 7:] = 29 joint angles in
        # the motion's joint_names order.
        real_joint_names = real["joint_names"].tolist()
        real_dof_frame0 = real["joint_pos"][0, 7:]  # (29,) walking pose
        for j_name, angle in zip(real_joint_names, real_dof_frame0):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id < 0:
                continue
            qpos_adr = model.jnt_qposadr[j_id]
            base_qpos[qpos_adr] = float(angle)
        # joint_vel in motion data: (T, nv) with [:, :6] = root (3 lin + 3 ang)
        # and [:, 6:6+29] = 29 hinge joint vels in the same `joint_names`
        # order. Splice frame-0 hinge dof vels into the synth motion so the
        # teleported robot starts with mid-stride dof_vel as well.
        real_joint_vel = real["joint_vel"]
        if real_joint_vel.ndim == 2 and real_joint_vel.shape[1] >= 6 + len(real_joint_names):
            real_jv_frame0 = real_joint_vel[0, 6 : 6 + len(real_joint_names)]
            for j_name, jv in zip(real_joint_names, real_jv_frame0):
                frame0_joint_dof_vel_by_name[j_name] = float(jv)
        print(
            f"[INFO] Init pose: real motion {Path(args.init_from_real_motion).name} frame 0 "
            f"({len(frame0_joint_dof_vel_by_name)} dof vels spliced into frame 0)"
        )

    rng = np.random.default_rng(args.seed)
    fixed_ascending = (
        None if args.fixed_direction is None else (args.fixed_direction == "up")
    )

    # Pelvis height above ground depends on joint pose. With a mid-stride
    # walking-init pose, knees are bent and pelvis sits lower above the feet
    # than a default standing pose. Measuring this from FK keeps the feet on
    # the terrain at spawn — otherwise the robot drops or penetrates ground
    # in the first few physics steps, regardless of velocity init.
    pelvis_above_ground = _measure_pelvis_above_foot(model, base_qpos)
    print(f"[INFO] pelvis_above_foot (from FK on init pose) = {pelvis_above_ground:.4f} m")

    manifest = []
    for i in range(args.num_motions):
        payload, meta = _generate_motion(
            model, base_qpos, body_names, joint_names_motion, rng,
            fps=args.fps, lin_vel_x=args.lin_vel_x,
            fixed_h=args.fixed_h, fixed_w=args.fixed_w,
            fixed_n_steps=args.fixed_n_steps, fixed_ascending=fixed_ascending,
            plateau_pre=args.plateau_pre, plateau_post=args.plateau_post,
            frame0_joint_dof_vel_by_name=frame0_joint_dof_vel_by_name or None,
            pelvis_above_ground=pelvis_above_ground,
        )
        # Filename encodes the random parameters so analysis can recover them
        # without re-loading the .npz.
        direction = "up" if meta["ascending"] else "down"
        name = (
            f"synth_h{int(round(meta['h']*100)):02d}_w{int(round(meta['w']*100)):02d}"
            f"_n{meta['n_steps']:02d}_{direction}_i{i:03d}"
        )
        out_path = out_dir / f"{name}.npz"
        np.savez(out_path, **payload)
        manifest.append({"file": out_path.name, **meta})
        print(f"[INFO] {out_path.name}: h={meta['h']:.3f} w={meta['w']:.3f} "
              f"n={meta['n_steps']} {direction} T={meta['T']}")

    # Also dump a summary JSON for downstream analysis.
    import json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Wrote manifest.json")


if __name__ == "__main__":
    main()
