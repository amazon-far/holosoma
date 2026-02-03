#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    # Path to the npz you saved from MuJoCo (with joint_pos, body_pos_w, body_lin_vel_w, etc.)
    npz_path: str

    # Robot URDF used for visualization
    robot_urdf: str

    # Visualization settings
    grid_width: float = 2.0
    grid_height: float = 2.0
    show_meshes: bool = True
    loop: bool = True

    # Playback / visualization
    fps_override: float | None = None  # if None, use fps from npz
    vel_scale: float = 0.1  # length scale for velocity arrows
    vel_min_norm: float = 1e-2  # threshold below which we hide arrows


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_npz_motion(npz_path: str):
    """
    Expected npz format:
        joint_pos      (T, 7 + ndof)  # [root_xyz(3), root_quat(4), ndof]
        joint_vel      (T, 6 + ndof)  # [root_lin(3), root_ang(3), ndof]
        body_pos_w     (T, nbody, 3)
        body_quat_w    (T, nbody, 4)
        body_lin_vel_w (T, nbody, 3)
        body_ang_vel_w (T, nbody, 3)
        joint_names    (ndof,)        # robot joints only, no free root
        body_names     (nbody,)
        fps            [fps]
        (optionally) object_* fields ignored here
    """
    data = np.load(npz_path, allow_pickle=True)

    joint_pos = data["joint_pos"]  # (T, 7 + ndof)
    joint_vel = data["joint_vel"]  # (T, 6 + ndof)  # not used here
    body_pos_w = data["body_pos_w"]  # (T, nbody, 3)
    body_quat_w = data["body_quat_w"]  # (T, nbody, 4)  # not used here
    body_lin_vel_w = data["body_lin_vel_w"]  # (T, nbody, 3)
    body_ang_vel_w = data["body_ang_vel_w"]  # (T, nbody, 3)  # not used here

    joint_names = data["joint_names"]  # (ndof,)
    body_names = data["body_names"]  # (nbody,)

    # fps saved as [fps]
    if "fps" in data:
        fps_arr = np.array(data["fps"]).reshape(-1)
        fps = float(fps_arr[0])
    else:
        fps = 30.0

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": joint_names,
        "body_names": body_names,
        "fps": fps,
    }


# ---------------------------------------------------------------------
# Main visualization logic
# ---------------------------------------------------------------------
def main(cfg: Config) -> None:
    data = load_npz_motion(cfg.npz_path)

    joint_pos = data["joint_pos"]  # (T, 7 + ndof)
    joint_names = list(data["joint_names"])  # names for ndof robot joints
    body_pos_w = data["body_pos_w"]  # (T, nbody, 3)
    body_lin_vel_w = data["body_lin_vel_w"]  # (T, nbody, 3)
    fps_npz = data["fps"]

    T, nq_total = joint_pos.shape
    _, nbody, _ = body_pos_w.shape

    # Split joint_pos into root + joints.
    # Layout: [0:3] root pos, [3:7] root quat (wxyz), [7:] robot joints
    root_pos_seq = joint_pos[:, 0:3]  # (T, 3)
    root_quat_seq = joint_pos[:, 3:7]  # (T, 4)
    joint_angles_seq = joint_pos[:, 7:]  # (T, ndof)
    ndof = joint_angles_seq.shape[1]

    print(f"[viser_body_vel_player] Loaded npz: {cfg.npz_path}")
    print(f"  frames: {T}, total joint_pos dim: {nq_total} (root 7 + ndof {ndof})")
    print(f"  bodies: {nbody}, fps (npz): {fps_npz}")
    # print(f"  joint names (npz): {joint_names}")
    # print(f"  body names (npz):  {body_names}")

    fps = cfg.fps_override if cfg.fps_override is not None else fps_npz
    print(f"  using fps: {fps}")

    # -------------------- Setup viser -------------------------
    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        width=cfg.grid_width,
        height=cfg.grid_height,
        position=(0.0, 0.0, 0.0),
    )

    # Root frame for the robot (this will follow root_pos / root_quat)
    robot_root = server.scene.add_frame("/robot", show_axes=False)

    # Load URDF (via yourdfpy so meshes are available)
    robot_urdf_y = yourdfpy.URDF.load(
        cfg.robot_urdf,
        load_meshes=True,
        build_scene_graph=True,
    )
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    # Actuated joints & mapping between URDF order and npz joint layout
    joint_limits = vr.get_actuated_joint_limits()  # dict: name -> (lower, upper)
    urdf_joint_order = list(joint_limits.keys())
    robot_dof = len(urdf_joint_order)

    print(f"  URDF actuated joints ({robot_dof}): {urdf_joint_order}")

    if robot_dof != ndof:
        print(
            f"[WARN] URDF actuated joint count ({robot_dof}) != ndof in npz joint_pos ({ndof}). "
            "We will map by joint name. If names don't match, this will error."
        )

    # joint_names from npz correspond to the *robot joints only*, in MuJoCo order,
    # which matches joint_pos columns [7: 7 + ndof].
    #
    # Build: URDF joint -> column index in joint_pos
    name_to_npz_joint_idx = {name: i for i, name in enumerate(joint_names)}
    urdf_to_jointpos_cols_list: list[int] = []
    for jname in urdf_joint_order:
        if jname not in name_to_npz_joint_idx:
            raise KeyError(f"URDF joint '{jname}' not found in npz joint_names. npz joint_names: {joint_names}")
        idx_npz = name_to_npz_joint_idx[jname]  # index in [0..ndof-1] for joint_angles_seq
        col_in_joint_pos = 7 + idx_npz  # shift by 7 to get into joint_pos columns
        urdf_to_jointpos_cols_list.append(col_in_joint_pos)
    urdf_to_jointpos_cols = np.array(urdf_to_jointpos_cols_list, dtype=int)

    # Initial URDF configuration & base pose
    root_pos0 = root_pos_seq[0]
    root_quat0 = root_quat_seq[0]

    robot_root.position = root_pos0
    robot_root.wxyz = root_quat0

    initial_cfg = joint_pos[0, urdf_to_jointpos_cols]
    vr.update_cfg(initial_cfg)

    # -------------------- GUI controls ------------------------
    with server.gui.add_folder("Playback"):
        playing_cb = server.gui.add_checkbox("Playing", initial_value=True)
        t_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=T - 1,
            step=1,
            initial_value=0,
        )

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)
        vel_scale_slider = server.gui.add_slider(
            "Velocity scale",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=cfg.vel_scale,
        )

    @show_meshes_cb.on_update
    def _on_meshes_update(_event) -> None:
        vr.show_visual = bool(show_meshes_cb.value)

    # -------------------- Body COM positions ------------------
    # Visualize body COM positions as a small point cloud
    init_body_points = body_pos_w[0]  # (nbody, 3)
    body_colors = np.zeros((nbody, 3), dtype=np.float32)
    body_colors[:] = np.array([0, 255, 0], dtype=np.float32)  # green

    body_points_handle = server.scene.add_point_cloud(
        "/body_com",
        points=init_body_points,
        colors=body_colors,
        point_size=0.015,
        point_shape="circle",
    )

    # -------------------- Velocity line segments --------------
    # We draw velocity as line segments from body_pos_w to body_pos_w + v * scale
    init_vel_points = np.zeros((nbody, 2, 3), dtype=np.float32)
    vel_colors = np.zeros((nbody, 2, 3), dtype=np.float32)
    vel_colors[..., :] = np.array([255, 0, 0], dtype=np.float32)  # red

    vel_lines = server.scene.add_line_segments(
        "/body_velocity_world",
        points=init_vel_points,
        colors=vel_colors,
        line_width=3.0,
    )

    # -------------------- Frame update ------------------------
    def update_frame(frame_idx: int) -> None:
        idx = int(np.clip(frame_idx, 0, T - 1))

        # 1) Update robot root pose
        root_pos = root_pos_seq[idx]  # (3,)
        root_quat = root_quat_seq[idx]  # (4,) wxyz

        robot_root.position = root_pos
        robot_root.wxyz = root_quat

        # 2) Update URDF joint configuration
        cfg_vec = joint_pos[idx, urdf_to_jointpos_cols]  # (robot_dof,)
        vr.update_cfg(cfg_vec)

        # 3) Update body COM points
        body_points_handle.points = body_pos_w[idx]

        # 4) Update velocity line segments
        pos = body_pos_w[idx]  # (nbody, 3)
        vel = body_lin_vel_w[idx] * float(vel_scale_slider.value)  # (nbody, 3)

        # vel_raw = body_lin_vel_w[idx]  # (nbody, 3)
        # norms = np.linalg.norm(vel_raw, axis=-1, keepdims=True)  # (nbody, 1)
        # norms_xy = np.linalg.norm(vel_raw[:, :2], axis=-1, keepdims=True)  # (nbody, 1)
        # eps = 1e-8

        # # Unit directions; zero out near-zero velocities to avoid NaNs
        # dirs = np.where(norms > eps, vel_raw / norms, 0.0)

        # Now every non-zero velocity has the same length = vel_scale_slider.value
        # vel = dirs * float(vel_scale_slider.value)  # (nbody, 3)

        # Hide small velocities (optional)
        # mask = norms_xy < cfg.vel_min_norm
        # vel = np.where(mask, 0.0, vel)

        pts = np.stack([pos, pos + vel], axis=1)  # (nbody, 2, 3)
        vel_lines.points = pts

    @t_slider.on_update
    def _on_slider_update(_event) -> None:
        update_frame(t_slider.value)

    # Initialize frame 0
    update_frame(0)

    # -------------------- Playback loop -----------------------
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_time = time.time()

    print(
        f"[viser_body_vel_player] Ready. Open the URL above to view. "
        f"{'Looping' if cfg.loop else 'One-shot'} playback at {fps:.2f} FPS."
    )

    while True:
        now = time.time()
        if playing_cb.value and (now - last_time) >= dt:
            last_time = now
            next_idx = int(t_slider.value) + 1
            if next_idx >= T:
                if cfg.loop:
                    next_idx = 0
                else:
                    next_idx = T - 1
                    playing_cb.value = False
            t_slider.value = next_idx  # triggers update_frame via callback

        time.sleep(0.002)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)

"""
python viser_body_vel_player.py \
--npz_path ../converted_res/robot_only/sub3_largebox_003_mj.npz \
--robot_urdf ../models/g1/g1_29dof.urdf
"""
