# viser_utils.py
from __future__ import annotations

import threading
import time
from typing import List, Tuple

import numpy as np
import viser  # type: ignore[import-not-found]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


def create_motion_control_sliders(
    server: viser.ViserServer,
    viser_robot: ViserUrdf,
    robot_base_frame: viser.FrameHandle,
    motion_sequence: np.ndarray,
    *,
    robot_dof: int,
    viser_object: ViserUrdf | None = None,
    object_base_frame: viser.FrameHandle | None = None,
    contains_object_in_qpos: bool = True,
    initial_fps: int = 30,
    initial_interp_mult: int = 2,
    loop: bool = True,
) -> Tuple[List[viser.GuiInputHandle[int]], List[float]]:
    """
    Create a slider + play/pause controls and a background player thread with smooth, slerp-based interpolation.

    Assumed qpos layout per frame (MuJoCo order):
        [0:3]   robot base position   (xyz)
        [3:7]   robot base quaternion (wxyz)
        [7:7+R] robot joints          (R = robot_dof)
        [-7:-4] object position  (xyz)            # only if contains_object_in_qpos and viser_object provided
        [-4:]   object quaternion (wxyz)          # only if contains_object_in_qpos and viser_object provided

    Args:
        server: Viser server.
        viser_robot: ViserUrdf for the robot.
        robot_base_frame: server.scene.add_frame(...) return for the robot root frame (we set wxyz/position here).
        motion_sequence: np.ndarray with shape [T, D], sequence of qpos frames.
        robot_dof: number of actuated joints expected by viser_robot.
        viser_object: optional ViserUrdf for an object.
        object_base_frame: optional frame handle for the object root.
        contains_object_in_qpos: set True if motion_sequence includes the object 7D pose at the end.
        initial_fps: base FPS for playback.
        initial_interp_mult: visual upsampling multiplier.
        loop: whether to wrap around at the end.

    Returns:
        (controls, initial_values) — currently returns the [frame_slider] and [0.0]
    """
    qpos = motion_sequence
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise ValueError("motion_sequence is empty.")

    has_object_input = (
        viser_object is not None
        and object_base_frame is not None
        and contains_object_in_qpos
        and qpos.shape[1] >= (7 + robot_dof + 7)
    )

    # ---------------- GUI ----------------
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(initial_fps), min=1, max=240, step=1)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=int(initial_interp_mult), min=1, max=8, step=1
        )

    # ---------------- helpers ----------------
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        n = float(np.linalg.norm(q))
        return q if n == 0.0 else q / n

    def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
        q = _quat_normalize(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        q0 = _quat_normalize(q0)
        q1 = _quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return _quat_normalize(q)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s

    def _interp_frame(qpos_arr: np.ndarray, i0: int, i1: int, u: float) -> np.ndarray:
        """SLERP for base & (optional) object quats; linear for positions and joints."""
        q0 = qpos_arr[i0]
        q1 = qpos_arr[i1]
        out = q0.copy()

        # Robot base (MuJoCo order: pos first, then quat)
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]  # pos (xyz)
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)  # quat (wxyz)

        # Joints
        j0 = q0[7 : 7 + robot_dof]
        j1 = q1[7 : 7 + robot_dof]
        out[7 : 7 + robot_dof] = (1.0 - u) * j0 + u * j1

        # Object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input:
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]  # obj pos (xyz)
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)  # obj quat (wxyz)
        return out

    # ---------------- state ----------------
    playing = {"flag": False}
    tick = {"next": time.perf_counter()}  # absolute time for next draw
    prev: dict[str, np.ndarray | None] = {"robot_q": None, "obj_q": None}  # for continuity
    nonlocal_f = {"f": float(frame_slider.value)}  # fractional frame cursor
    updating_programmatically = {"flag": False}  # flag to prevent callback from pausing during programmatic updates

    # ---------------- draw ----------------
    def _apply_frame_from_q(q: np.ndarray) -> None:
        # joints -> ensure length
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            joints = (
                joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
            )
        viser_robot.update_cfg(joints)

        # robot base (MuJoCo order: pos first, then quat)
        robot_base_frame.position = q[0:3]  # pos (xyz)
        r_q = _quat_continuous(prev["robot_q"], q[3:7])
        prev["robot_q"] = r_q
        robot_base_frame.wxyz = r_q

        # object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input and object_base_frame is not None:
            object_base_frame.position = q[-7:-4]  # obj pos (xyz)
            o_q = _quat_continuous(prev["obj_q"], q[-4:])
            prev["obj_q"] = o_q
            object_base_frame.wxyz = o_q
        elif object_base_frame is not None and viser_object is not None:
            # fallback static pose
            object_base_frame.position = np.zeros(3)
            object_base_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    def _apply_discrete_frame(i: int) -> None:
        i = int(np.clip(i, 0, n_frames - 1))
        _apply_frame_from_q(qpos[i])

    # ---------------- controls ----------------
    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        # reset timing & continuity starting from the current slider frame
        tick["next"] = time.perf_counter()
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_slider.value)

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt) -> None:
        # Only pause if this is a user interaction, not a programmatic update
        if not updating_programmatically["flag"]:
            # Pause when scrubbing so the background loop doesn't overwrite immediately
            playing["flag"] = False
            tick["next"] = time.perf_counter()
            frame_val = int(frame_slider.value)
            _apply_discrete_frame(frame_val)
            prev["robot_q"] = None
            prev["obj_q"] = None
            nonlocal_f["f"] = float(frame_val)

    # ---------------- player loop ----------------
    def _player_loop() -> None:
        if n_frames <= 1:
            return
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)

                if now >= tick["next"]:
                    # advance by one visual step
                    f = nonlocal_f["f"] + 1.0 / mult
                    if loop:
                        f = f % max(1, n_frames)
                    else:
                        f = min(f, float(n_frames - 1))
                    nonlocal_f["f"] = f

                    k0 = int(np.floor(f))
                    k1 = (k0 + 1) % max(1, n_frames) if loop else min(k0 + 1, n_frames - 1)
                    u = float(f - k0)

                    q_interp = _interp_frame(qpos, k0, k1, u)
                    _apply_frame_from_q(q_interp)

                    # Update slider to show current frame number in real-time
                    # Use flag to prevent callback from pausing playback
                    updating_programmatically["flag"] = True
                    frame_slider.value = k0
                    updating_programmatically["flag"] = False

                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    # initial draw
    _apply_discrete_frame(0)

    # keep consistent with your previous return convention
    return [frame_slider], [0.0]
