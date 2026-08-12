# viser_utils.py
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import viser  # type: ignore[import-not-found]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


class CameraFollowController:
    """Translate connected Viser cameras with a moving world-space target."""

    def __init__(self, server: viser.ViserServer, *, initial_enabled: bool = False) -> None:
        self.server = server
        self.checkbox = server.gui.add_checkbox(
            "Automatically follow subjects",
            initial_value=initial_enabled,
        )
        self._target: np.ndarray | None = None
        self._lock = threading.Lock()

        @self.checkbox.on_update
        def _(_event) -> None:
            with self._lock:
                target = None if self._target is None else self._target.copy()
            if self.checkbox.value and target is not None:
                self._move_cameras(target)

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            with self._lock:
                target = None if self._target is None else self._target.copy()
            if self.checkbox.value and target is not None:
                self._move_camera(client, target)

    def update_target(self, target: np.ndarray) -> None:
        """Store the latest target and update all cameras when following is active."""

        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,) or not np.isfinite(target_array).all():
            raise ValueError("Camera follow target must contain three finite xyz values")
        with self._lock:
            self._target = target_array.copy()
        if self.checkbox.value:
            self._move_cameras(target_array)

    def _move_cameras(self, target: np.ndarray) -> None:
        clients = tuple(self.server.get_clients().values())
        if not clients:
            return
        with self.server.atomic():
            for client in clients:
                self._move_camera(client, target)

    @staticmethod
    def _move_camera(client: viser.ClientHandle, target: np.ndarray) -> None:
        camera = client.camera
        translation = target - np.asarray(camera.look_at, dtype=float)
        camera.position = np.asarray(camera.position, dtype=float) + translation
        camera.look_at = target


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    return q if n == 0.0 else q / n


def quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
    q = quat_normalize(curr_q)
    if prev_q is None:
        return q
    return -q if float(np.dot(prev_q, q)) < 0.0 else q


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Interpolate scalar-first quaternions along the shortest arc."""

    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + float(u) * (q1 - q0))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    return (
        np.sin((1.0 - float(u)) * theta) * q0 + np.sin(float(u) * theta) * q1
    ) / sin_theta


def interpolation_window(times_s: np.ndarray, time_s: float) -> tuple[int, int, float]:
    """Return bracketing sample indices and interpolation weight for a timestamp."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("Cannot sample an empty timeline.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Timeline timestamps must be strictly increasing.")
    clamped = float(np.clip(time_s, times[0], times[-1]))
    upper = int(np.searchsorted(times, clamped, side="right"))
    if upper <= 0:
        return 0, 0, 0.0
    if upper >= times.size:
        last = int(times.size - 1)
        return last, last, 0.0
    lower = upper - 1
    weight = (clamped - float(times[lower])) / float(times[upper] - times[lower])
    return lower, upper, float(weight)


def resolve_frame_times(
    n_frames: int,
    *,
    initial_fps: float,
    frame_times_s: np.ndarray | None = None,
) -> np.ndarray:
    """Return elapsed seconds for every source frame in a playback sequence."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    if frame_times_s is None:
        if initial_fps <= 0.0:
            raise ValueError("initial_fps must be positive when frame timestamps are unavailable.")
        return np.arange(n_frames, dtype=float) / float(initial_fps)
    times = np.asarray(frame_times_s, dtype=float).reshape(-1)
    if times.shape[0] != n_frames:
        raise ValueError(f"Expected {n_frames} frame timestamps, got {times.shape[0]}.")
    times = times - times[0]
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Frame timestamps must be strictly increasing.")
    return times


def sample_qpos_at_time(
    qpos: np.ndarray,
    time_s: float,
    *,
    fps: float,
    robot_dof: int,
    has_object_input: bool,
) -> np.ndarray:
    """Sample a qpos sequence with linear values and quaternion SLERP."""

    values = np.asarray(qpos, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("qpos must have shape [frames, values] with at least one frame.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    times = np.arange(values.shape[0], dtype=float) / float(fps)
    lower, upper, weight = interpolation_window(times, time_s)
    if lower == upper:
        return values[lower].copy()
    result = (1.0 - weight) * values[lower] + weight * values[upper]
    result[3:7] = quat_slerp(values[lower, 3:7], values[upper, 3:7], weight)
    if has_object_input and values.shape[1] >= 7 + robot_dof + 7:
        result[-4:] = quat_slerp(values[lower, -4:], values[upper, -4:], weight)
    return result


@dataclass
class QposViserApplier:
    """Apply qpos frames to a Viser robot and optional object."""

    viser_robot: ViserUrdf
    robot_base_frame: viser.FrameHandle
    robot_dof: int
    viser_object: ViserUrdf | None = None
    object_base_frame: viser.FrameHandle | None = None
    contains_object_in_qpos: bool = True
    _prev_robot_q: np.ndarray | None = field(default=None, init=False)
    _prev_obj_q: np.ndarray | None = field(default=None, init=False)

    def has_object_input(self, qpos: np.ndarray) -> bool:
        return (
            self.viser_object is not None
            and self.object_base_frame is not None
            and self.contains_object_in_qpos
            and qpos.shape[1] >= (7 + self.robot_dof + 7)
        )

    def reset_quat_continuity(self) -> None:
        self._prev_robot_q = None
        self._prev_obj_q = None

    def apply_qpos(self, q: np.ndarray, *, has_object_input: bool) -> None:
        joints = q[7 : 7 + self.robot_dof]
        if joints.shape[0] != self.robot_dof:
            joints = (
                joints[: self.robot_dof]
                if joints.shape[0] > self.robot_dof
                else np.pad(joints, (0, self.robot_dof - joints.shape[0]))
            )
        self.viser_robot.update_cfg(joints)

        self.robot_base_frame.position = q[0:3]
        robot_q = quat_continuous(self._prev_robot_q, q[3:7])
        self._prev_robot_q = robot_q
        self.robot_base_frame.wxyz = robot_q

        if has_object_input and self.object_base_frame is not None:
            self.object_base_frame.position = q[-7:-4]
            obj_q = quat_continuous(self._prev_obj_q, q[-4:])
            self._prev_obj_q = obj_q
            self.object_base_frame.wxyz = obj_q
        elif self.object_base_frame is not None and self.viser_object is not None:
            self.object_base_frame.position = np.zeros(3)
            self.object_base_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    def apply_frame(self, qpos: np.ndarray, frame_idx: int) -> None:
        i = int(np.clip(frame_idx, 0, int(qpos.shape[0]) - 1))
        self.apply_qpos(qpos[i], has_object_input=self.has_object_input(qpos))


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
    initial_playback_speed: float = 1.0,
    loop: bool = True,
    frame_times_s: np.ndarray | None = None,
    on_pose_applied: Callable[[np.ndarray], None] | None = None,
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
        initial_playback_speed: playback speed relative to real time.
        loop: whether to wrap around at the end.
        frame_times_s: optional source timestamps for the elapsed-time readout.
        on_pose_applied: optional callback receiving each displayed qpos pose.

    Returns:
        (controls, initial_values) — currently returns the [frame_slider] and [0.0]
    """
    qpos = motion_sequence
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise ValueError("motion_sequence is empty.")
    source_frame_times = resolve_frame_times(
        n_frames,
        initial_fps=float(initial_fps),
        frame_times_s=frame_times_s,
    )

    has_object_input = (
        viser_object is not None
        and object_base_frame is not None
        and contains_object_in_qpos
        and qpos.shape[1] >= (7 + robot_dof + 7)
    )

    # ---------------- GUI ----------------
    with server.gui.add_folder("Playback", order=0.0):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        time_readout = server.gui.add_number(
            "Elapsed time (s)",
            initial_value=0.0,
            min=0.0,
            step=0.001,
            disabled=True,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(initial_fps), min=1, max=240, step=1)
        playback_speed_in = server.gui.add_slider(
            "Playback speed (x real-time)",
            min=0.1,
            max=2.0,
            step=0.1,
            initial_value=float(initial_playback_speed),
        )
    with server.gui.add_folder("Smoothing", order=10.0):
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

        if on_pose_applied is not None:
            on_pose_applied(q)

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

    @playback_speed_in.on_update
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
            time_readout.value = float(source_frame_times[frame_val])
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
                playback_speed = max(0.1, float(playback_speed_in.value))
                dt = 1.0 / (fps_val * mult * playback_speed)

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
                    time_readout.value = float(source_frame_times[k0])
                    updating_programmatically["flag"] = False

                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    # initial draw
    _apply_discrete_frame(0)
    time_readout.value = float(source_frame_times[0])

    # keep consistent with your previous return convention
    return [frame_slider], [0.0]


def create_timed_motion_control_sliders(
    server: viser.ViserServer,
    frame_times_s: np.ndarray,
    apply_time: Callable[[float], None],
    *,
    initial_fps: float = 60.0,
    initial_interp_mult: int = 1,
    initial_playback_speed: float = 1.0,
    loop: bool = True,
) -> Tuple[List[viser.GuiInputHandle[int]], List[float]]:
    """Create timestamp-driven playback controls for one or more scene actors.

    ``apply_time`` receives seconds relative to the first master timestamp. The
    Render rate controls how often it is called, while playback speed scales
    elapsed real time.
    """

    raw_frame_times = np.asarray(frame_times_s, dtype=float).reshape(-1)
    if raw_frame_times.size == 0:
        raise ValueError("frame_times_s is empty.")
    frame_times = resolve_frame_times(
        raw_frame_times.size,
        initial_fps=initial_fps,
        frame_times_s=raw_frame_times,
    )
    duration_s = float(frame_times[-1])

    with server.gui.add_folder("Playback", order=0.0):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(frame_times.size) - 1),
            step=1,
            initial_value=0,
        )
        time_readout = server.gui.add_number(
            "Elapsed time (s)",
            initial_value=0.0,
            min=0.0,
            step=0.001,
            disabled=True,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number(
            "Render FPS",
            initial_value=max(1, round(initial_fps)),
            min=1,
            max=240,
            step=1,
        )
        playback_speed_in = server.gui.add_slider(
            "Playback speed (x real-time)",
            min=0.1,
            max=2.0,
            step=0.1,
            initial_value=float(initial_playback_speed),
        )
    with server.gui.add_folder("Smoothing", order=10.0):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier",
            initial_value=max(1, int(initial_interp_mult)),
            min=1,
            max=8,
            step=1,
        )

    playing = {"flag": False}
    cursor = {"time_s": 0.0}
    clock = {"last_wall_s": time.perf_counter(), "next_draw_s": time.perf_counter()}
    updating_programmatically = {"flag": False}

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        now = time.perf_counter()
        clock["last_wall_s"] = now
        clock["next_draw_s"] = now

    @fps_in.on_update
    def _(_evt) -> None:
        clock["next_draw_s"] = time.perf_counter()

    @playback_speed_in.on_update
    def _(_evt) -> None:
        now = time.perf_counter()
        clock["last_wall_s"] = now
        clock["next_draw_s"] = now

    @interp_mult_in.on_update
    def _(_evt) -> None:
        clock["next_draw_s"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_programmatically["flag"]:
            return
        playing["flag"] = False
        frame_index = int(np.clip(frame_slider.value, 0, frame_times.size - 1))
        cursor["time_s"] = float(frame_times[frame_index])
        time_readout.value = cursor["time_s"]
        apply_time(cursor["time_s"])

    def _player_loop() -> None:
        if frame_times.size <= 1:
            return
        while True:
            if not playing["flag"]:
                time.sleep(0.02)
                clock["last_wall_s"] = time.perf_counter()
                continue

            now = time.perf_counter()
            if now < clock["next_draw_s"]:
                time.sleep(min(0.002, max(0.0, clock["next_draw_s"] - now)))
                continue

            elapsed_s = max(0.0, now - clock["last_wall_s"])
            clock["last_wall_s"] = now
            playback_speed = max(0.1, float(playback_speed_in.value))
            next_time_s = cursor["time_s"] + elapsed_s * playback_speed
            if next_time_s >= duration_s:
                if loop and duration_s > 0.0:
                    next_time_s %= duration_s
                else:
                    next_time_s = duration_s
                    playing["flag"] = False
            cursor["time_s"] = next_time_s
            time_readout.value = next_time_s
            apply_time(next_time_s)

            frame_index = int(np.searchsorted(frame_times, next_time_s, side="right") - 1)
            frame_index = int(np.clip(frame_index, 0, frame_times.size - 1))
            updating_programmatically["flag"] = True
            frame_slider.value = frame_index
            updating_programmatically["flag"] = False

            render_fps = max(1, int(fps_in.value)) * max(1, int(interp_mult_in.value))
            clock["next_draw_s"] = now + 1.0 / float(render_fps)

    threading.Thread(target=_player_loop, daemon=True).start()
    apply_time(0.0)
    time_readout.value = 0.0
    return [frame_slider], [0.0]
