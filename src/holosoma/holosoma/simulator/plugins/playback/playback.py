"""Kinematic motion-playback plugin: drive the robot (and optionally scene objects) from a clip.

Rendering-only playback. The plugin samples the clip at the current playback time each control
frame (``FRAME_BEGIN``) and writes robot DOF/root state and object poses on every ``PRE_STEP``.
With ``simulator.config.sim.kinematic_playback`` set, each sim step propagates those writes with
forward kinematics instead of dynamics, so the rendered motion is exactly the clip; without it,
the per-substep re-pin bounds dynamics influence to one ``sim_dt``. Pair with a mounted camera
and a frame consumer (``frame-writer``, ``viz-record``, ``ros2-image``) to render recorded joint
angles + object motion.

Clip format: any 2D matrix — ``.npy``, ``.npz`` (one array), ``.csv`` (comma), or ``.txt``
(whitespace). One row per frame; columns left to right are the robot block followed by one
7-column ``[x, y, z, quat]`` block per controlled object (in ``objects`` config order). The robot
block is either ``root_pose(7) + joint_pos(num_dof)`` (floating base) or ``joint_pos(num_dof)``
(fixed base); which one is inferred from the column count. Joint columns are in the sim's DOF
order. Frame rate and quaternion order are plugin config, not in the file.

A ``.npz`` with a ``qpos`` key uses ``qpos [T, 7+nj]`` (root pose wxyz + joints) as the robot
block, and ``object_motion_pos [T, 3]`` +
``object_motion_rot_wxyz [T, 4]`` (when present) form one object block appended after it.

Playlist entries may be single files or a glob pattern, local (``motions/*.npz``) or remote
(``s3://bucket/motions/*.npz``), so a 100-clip run does not have to spell out 100 paths.

Replay is single-trajectory: the same clip state is written to every env, so a vectorized run
renders N identical copies. Per-env clip assignment is out of scope.
"""

from __future__ import annotations

from collections.abc import Iterator
from glob import has_magic
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path, resolve_paths
from holosoma.utils.rotations import quat_slerp

if TYPE_CHECKING:
    from holosoma.config_types.plugin import MotionPlaybackPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

_QUAT = 7  # width of an [x, y, z, quat(4)] pose block
_ROOT = 7  # width of a floating-base root pose block
_CLIP_SUFFIXES = (".npy", ".npz", ".csv", ".txt")  # clip formats a pattern picks up


def _qpos_matrix(data: np.lib.npyio.NpzFile) -> npt.NDArray[np.float32]:
    """Assemble the standard matrix layout from ``qpos`` and an optional object track."""
    qpos = np.asarray(data["qpos"], dtype=np.float32)
    blocks = [qpos]
    if "object_motion_pos" in data.files:
        blocks.append(np.asarray(data["object_motion_pos"], dtype=np.float32))
        blocks.append(np.asarray(data["object_motion_rot_wxyz"], dtype=np.float32))
    return np.concatenate(blocks, axis=1, dtype=np.float32)


def _load_matrix(motion_file: str) -> npt.NDArray[np.float32]:
    """Load a 2D ``[frames, columns]`` float32 matrix from a .npy/.npz/.csv/.txt file."""
    path = resolve_data_file_path(motion_file)
    logger.info(f"MotionPlaybackPlugin loading clip: {path}")
    ext = Path(path).suffix.lower()
    with cached_open(path, "rb") as f:
        if ext == ".npy":
            arr = np.load(f)
        elif ext == ".npz":
            with np.load(f) as data:
                if "qpos" in data.files:
                    arr = _qpos_matrix(data)
                elif len(data.files) == 1:
                    arr = data[data.files[0]]
                else:
                    raise ValueError(
                        f"Motion file '{motion_file}' (.npz) must hold exactly one array or a "
                        f"'qpos' key; found {sorted(data.files)}."
                    )
        elif ext == ".csv":
            arr = np.loadtxt(f, delimiter=",", ndmin=2)
        else:  # .txt or any whitespace-delimited matrix
            arr = np.loadtxt(f, ndmin=2)
    arr = np.asarray(arr, dtype=np.float32)
    assert isinstance(arr, np.ndarray)
    if arr.ndim != 2:
        raise ValueError(f"Motion file '{motion_file}' must be a 2D matrix [frames, columns], got shape {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"Motion file '{motion_file}' has no frames (shape {arr.shape}).")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"Motion file '{motion_file}' contains {bad} non-finite value(s) (NaN/inf).")
    return arr


class PlaybackClip:
    """A motion clip: a ``[frames, columns]`` matrix sliced into the robot block and object poses.

    Tensors are float32 on ``device``; quaternions are stored xyzw (sim convention).
    """

    def __init__(self, motion_file: str, num_dof: int, num_objects: int, fps: float, quat_order: str, device: str):
        mat = _load_matrix(motion_file)
        self.fps = fps
        self.num_frames = mat.shape[0]

        # Robot block is root(7)+joints or just joints; infer from the column count once the
        # object blocks (7 each) are removed.
        object_cols = num_objects * _QUAT
        robot_cols = mat.shape[1] - object_cols
        if robot_cols == _ROOT + num_dof:
            self.has_root = True
        elif robot_cols == num_dof:
            self.has_root = False
        else:
            raise ValueError(
                f"Motion file '{motion_file}' has {mat.shape[1]} columns; expected "
                f"{_ROOT + num_dof + object_cols} (floating base: root 7 + {num_dof} joints + "
                f"{num_objects} objects x 7) or {num_dof + object_cols} (fixed base)."
            )

        def to_t(a: npt.NDArray[np.float32]) -> torch.Tensor:
            return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

        col = 0
        if self.has_root:
            self.root_pos = to_t(mat[:, col : col + 3])
            self.root_quat = self._as_unit_xyzw(to_t(mat[:, col + 3 : col + _ROOT]), quat_order, "root", motion_file)
            col += _ROOT
        self.joint_pos = to_t(mat[:, col : col + num_dof])
        col += num_dof

        # Objects: [frames, K, 3] positions and [frames, K, 4] xyzw quats, in config order.
        obj_pos = torch.zeros(self.num_frames, num_objects, 3, device=device)
        obj_quat = torch.zeros(self.num_frames, num_objects, 4, device=device)
        for k in range(num_objects):
            obj_pos[:, k] = to_t(mat[:, col : col + 3])
            obj_quat[:, k] = self._as_unit_xyzw(
                to_t(mat[:, col + 3 : col + _QUAT]), quat_order, f"object {k}", motion_file
            )
            col += _QUAT
        self.object_pos = obj_pos
        self.object_quat = obj_quat

    @staticmethod
    def _as_unit_xyzw(quat: torch.Tensor, quat_order: str, label: str, motion_file: str) -> torch.Tensor:
        """Return quats as xyzw; input columns are ``quat_order`` (``"xyzw"`` or ``"wxyz"``).

        Rejects non-unit rows: a systematically wrong norm usually means the block boundaries
        are misaligned (wrong ``objects`` count or a robot-block width mismatch).
        """
        norms = quat.norm(dim=-1)
        bad = (norms - 1.0).abs() > 1e-2
        if bool(bad.any()):
            i = int(bad.nonzero()[0, 0])
            raise ValueError(
                f"Motion file '{motion_file}': {label} quaternion at frame {i} is not unit-norm "
                f"(|q|={norms[i]:.4f}). Check the column layout and quat_order."
            )
        return quat if quat_order == "xyzw" else quat[..., [1, 2, 3, 0]]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps

    def sample(self, t: float, interpolate: bool) -> dict[str, torch.Tensor]:
        """Robot + object state at playback time ``t`` seconds, clamped to the clip."""
        f = min(max(t, 0.0) * self.fps, float(self.num_frames - 1))
        i0 = int(f)
        i1 = min(i0 + 1, self.num_frames - 1)
        alpha = f - i0
        if not interpolate or alpha == 0.0 or i0 == i1:
            return self._frame(i1 if alpha >= 0.5 and not interpolate else i0)
        lo, hi = self._frame(i0), self._frame(i1)
        out = {k: torch.lerp(lo[k], hi[k], alpha) for k in lo if not k.endswith("quat")}
        for k in lo:
            if k.endswith("quat"):
                out[k] = quat_slerp(lo[k], hi[k], alpha)
        return out

    def _frame(self, i: int) -> dict[str, torch.Tensor]:
        frame = {"joint_pos": self.joint_pos[i], "object_pos": self.object_pos[i], "object_quat": self.object_quat[i]}
        if self.has_root:
            frame["root_pos"] = self.root_pos[i]
            frame["root_quat"] = self.root_quat[i]
        return frame


class MotionPlaybackPlugin:
    """Pin the sim to a playlist of recorded clips, one episode per clip.

    ``FRAME_BEGIN`` advances the playback clock by one control dt (times ``speed``) and samples
    the current clip; ``PRE_STEP`` writes the sampled robot/object state.

    Each episode is one full clip playthrough (its final frame is always rendered before the
    boundary). Between episodes the plugin rewinds to the next clip's first frame (a hard reset —
    the ``PRE_STEP`` write re-pins robot+objects to frame 0 with zero velocity) and emits
    ``EPISODE_END`` then ``EPISODE_START`` per env, so downstream plugins re-roll per-episode
    settings. The playlist repeats ``n_run`` times in order; after ``len(playlist) * n_run``
    episodes the run shuts down, or forever when ``n_run == 0``. Shutdown is requested from the last
    episode's final ``FRAME_END``, so that frame is egressed exactly once and no tick repeats it; a
    single clip played once therefore ends without emitting a boundary. ``settle_frames`` optionally
    holds each episode at frame 0 for a few frames so per-episode changes materialize (and RTX
    reconverges) before the clip advances.

    Each ``motion_files`` entry creates its :func:`holosoma.utils.path.resolve_paths` iterator at
    construction, but paths are appended to ``playlist`` only as episodes reach them. Remote
    listings can therefore happen during construction while downloads remain lazy. Clip matrix
    parsing and tensor conversion likewise happen only when each episode is reached.
    """

    cfg: MotionPlaybackPluginConfig

    def __init__(self, cfg: MotionPlaybackPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
        if not cfg.motion_files:
            raise ValueError(
                "MotionPlaybackPluginConfig requires a clip: set motion_files (--plugin.<key>.motion_files)."
            )

        entries = [(source, resolve_paths(source)) for source in cfg.motion_files]
        self._pending_paths = self._iter_playlist(entries)
        self._playlist_complete = False
        self.playlist: list[str] = []
        self._playlist_sources: list[str] = []
        self._clips: list[PlaybackClip | None] = []
        sim_cfg = simulator.simulator_config.sim
        self._control_dt = sim_cfg.control_decimation_steps / sim_cfg.fps
        self._t = cfg.start_time
        self._state: dict[str, torch.Tensor] | None = None
        self._done_logged = False
        self._episode_idx = 0
        self._infinite = cfg.n_run == 0
        self._settle_left = 0
        self._final_frame_rendered = False
        self._final_episode = False
        self._lookahead_error: Exception | None = None

        simulator.hooks.add(Phase.FRAME_BEGIN, self.advance, name="motion_playback.advance")
        simulator.hooks.add(Phase.PRE_STEP, self.write_state, name="motion_playback.write_state")
        simulator.hooks.add(Phase.FRAME_END, self.finish_when_egressed, name="motion_playback.finish")

    def _is_playlist_clip(self, path: str) -> bool:
        """Whether a glob result is a clip supported by this playback plugin."""
        local_path = Path(path)
        return (
            local_path.is_file()
            and not local_path.name.startswith(".")
            and local_path.name.lower().endswith(_CLIP_SUFFIXES)
        )

    def _iter_playlist(self, entries: list[tuple[str, Iterator[str]]]) -> Iterator[tuple[str, str]]:
        """Yield resolved local paths, filtering only entries that are syntactic globs."""
        for source, paths in entries:
            is_pattern = has_magic(source)
            seen = accepted = 0
            for path in paths:
                seen += 1
                if not is_pattern or self._is_playlist_clip(path):
                    accepted += 1
                    yield source, path
            if seen == 0:
                raise ValueError(f"Motion playlist entry '{source}' resolved to no paths.")
            if accepted == 0:
                raise ValueError(
                    f"Motion playlist entry '{source}' resolved to no clip files supported by this plugin."
                )
            if is_pattern:
                skipped = f", {seen - accepted} non-clip path(s) skipped" if seen > accepted else ""
                logger.info(f"MotionPlaybackPlugin: playlist entry '{source}' -> {accepted} clip(s){skipped}.")

    def _append_next_playlist_path(self) -> bool:
        """Append one lazily resolved local path, returning whether one was available."""
        if self._playlist_complete:
            return False
        try:
            source, path = next(self._pending_paths)
        except StopIteration:
            self._playlist_complete = True
            return False
        self.playlist.append(path)
        self._playlist_sources.append(source)
        self._clips.append(None)
        return True

    @property
    def clips(self) -> list[PlaybackClip]:
        """Load and return every clip; normal playback loads only the current clip."""
        while self._append_next_playlist_path():
            pass
        return [self._load_clip(index) for index in range(len(self.playlist))]

    def _playlist_index(self, episode_index: int) -> int | None:
        """Return this episode's index, resolving only as far as this episode needs."""
        while episode_index >= len(self.playlist) and self._append_next_playlist_path():
            pass
        if episode_index < len(self.playlist):
            return episode_index
        if not self.playlist:
            raise ValueError("Motion playlist resolved to no clips.")
        if not self._infinite and episode_index >= len(self.playlist) * self.cfg.n_run:
            return None
        return episode_index % len(self.playlist)

    def _load_clip(self, index: int) -> PlaybackClip:
        clip = self._clips[index]
        if clip is None:
            cfg = self.cfg
            local_path = self.playlist[index]
            source = self._playlist_sources[index]
            if source != local_path:
                logger.info(f"MotionPlaybackPlugin: resolved clip {source} -> {local_path}")
            clip = PlaybackClip(
                local_path,
                self.simulator.num_dof,
                len(cfg.objects),
                cfg.fps,
                cfg.quat_order,
                self.simulator.sim_device,
            )
            self._clips[index] = clip
        return clip

    @property
    def clip(self) -> PlaybackClip:
        """The current episode's clip (playlist wraps by episode index)."""
        index = self._playlist_index(self._episode_idx)
        if index is None:
            raise IndexError(f"Motion playback episode {self._episode_idx} is past the end of the playlist.")
        return self._load_clip(index)

    def advance(self) -> None:
        """FRAME_BEGIN: choose the frame to render this tick and advance the clock/episode."""
        # A playlist error held back from the previous frame (see the clip-end branch below) surfaces
        # here, once that frame has been written and egressed and before any episode boundary.
        if self._lookahead_error is not None:
            raise self._lookahead_error

        if self._done_logged:
            self._state = self.clip.sample(self.clip.duration, self.cfg.interpolate)
            return

        # Settle window: hold the current clip's first frame, do not advance the clock, so the
        # extension's EPISODE_START changes materialize (and RTX reconverges) before playback.
        if self._settle_left > 0:
            self._settle_left -= 1
            self._state = self.clip.sample(self.cfg.start_time, self.cfg.interpolate)
            return

        # Within the clip: sample the current time and advance the clock.
        if self._t < self.clip.duration:
            self._state = self.clip.sample(self._t, self.cfg.interpolate)
            self._t += self._control_dt * self.cfg.speed
            return

        # Reached the clip end: render the clip's final frame once (never dropped), then close the
        # episode on the following tick.
        if not self._final_frame_rendered:
            self._final_frame_rendered = True
            self._state = self.clip.sample(self.clip.duration, self.cfg.interpolate)
            # Whether an episode follows decides whether :meth:`finish_when_egressed` ends the run
            # from this frame's FRAME_END, and answering it resolves one playlist entry ahead.
            #
            # That resolution can fail (a glob matching nothing, a remote listing error) while this
            # frame is already chosen, so the failure waits for the next tick and the clip keeps its
            # last frame. The exception is held rather than re-resolved: a generator that raised is
            # closed, so a second attempt reports no further entries and would end the run silently.
            try:
                self._final_episode = self._playlist_index(self._episode_idx + 1) is None
            except Exception as error:
                self._lookahead_error = error
                self._final_episode = False  # do not end the run on a question that went unanswered
            return

        # The final episode ends the run from its FRAME_END, and a loop that honours the shutdown flag
        # stops at the next frame boundary, so reaching here means another frame was opened after it.
        # A driver that never emits FRAME_END lands here with the run still unfinished: end it now, so
        # a finite playlist always terminates even though the frame count is then one too many.
        if self._final_episode:
            self._finish("motion playback finished (driver emitted no FRAME_END)")
            self._state = self.clip.sample(self.clip.duration, self.cfg.interpolate)
            return

        # Episode finished — roll to the next episode.
        self._episode_idx += 1
        self._final_frame_rendered = False
        self._begin_episode()
        if self._settle_left > 0:  # boundary frame is the first of the settle window
            self._settle_left -= 1
            self._state = self.clip.sample(self.cfg.start_time, self.cfg.interpolate)
            return
        # No settle window: render the next clip's first frame now.
        self._state = self.clip.sample(self._t, self.cfg.interpolate)
        self._t += self._control_dt * self.cfg.speed

    def finish_when_egressed(self) -> None:
        """FRAME_END: end a finite run once its final frame has been egressed.

        Asking here rather than while the frame is being selected is what publishes the last frame
        exactly once. The driving loop completes the frame in progress and stops before opening
        another, so no tick repeats that frame and every FRAME_BEGIN keeps its FRAME_END. A driver
        that emits no FRAME_END still terminates, from :meth:`advance` — one frame later.
        """
        if self._final_episode:
            self._finish("motion playback finished")

    def _finish(self, reason: str) -> None:
        """Log the completed episode count once and ask the driving loop to stop."""
        if self._done_logged:
            return
        self._done_logged = True
        logger.info(f"MotionPlaybackPlugin: all {self._episode_idx + 1} episode(s) finished.")
        self.simulator.request_shutdown(reason)

    def _begin_episode(self) -> None:
        """Close the previous episode and open the next: emit boundaries, rewind clock, arm settle."""
        sim = self.simulator
        for env_id in range(sim.num_envs):
            sim.hooks.emit(Phase.EPISODE_END, env_id)
        self._t = self.cfg.start_time
        self._settle_left = self.cfg.settle_frames
        for env_id in range(sim.num_envs):
            sim.hooks.emit(Phase.EPISODE_START, env_id)
        playlist_index = self._playlist_index(self._episode_idx)
        assert playlist_index is not None
        local_path = self.playlist[playlist_index]
        source = self._playlist_sources[playlist_index]
        clip_name = source if source == local_path else f"{source} -> {local_path}"
        if self._infinite:
            last: int | str = "inf"
        elif self._playlist_complete:
            last = len(self.playlist) * self.cfg.n_run - 1
        else:
            last = "?"
        logger.info(
            f"MotionPlaybackPlugin: episode {self._episode_idx}/{last} "
            f"clip={clip_name} settle_frames={self.cfg.settle_frames}"
        )

    def write_state(self) -> None:
        """PRE_STEP: write the sampled robot (and object) state into the simulator."""
        if self._state is None:  # PRE_STEP before the first FRAME_BEGIN (BaseTask emits them together)
            self._state = self.clip.sample(self._t, self.cfg.interpolate)
        state = self._state
        sim = self.simulator
        env_ids = torch.arange(sim.num_envs, device=sim.sim_device)

        sim.dof_pos[:] = state["joint_pos"]
        sim.dof_vel[:] = 0.0
        if self.clip.has_root:
            rs = sim.robot_root_states
            rs[:, :3] = state["root_pos"]
            rs[:, 3:7] = state["root_quat"]
            rs[:, 7:13] = 0.0
            sim.set_actor_root_state_tensor_robots(env_ids, rs)
        sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)  # type: ignore[attr-defined]

        if self.cfg.objects:
            # set_actor_states rows are name-major, env-minor: all envs of object 0, then object 1, ...
            obj = torch.zeros(len(self.cfg.objects) * sim.num_envs, 13, device=sim.sim_device)
            for k in range(len(self.cfg.objects)):
                rows = obj[k * sim.num_envs : (k + 1) * sim.num_envs]
                rows[:, :3] = state["object_pos"][k]
                rows[:, 3:7] = state["object_quat"][k]
            sim.set_actor_states(list(self.cfg.objects), env_ids, obj)
