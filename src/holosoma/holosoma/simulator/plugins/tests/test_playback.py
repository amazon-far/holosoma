"""Unit tests for the motion-playback plugin (pure, no simulator backend).

Covers: matrix loading across formats (.npy/.npz/.csv/.txt), robot-block width inference
(floating vs fixed base), object-block slicing, quat-order handling, time sampling
(clamping/interpolation/nearest-frame), and the plugin's hook behavior against a fake simulator
(state written on PRE_STEP, playback clock advanced on FRAME_BEGIN, episode sequencing and
shutdown, object routing).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from holosoma.config_types.plugin import MotionPlaybackPluginConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.playback import MotionPlaybackPlugin, PlaybackClip
from holosoma.simulator.plugins.playback import playback as playback_module

pytestmark = pytest.mark.no_sim

NUM_DOF = 2


def _make(cfg: MotionPlaybackPluginConfig, sim: Any) -> MotionPlaybackPlugin:
    # Construct via get_cls() (untyped), as the other plugin tests do with their fake simulators.
    plugin: MotionPlaybackPlugin = cfg.get_cls()(cfg, sim)
    return plugin


def _resolve_playlist(plugin: MotionPlaybackPlugin) -> list[str]:
    while plugin._append_next_playlist_path():
        pass
    return plugin.playlist


def _matrix(frames: int = 5, num_objects: int = 0, fixed_base: bool = False) -> npt.NDArray[np.float32]:
    """A clip matrix: root walks +x 1m/frame, joints ramp, each object walks +x at obj-index speed."""
    t = np.arange(frames, dtype=np.float32)
    cols = []
    if not fixed_base:
        root = np.zeros((frames, 7), dtype=np.float32)
        root[:, 0] = t  # root x
        root[:, 3] = 1.0  # identity quat wxyz (w first)
        cols.append(root)
    joints = np.stack([10 + t, 20 + t], axis=1).astype(np.float32)  # dof0, dof1
    cols.append(joints)
    for k in range(num_objects):
        obj = np.zeros((frames, 7), dtype=np.float32)
        obj[:, 0] = (k + 1) * t  # object k walks +x
        obj[:, 3] = 1.0  # identity quat wxyz
        cols.append(obj)
    return np.concatenate(cols, axis=1)


def _write(path: Path, name: str, mat: npt.NDArray[np.float32]) -> str:
    ext = name.rsplit(".", 1)[1]
    f = path / name
    if ext == "npy":
        np.save(f, mat)
    elif ext == "npz":
        np.savez(f, mat)
    elif ext == "csv":
        np.savetxt(f, mat, delimiter=",")
    else:
        np.savetxt(f, mat)
    return str(f)


# ----- PlaybackClip: loading + slicing -----


@pytest.mark.parametrize("name", ["clip.npy", "clip.npz", "clip.csv", "clip.txt"])
def test_load_all_matrix_formats(tmp_path: Path, name: str) -> None:
    clip = PlaybackClip(_write(tmp_path, name, _matrix()), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")
    assert clip.num_frames == 5
    assert clip.has_root
    assert clip.joint_pos[0].tolist() == pytest.approx([10.0, 20.0])
    assert clip.root_pos[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert clip.root_quat[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # wxyz -> xyzw


def test_playback_clip_resolves_source_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_path = _write(tmp_path, "clip.npy", _matrix())
    calls: list[str] = []

    def resolve_once(value: str) -> str:
        calls.append(value)
        return local_path

    monkeypatch.setattr(playback_module, "resolve_data_file_path", resolve_once)

    PlaybackClip(
        "@extension/clip.npy",
        NUM_DOF,
        0,
        fps=10.0,
        quat_order="wxyz",
        device="cpu",
    )

    assert calls == ["@extension/clip.npy"]


def test_fixed_base_matrix_has_no_root(tmp_path: Path) -> None:
    clip = PlaybackClip(
        _write(tmp_path, "c.npy", _matrix(fixed_base=True)), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu"
    )
    assert not clip.has_root
    assert clip.joint_pos[0].tolist() == pytest.approx([10.0, 20.0])


def test_object_blocks_sliced_in_order(tmp_path: Path) -> None:
    clip = PlaybackClip(
        _write(tmp_path, "c.npy", _matrix(frames=3, num_objects=2)),
        NUM_DOF,
        2,
        fps=10.0,
        quat_order="wxyz",
        device="cpu",
    )
    # object 0 walks +x at 1/frame, object 1 at 2/frame; frame 2 -> x=2 and x=4.
    assert clip.object_pos[2, 0, 0].item() == pytest.approx(2.0)
    assert clip.object_pos[2, 1, 0].item() == pytest.approx(4.0)
    assert clip.object_quat[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # wxyz -> xyzw


def test_quat_order_xyzw_passthrough(tmp_path: Path) -> None:
    mat = _matrix(frames=2)
    mat[:, 3:7] = [0.5, 0.5, 0.5, 0.5]  # unit quat, distinct per component position
    clip = PlaybackClip(_write(tmp_path, "c.npy", mat), NUM_DOF, 0, fps=10.0, quat_order="xyzw", device="cpu")
    assert clip.root_quat[0].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])  # unchanged


def test_non_unit_quat_fails_loud(tmp_path: Path) -> None:
    mat = _matrix(frames=2)
    mat[:, 3:7] = [0.1, 0.2, 0.3, 0.4]  # |q| far from 1 -> misaligned columns
    with pytest.raises(ValueError, match="not unit-norm"):
        PlaybackClip(_write(tmp_path, "c.npy", mat), NUM_DOF, 0, fps=10.0, quat_order="xyzw", device="cpu")


def test_non_finite_values_fail_loud(tmp_path: Path) -> None:
    mat = _matrix(frames=2)
    mat[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PlaybackClip(_write(tmp_path, "c.npy", mat), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")


def test_wrong_column_count_fails_loud(tmp_path: Path) -> None:
    bad = np.zeros((4, 5), dtype=np.float32)  # neither 7+2 nor 2 (+object blocks)
    with pytest.raises(ValueError, match="columns; expected"):
        PlaybackClip(_write(tmp_path, "c.npy", bad), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")


def test_non_2d_matrix_fails_loud(tmp_path: Path) -> None:
    f = tmp_path / "c.npy"
    np.save(f, np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="2D matrix"):
        PlaybackClip(str(f), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")


def test_qpos_npz_layout(tmp_path: Path) -> None:
    # qpos [T, 7+nj] (root wxyz + joints) + one object track + extra keys.
    frames = 4
    t = np.arange(frames, dtype=np.float32)
    qpos = np.zeros((frames, 7 + NUM_DOF), dtype=np.float32)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0  # identity wxyz
    qpos[:, 7] = 10 + t
    qpos[:, 8] = 20 + t
    obj_pos = np.tile(t[:, None], (1, 3))
    obj_rot = np.zeros((frames, 4), dtype=np.float32)
    obj_rot[:, 0] = 1.0  # identity wxyz
    f = tmp_path / "motion.npz"
    np.savez(
        f,
        qpos=qpos,
        object_motion_pos=obj_pos,
        object_motion_rot_wxyz=obj_rot,
        contact_labels=np.zeros((frames, 2), np.float32),
        box_obj_relpath="procedural_boxes/Box_39.obj",
        ref_pos=np.zeros(3, np.float32),
        ref_wxyz=np.array([1, 0, 0, 0], np.float32),
    )
    clip = PlaybackClip(str(f), NUM_DOF, 1, fps=30.0, quat_order="wxyz", device="cpu")
    assert clip.has_root
    assert clip.num_frames == frames
    assert clip.root_pos[0].tolist() == pytest.approx([0.0, 0.0, 0.78])
    assert clip.joint_pos[1].tolist() == pytest.approx([11.0, 21.0])
    assert clip.object_pos[2, 0].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert clip.object_quat[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # wxyz -> xyzw


def test_qpos_npz_without_object_track(tmp_path: Path) -> None:
    qpos = np.zeros((3, 7 + NUM_DOF), dtype=np.float32)
    qpos[:, 3] = 1.0
    f = tmp_path / "motion.npz"
    np.savez(f, qpos=qpos, ref_pos=np.zeros(3, np.float32))
    clip = PlaybackClip(str(f), NUM_DOF, 0, fps=30.0, quat_order="wxyz", device="cpu")
    assert clip.has_root and clip.num_frames == 3


def test_multi_array_npz_fails_loud(tmp_path: Path) -> None:
    f = tmp_path / "c.npz"
    np.savez(f, a=_matrix(), b=_matrix())
    with pytest.raises(ValueError, match="exactly one array"):
        PlaybackClip(str(f), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")


# ----- PlaybackClip: sampling -----


def test_sample_clamps_beyond_clip_end(tmp_path: Path) -> None:
    clip = PlaybackClip(
        _write(tmp_path, "c.npy", _matrix(frames=5)), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu"
    )
    assert clip.duration == pytest.approx(0.4)
    assert clip.sample(99.0, interpolate=True)["root_pos"][0].item() == pytest.approx(4.0)


def test_sample_interpolates_between_frames(tmp_path: Path) -> None:
    clip = PlaybackClip(_write(tmp_path, "c.npy", _matrix()), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")
    mid = clip.sample(0.15, interpolate=True)  # halfway between frames 1 and 2
    assert mid["root_pos"][0].item() == pytest.approx(1.5)
    assert mid["joint_pos"].tolist() == pytest.approx([11.5, 21.5])
    assert mid["root_quat"].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # identity slerp


def test_sample_without_interpolation_snaps_to_frame(tmp_path: Path) -> None:
    clip = PlaybackClip(_write(tmp_path, "c.npy", _matrix()), NUM_DOF, 0, fps=10.0, quat_order="wxyz", device="cpu")
    assert clip.sample(0.14, interpolate=False)["root_pos"][0].item() == pytest.approx(1.0)
    assert clip.sample(0.16, interpolate=False)["root_pos"][0].item() == pytest.approx(2.0)


# ----- MotionPlaybackPlugin against a fake simulator -----


class _FakeSimEngineCfg:
    fps = 100.0
    control_decimation_steps = 2  # control dt = 0.02s


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeSimulator:
    """Stand-in exposing only what MotionPlaybackPlugin touches."""

    def __init__(self, num_envs: int = 2) -> None:
        self.hooks = HookRegistry()
        self.simulator_config = _FakeSimulatorConfig()
        self.sim_device = "cpu"
        self.num_envs = num_envs
        self.num_dof = NUM_DOF
        self.dof_pos = torch.zeros(num_envs, NUM_DOF)
        self.dof_vel = torch.zeros(num_envs, NUM_DOF)
        self.robot_root_states = torch.zeros(num_envs, 13)
        self.shutdown_requested = False
        self.root_writes: list[torch.Tensor] = []
        self.dof_writes: list[torch.Tensor] = []
        self.object_writes: list[tuple[list[str], torch.Tensor]] = []

    @property
    def dof_state(self) -> torch.Tensor:
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def set_actor_root_state_tensor_robots(self, env_ids: torch.Tensor, root_states: torch.Tensor) -> None:
        self.root_writes.append(root_states.clone())

    def set_dof_state_tensor_robots(self, env_ids: torch.Tensor, dof_states: torch.Tensor) -> None:
        self.dof_writes.append(dof_states.clone())

    def set_actor_states(self, names: list[str], env_ids: torch.Tensor, states: torch.Tensor) -> None:
        self.object_writes.append((names, states.clone()))

    def request_shutdown(self, reason: str = "") -> None:
        self.shutdown_requested = True


def _frame(sim: _FakeSimulator, substeps: int = 2) -> None:
    """Emit one control frame the way run_sim/BaseTask do: open, every substep, then close.

    FRAME_END matters as much as the rest here — the plugin ends a finite run from it, so a driver
    that left it out would never see the run stop.
    """
    sim.hooks.emit(Phase.FRAME_BEGIN)
    for _ in range(substeps):
        sim.hooks.emit(Phase.PRE_STEP)
        sim.hooks.emit(Phase.POST_STEP)
    sim.hooks.emit(Phase.FRAME_END)


def _cfg(tmp_path: Path, **kw: Any) -> MotionPlaybackPluginConfig:
    mat = _matrix(frames=kw.pop("frames", 5), num_objects=len(kw.get("objects", [])), fixed_base=kw.pop("fixed", False))
    kw.setdefault("fps", 10.0)
    return MotionPlaybackPluginConfig(motion_files=[_write(tmp_path, "clip.npy", mat)], **kw)


def test_plugin_writes_state_each_substep(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path), sim)
    _frame(sim)
    assert len(sim.root_writes) == 2  # once per PRE_STEP substep
    assert len(sim.dof_writes) == 2
    assert sim.dof_pos.tolist() == [[10.0, 20.0], [10.0, 20.0]]  # frame 0 joints on every env
    assert sim.robot_root_states[0, 0].item() == 0.0
    assert torch.all(sim.dof_vel == 0)  # velocities zeroed in kinematic playback


def test_plugin_advances_playback_clock_per_frame(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path), sim)
    # control dt = 0.02s, clip frame dt = 0.1s -> after 5 frames t=0.08 -> root x interpolates
    for _ in range(5):
        _frame(sim)
    assert sim.robot_root_states[0, 0].item() == pytest.approx(0.8)


def test_plugin_holds_last_frame_and_requests_shutdown(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, frames=2), sim)
    for _ in range(10):
        _frame(sim)
    assert sim.robot_root_states[0, 0].item() == pytest.approx(1.0)  # clamped to last frame
    assert sim.shutdown_requested


def test_shutdown_is_requested_from_the_final_frame_not_while_selecting_it(tmp_path: Path) -> None:
    """The run ends only once the last frame has been egressed.

    Asking during FRAME_BEGIN would leave the driving loop a frame it has opened but should not
    publish; asking from that frame's FRAME_END lets the loop close it normally and stop before the
    next one, so the clip's final frame is published exactly once.
    """
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, frames=2), sim)

    asked_while_selecting = []
    for _ in range(20):
        sim.hooks.emit(Phase.FRAME_BEGIN)
        asked_while_selecting.append(sim.shutdown_requested)
        sim.hooks.emit(Phase.PRE_STEP)
        sim.hooks.emit(Phase.POST_STEP)
        sim.hooks.emit(Phase.FRAME_END)
        if sim.shutdown_requested:
            break

    assert not any(asked_while_selecting), "shutdown was requested while a frame was being selected"
    assert sim.shutdown_requested
    assert sim.robot_root_states[0, 0].item() == pytest.approx(1.0)  # the clip's last frame


def test_a_playlist_error_waits_until_the_chosen_frame_is_egressed(tmp_path: Path) -> None:
    """Resolving the next entry can fail, and the frame already chosen must still be written out.

    Deciding who ends the run means resolving one entry ahead, which happens during FRAME_BEGIN —
    after the clip's final frame has been selected but before PRE_STEP writes it. A failure there
    (an entry whose glob matches nothing, a remote listing error) is held back for one frame so the
    clip does not silently lose its last frame to the error.
    """
    clip = _write(tmp_path, "clip.npy", _matrix(frames=2))
    cfg = MotionPlaybackPluginConfig(motion_files=[clip, str(tmp_path / "absent-*.npy")], fps=10.0)
    sim = _FakeSimulator()
    _make(cfg, sim)

    # Six frames reach and egress the clip's final frame; the bad entry is resolved on the last one.
    for _ in range(6):
        _frame(sim, substeps=1)

    assert sim.robot_root_states[0, 0].item() == pytest.approx(1.0)  # the final frame was written
    assert not sim.shutdown_requested  # an unanswered lookahead must not end the run quietly

    with pytest.raises(ValueError, match="resolved to no"):
        _frame(sim, substeps=1)


def test_a_driver_that_never_egresses_still_ends_a_finite_run(tmp_path: Path) -> None:
    """A driver emitting no FRAME_END ends the run from the next FRAME_BEGIN instead of looping.

    A driver that opens frames without closing them is not a supported lifecycle — the phases are
    meant to pair. This is the fallback for one anyway: ending a finite run is the plugin's own
    responsibility, and a mistake in a driver should cost one repeated frame, not hang the process.
    The supported path, asserted above, ends the run from FRAME_END and publishes each frame once.
    """
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, frames=2), sim)

    for _ in range(20):
        sim.hooks.emit(Phase.FRAME_BEGIN)
        sim.hooks.emit(Phase.PRE_STEP)
        if sim.shutdown_requested:
            break

    assert sim.shutdown_requested


def test_plugin_n_run_zero_loops_forever(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, frames=2, n_run=0), sim)
    xs = []
    for _ in range(30):
        _frame(sim)
        xs.append(sim.robot_root_states[0, 0].item())
    assert not sim.shutdown_requested  # n_run == 0 never terminates on its own
    assert all(x <= 1.0 for x in xs)  # every rendered frame stayed inside the 2-frame clip
    assert xs.count(0.0) >= 2  # the clip genuinely replays: frame 0 (root-x 0) is revisited each loop


def test_plugin_drives_multiple_objects_in_order(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, frames=3, objects=["crate", "ball"]), sim)
    for _ in range(3):
        _frame(sim)
    names, states = sim.object_writes[-1]
    assert names == ["crate", "ball"]
    assert states.shape == (4, 13)  # 2 objects x 2 envs, name-major
    # crate is object-block 0 (x at 1/frame), ball is block 1 (x at 2/frame); rows name-major.
    # ball always moves twice as far along +x as crate, whatever clip time the frames reached.
    assert states[0, 0].item() > 0
    assert states[2, 0].item() == pytest.approx(2 * states[0, 0].item())
    assert states[0, 3:7].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # xyzw identity


def test_plugin_fixed_base_skips_root_write(tmp_path: Path) -> None:
    sim = _FakeSimulator()
    _make(_cfg(tmp_path, fixed=True), sim)
    _frame(sim)
    assert sim.dof_writes  # joints still written
    assert not sim.root_writes  # no root state on a fixed-base clip


def test_plugin_requires_clip() -> None:
    with pytest.raises(ValueError, match="motion_files"):
        _make(MotionPlaybackPluginConfig(), _FakeSimulator())


def test_config_rejects_duplicate_objects() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        MotionPlaybackPluginConfig(objects=["a", "a"])


def test_kinematic_playback_config_flag_default_off() -> None:
    # Kinematic playback is a simulator-config flag (sim.kinematic_playback), not plugin-derived.
    from holosoma.config_values.run_sim import mujoco as mujoco_preset

    assert mujoco_preset.config.sim.kinematic_playback is False


# ----- MotionPlaybackPlugin: episodic playlist (motion_files + n_run + settle) -----
#
# Each clip is one episode. Clips are distinguished by an additive root-x offset (x0), so the
# root-x written at an episode's frame 0 identifies which clip is playing. fps=50 with 2 frames
# makes each clip exactly one control-dt long (duration == control_dt == 0.02s), so one playing
# frame per episode — keeping the frame bookkeeping in these tests short and explicit.


def _multi_cfg(
    tmp_path: Path, x0s: list[float], frames: int = 2, objects: list[str] | None = None, **kw: Any
) -> MotionPlaybackPluginConfig:
    """A playlist config; clip i shifts root-x AND each object-x by x0s[i], so frame 0 x == x0s[i]."""
    objects = objects or []
    files = []
    for i, x0 in enumerate(x0s):
        mat = _matrix(frames=frames, num_objects=len(objects))
        mat[:, 0] += x0  # root x
        for k in range(len(objects)):
            mat[:, 7 + NUM_DOF + 7 * k] += x0  # object k x column (after root 7 + joints)
        files.append(_write(tmp_path, f"clip{i}.npy", mat))
    kw.setdefault("fps", 50.0)
    if objects:
        kw["objects"] = objects
    return MotionPlaybackPluginConfig(motion_files=files, **kw)


def _spy(sim: _FakeSimulator) -> list[tuple[str, int]]:
    """Record episode-boundary emits in order as (phase, env_id)."""
    log: list[tuple[str, int]] = []
    sim.hooks.add(Phase.EPISODE_END, lambda env_id: log.append(("end", env_id)), name="spy.end")
    sim.hooks.add(Phase.EPISODE_START, lambda env_id: log.append(("start", env_id)), name="spy.start")
    return log


def _run_until_shutdown(sim: _FakeSimulator, max_frames: int = 1000) -> tuple[int, list[float]]:
    """Drive frames until shutdown; return (frame_count, root-x written each frame)."""
    xs: list[float] = []
    n = 0
    while not sim.shutdown_requested and n < max_frames:
        _frame(sim, substeps=1)
        xs.append(sim.robot_root_states[0, 0].item())
        n += 1
    return n, xs


def test_episodic_playlist_plays_clips_in_order(tmp_path: Path) -> None:
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0, 200.0]), sim)  # 3 clips, n_run=1 -> 3 episodes
    log = _spy(sim)
    _, xs = _run_until_shutdown(sim)
    # The frame-0 root-x of each episode appears in playlist order.
    assert [x for x in xs if x in (0.0, 100.0, 200.0)][:3] == [0.0, 100.0, 200.0]
    assert sim.shutdown_requested
    # 3 episodes -> 2 internal boundaries (episode 0 uses the sim's startup EPISODE_START).
    assert [p for p, _ in log] == ["end", "start", "end", "start"]


def test_n_run_repeats_playlist(tmp_path: Path) -> None:
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], n_run=2), sim)  # 2 clips x 2 runs = 4 episodes
    log = _spy(sim)
    _, xs = _run_until_shutdown(sim)
    assert [x for x in xs if x in (0.0, 100.0)][:4] == [0.0, 100.0, 0.0, 100.0]  # playlist-repeat order
    assert sim.shutdown_requested
    assert log.count(("start", 0)) == 3  # 4 episodes -> 3 boundaries


def test_n_run_repeats_single_clip(tmp_path: Path) -> None:
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0], n_run=3), sim)  # 1 clip x 3 runs = 3 episodes
    log = _spy(sim)
    _, xs = _run_until_shutdown(sim)
    assert sim.shutdown_requested
    assert log.count(("start", 0)) == 2  # 3 episodes -> 2 boundaries
    # The clip is genuinely re-played (not just boundaries fired): its frame-0 pose (root-x == 0)
    # is rendered once at the start of each of the 3 episodes.
    assert xs.count(0.0) == 3


def test_episode_boundary_emits_end_then_start_per_env(tmp_path: Path) -> None:
    sim = _FakeSimulator(num_envs=2)
    _make(_multi_cfg(tmp_path, [0.0, 100.0]), sim)  # 2 episodes -> 1 boundary
    log = _spy(sim)
    _run_until_shutdown(sim)
    # One boundary: EPISODE_END for every env, then EPISODE_START for every env.
    assert log == [("end", 0), ("end", 1), ("start", 0), ("start", 1)]


def test_settle_frames_hold_after_each_boundary(tmp_path: Path) -> None:
    # Baseline (no settle) vs settle=3: the single A->B boundary adds exactly 3 held frames.
    base_sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], settle_frames=0), base_sim)
    base_frames, _ = _run_until_shutdown(base_sim)

    settle_sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], settle_frames=3), settle_sim)
    settle_frames, xs = _run_until_shutdown(settle_sim)

    assert settle_frames == base_frames + 3
    # During the settle window the clip is held at B's frame 0 (root-x == 100, clock frozen).
    assert xs.count(100.0) >= 3


def test_single_clip_once_emits_no_boundaries(tmp_path: Path) -> None:
    # A single clip played once holds its last frame and shuts down without emitting a boundary.
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0], n_run=1), sim)
    log = _spy(sim)
    _run_until_shutdown(sim)
    assert log == []  # no episode boundaries for a single once-through clip
    assert sim.shutdown_requested


def test_config_rejects_bad_n_run_and_settle() -> None:
    with pytest.raises(ValueError, match="n_run"):
        MotionPlaybackPluginConfig(n_run=-1)
    with pytest.raises(ValueError, match="settle_frames"):
        MotionPlaybackPluginConfig(settle_frames=-1)


def test_settle_emits_single_episode_start_per_boundary(tmp_path: Path) -> None:
    # A boundary emits EPISODE_END/START exactly once even with a multi-frame settle window
    # (the held settle frames must not re-emit the boundary).
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], settle_frames=5), sim)  # 2 episodes -> 1 boundary
    log = _spy(sim)
    _run_until_shutdown(sim)
    assert log.count(("start", 0)) == 1
    assert log.count(("end", 0)) == 1


def test_episodic_renders_each_clips_final_frame(tmp_path: Path) -> None:
    # Each episode plays through its clip's LAST frame before the boundary (never dropped).
    # A 2-frame clip's final frame (index 1) has root-x == x0 + 1, distinct from its frame 0 (x0).
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], frames=2), sim)
    _, xs = _run_until_shutdown(sim)
    assert 1.0 in xs  # clip A final frame (x0=0 + last-frame offset 1)
    assert 101.0 in xs  # clip B final frame (x0=100 + 1)


def test_playlist_entry_forms_all_expand(tmp_path: Path) -> None:
    # The three ways to name clips must all reach the same playlist: one explicit file, several
    # explicit files, and a glob pattern.
    cfg = _multi_cfg(tmp_path, [0.0, 100.0, 200.0])  # writes clip0/clip1/clip2.npy in tmp_path
    files = cfg.motion_files

    single = _make(MotionPlaybackPluginConfig(motion_files=[files[0]], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(single) == [files[0]]

    multi = _make(MotionPlaybackPluginConfig(motion_files=files, fps=50.0), _FakeSimulator())
    assert _resolve_playlist(multi) == files  # explicit entries keep the order they were written in

    pattern = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*.npy")], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(pattern) == sorted(files)


def test_playlist_expansion_drives_episode_count(tmp_path: Path) -> None:
    # A glob playlist is episodic like an explicit one: 3 matched clips x n_run=2 -> 6 episodes.
    _multi_cfg(tmp_path, [0.0, 100.0, 200.0])
    sim = _FakeSimulator(num_envs=1)
    _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*.npy")], fps=50.0, n_run=2), sim)
    log = _spy(sim)
    _, xs = _run_until_shutdown(sim)
    assert [x for x in xs if x in (0.0, 100.0, 200.0)][:6] == [0.0, 100.0, 200.0, 0.0, 100.0, 200.0]
    assert log.count(("start", 0)) == 5  # 6 episodes -> 5 boundaries
    assert sim.shutdown_requested


def test_playlist_mixes_explicit_and_expanded_entries(tmp_path: Path) -> None:
    # Explicit files and a glob can be combined; expansion happens in place, keeping entry order.
    files = _multi_cfg(tmp_path, [0.0, 100.0, 200.0]).motion_files
    sub = tmp_path / "more"
    sub.mkdir()
    extra = _write(sub, "extra.npy", _matrix(frames=2))
    plugin = _make(
        MotionPlaybackPluginConfig(motion_files=[extra, str(tmp_path / "*.npy")], fps=50.0), _FakeSimulator()
    )
    assert _resolve_playlist(plugin) == [extra, *sorted(files)]


def test_expansion_skips_non_clip_files(tmp_path: Path) -> None:
    # A capture folder collects junk beside the clips: notes must not abort the run, and a backup
    # copy of a real clip (loadable, right shape) must not sneak in as an extra episode.
    files = _multi_cfg(tmp_path, [0.0, 100.0]).motion_files
    (tmp_path / "README.md").write_text("not a clip")
    _write(tmp_path, "clip0.npy.bak", _matrix(frames=2))  # a valid clip under a non-clip suffix
    (tmp_path / "bundle.npz").mkdir()  # a *directory* that happens to carry a clip suffix
    _write(tmp_path, "._clip0.npy", _matrix(frames=2))  # a macOS resource fork keeping the .npy suffix
    plugin = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*")], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(plugin) == sorted(files)  # the clips only: no README, .bak, directory or dotfile


def test_expansion_matches_clip_suffixes_case_insensitively(tmp_path: Path) -> None:
    # Capture tools emit .NPZ as readily as .npz, and the loader keys off the lowercased suffix, so
    # expansion must not drop an uppercase clip (it would silently shorten the playlist).
    clip = _write(tmp_path, "CLIP.NPZ", _matrix(frames=2))
    plugin = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*")], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(plugin) == [clip]


def test_repeated_clip_stays_repeated(tmp_path: Path) -> None:
    # Naming a clip twice is a legitimate playlist (play A, then A again), so neither an explicit
    # repeat nor two entries that expand onto the same clip may be deduped.
    files = _multi_cfg(tmp_path, [0.0]).motion_files
    explicit = _make(MotionPlaybackPluginConfig(motion_files=[files[0], files[0]], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(explicit) == [files[0], files[0]]
    overlapping = _make(
        MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*.npy")] * 2, fps=50.0), _FakeSimulator()
    )
    assert _resolve_playlist(overlapping) == files + files

    sim = _FakeSimulator(num_envs=1)  # and the repeat is really played twice, not just listed twice
    _make(MotionPlaybackPluginConfig(motion_files=[files[0], files[0]], fps=50.0), sim)
    log = _spy(sim)
    _, xs = _run_until_shutdown(sim)
    assert log.count(("start", 0)) == 1  # 2 episodes -> 1 boundary
    assert xs.count(0.0) == 2  # the clip's frame 0 is rendered once per episode


def test_pattern_scope_is_the_pattern_not_the_folder(tmp_path: Path) -> None:
    # A single-level pattern takes only what it names, so a nested take does not silently join the
    # playlist; recursion is opt-in via **. A bare folder is not a pattern, so it is passed through.
    files = _multi_cfg(tmp_path, [0.0, 100.0]).motion_files
    take2 = tmp_path / "take2"
    take2.mkdir()
    nested = _write(take2, "clip9.npy", _matrix(frames=2))

    flat = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*.npy")], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(flat) == sorted(files)
    assert nested not in flat.playlist  # the subfolder's clip is not swept in

    recursive = _make(
        MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "**" / "*.npy")], fps=50.0), _FakeSimulator()
    )
    assert [Path(p).name for p in _resolve_playlist(recursive)] == ["clip0.npy", "clip1.npy", "clip9.npy"]

    # A non-pattern entry is left exactly as configured, directory included: subclasses give such
    # entries their own meaning (FAR-pi's SimulatorStatePlaybackPlugin reads a folder of MCAPs).
    folder = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path)], fps=50.0), _FakeSimulator())
    assert _resolve_playlist(folder) == [str(tmp_path)]


def test_expansion_order_is_sorted_and_stable(tmp_path: Path) -> None:
    # An expanded entry must be ordered, not filesystem-arbitrary, so a rerun replays the same
    # episodes in the same order. Sorting is lexicographic, which is what the clip names must
    # anticipate (zero-pad numbering: clip10 sorts before clip2).
    for name in ("clip2.npy", "clip10.npy", "clip1.npy"):
        _write(tmp_path, name, _matrix(frames=2))
    plugin = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*.npy")], fps=50.0), _FakeSimulator())
    assert [Path(p).name for p in _resolve_playlist(plugin)] == ["clip1.npy", "clip10.npy", "clip2.npy"]


def test_single_remote_clip_uses_generic_paths_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = "s3://bucket/motions/clip.npz"
    cached = _write(tmp_path, "clip.npz", _matrix(frames=2))
    calls: list[str] = []

    def resolve(value: str) -> Iterator[str]:
        calls.append(value)
        return iter((cached,))

    monkeypatch.setattr(playback_module, "resolve_paths", resolve)

    plugin = _make(MotionPlaybackPluginConfig(motion_files=[remote], fps=50.0), _FakeSimulator())

    assert calls == [remote]
    assert _resolve_playlist(plugin) == [cached]
    assert plugin._playlist_sources == [remote]


def test_pattern_expansion_uses_generic_paths_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_a = _write(tmp_path, "a.npy", _matrix(frames=2))
    clip_b = _write(tmp_path, "b.npz", _matrix(frames=2))
    non_clip = tmp_path / "notes.md"
    non_clip.touch()
    pattern = "s3://bucket/motions/*"
    calls: list[str] = []

    def resolve(pattern_value: str) -> Iterator[str]:
        calls.append(pattern_value)
        return iter((clip_b, str(non_clip), clip_a))

    monkeypatch.setattr(playback_module, "resolve_paths", resolve)

    plugin = _make(MotionPlaybackPluginConfig(motion_files=[pattern], fps=50.0), _FakeSimulator())

    assert calls == [pattern]
    assert _resolve_playlist(plugin) == [clip_b, clip_a]
    assert plugin._playlist_sources == [pattern, pattern]


def test_non_pattern_loader_results_are_preserved_for_subclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "recordings:take-set"
    mcap = tmp_path / "episode.mcap"
    folder = tmp_path / "rosbag"
    mcap.touch()
    folder.mkdir()
    monkeypatch.setattr(playback_module, "resolve_paths", lambda _value: iter((str(mcap), str(folder))))

    plugin = _make(MotionPlaybackPluginConfig(motion_files=[source], fps=50.0), _FakeSimulator())

    assert _resolve_playlist(plugin) == [str(mcap), str(folder)]
    assert plugin._playlist_sources == [source, source]


def test_remote_pattern_localizes_paths_only_as_consumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_a = _write(tmp_path, "a.npy", _matrix(frames=2))
    clip_b = _write(tmp_path, "b.npy", _matrix(frames=2))
    pattern = "s3://bucket/motions/*.npy"
    localized: list[str] = []

    def resolve(pattern_value: str) -> Iterator[str]:
        assert pattern_value == pattern

        def paths() -> Iterator[str]:
            localized.append(clip_a)
            yield clip_a
            localized.append(clip_b)
            yield clip_b

        return paths()

    monkeypatch.setattr(playback_module, "resolve_paths", resolve)

    plugin = _make(MotionPlaybackPluginConfig(motion_files=[pattern], fps=50.0), _FakeSimulator())

    assert localized == []
    assert plugin._playlist_index(0) == 0
    assert plugin.playlist == [clip_a]
    assert localized == [clip_a]
    assert plugin._playlist_index(0) == 0
    assert localized == [clip_a]
    assert plugin._playlist_index(1) == 1
    assert plugin.playlist == [clip_a, clip_b]
    assert localized == [clip_a, clip_b]


@pytest.mark.parametrize(
    ("name", "is_directory"),
    [
        ("README.md", False),
        ("clip.npy.bak", False),
        ("bundle.npz", True),
    ],
)
def test_singleton_pattern_still_rejects_non_clip_match(tmp_path: Path, name: str, is_directory: bool) -> None:
    candidate = tmp_path / name
    if is_directory:
        candidate.mkdir()
    else:
        candidate.touch()
    plugin = _make(MotionPlaybackPluginConfig(motion_files=[str(tmp_path / "*")], fps=50.0), _FakeSimulator())

    with pytest.raises(ValueError, match="resolved to no clip files"):
        _resolve_playlist(plugin)


def test_playlist_entry_matching_nothing_fails_when_consumed(tmp_path: Path) -> None:
    # A typo'd pattern, one over an empty folder, and one that matches only non-clip files must all
    # fail when playback first consumes the lazy path iterator.
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "notes.txt.bak").write_text("junk")
    for entry in (str(tmp_path / "*.npz"), str(tmp_path / "missing" / "*.npy"), str(empty / "*"), str(tmp_path / "*")):
        plugin = _make(MotionPlaybackPluginConfig(motion_files=[entry], fps=50.0), _FakeSimulator())
        with pytest.raises(ValueError, match="resolved to no (paths|clip files)"):
            _resolve_playlist(plugin)


def test_episodic_object_rewinds_to_new_clip_frame0(tmp_path: Path) -> None:
    # At an episode boundary the driven object is re-pinned to the NEXT clip's frame 0
    # (object-x jumps from clip A's 0 to clip B's 100), same rewind as the robot root.
    sim = _FakeSimulator(num_envs=1)
    _make(_multi_cfg(tmp_path, [0.0, 100.0], objects=["crate"]), sim)
    _run_until_shutdown(sim)
    obj_x = [states[0, 0].item() for _, states in sim.object_writes]
    assert 0.0 in obj_x and 100.0 in obj_x
    assert obj_x.index(0.0) < obj_x.index(100.0)  # clip A object before clip B object
