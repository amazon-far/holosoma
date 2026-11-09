"""Unit tests for FrameWriterPlugin (pure, no simulator backend; real cv2 + tmp dir I/O).

Reuses the fake-simulator pattern from test_camera_consumer: the plugin is constructed against a
minimal stand-in and driven by emitting FRAME_END. Covers: stream selection (all cameras/modalities
by default, config subsetting), on-disk layout (per-stream dirs, numbered frames, index.jsonl with
an intrinsics header), RGB round-trip through PNG, and both depth formats.
"""

from __future__ import annotations

import json
import shutil
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch
from loguru import logger

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.plugin import FrameWriterPluginConfig
from holosoma.config_types.sensor import CameraDataType, CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.frame_writer import FrameWriterPlugin
from holosoma.utils.simulator_config import SimulatorType


def _make(cfg: FrameWriterPluginConfig, sim: Any) -> FrameWriterPlugin:
    # Construct via get_cls() (untyped), as the other plugin tests do with their fake simulators.
    plugin: FrameWriterPlugin = cfg.get_cls()(cfg, sim)
    return plugin


pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


class _FakeCameraRecord:
    def __init__(self, effective_decimation: int) -> None:
        self.effective_decimation = effective_decimation


class _FakeSensorManager:
    def __init__(self, sensors_config: dict[str, CameraSensorConfig], control_hz: float) -> None:
        self.last_due: set[str] = set()
        self._sensors_config = sensors_config
        self._control_hz = control_hz

    def get(self, name: str) -> _FakeCameraRecord:
        # Resolved as the real SensorManager does at registration.
        cfg = self._sensors_config[name]
        return _FakeCameraRecord(resolve_decimation(cfg.update_decimation, self._control_hz, field=name))


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeTrainingConfig:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs


class _FakeSimulator:
    def __init__(
        self,
        sensors_config: dict[str, CameraSensorConfig],
        frames: dict[tuple[str, str], npt.NDArray[Any]],
        num_envs: int = 1,
    ) -> None:
        self.hooks = HookRegistry()
        self.sensor_render_hook = self.hooks.add(Phase.FRAME_END, lambda: None, name="sensors.render")
        self.sensor_config = sensors_config
        sim = _FakeSimEngineCfg
        self.sensor_manager = _FakeSensorManager(sensors_config, sim.fps / sim.control_decimation_steps)
        self.training_config = _FakeTrainingConfig(num_envs)
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self._frames = frames
        self._t = 0.0

    def time(self) -> float:
        return self._t

    def get_simulator_type(self) -> SimulatorType:
        return SimulatorType.MUJOCO

    def get_camera_data(self, name: str, data_type: str = "rgb", env_ids: Any = None, device: Any = None) -> Any:
        return torch.from_numpy(self._frames[(name, data_type)])


def _cam(data_types: Sequence[CameraDataType] = ("rgb",)) -> CameraSensorConfig:
    return CameraSensorConfig(mount=_MOUNT, data_types=list(data_types))


def _rgb(n: int = 1, h: int = 4, w: int = 6) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)


def _step(sim: _FakeSimulator, due: set[str]) -> None:
    sim.sensor_manager.last_due = due
    sim.hooks.emit(Phase.FRAME_END)


def test_writes_numbered_frames_and_index(tmp_path: Path) -> None:
    frames = _rgb()
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): frames})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path / "out")), sim)
    _step(sim, {"head"})
    sim._t = 0.02
    _step(sim, {"head"})
    sim.hooks.emit(Phase.CLOSE)

    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "head_rgb_env0").iterdir()) == ["000000.png", "000001.png"]
    lines = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    header, records = lines[0], lines[1:]
    assert header["intrinsics"]["head"]["width"] == 128  # config default
    assert header["intrinsics"]["head"]["simulator_type"] == "mujoco"
    assert header["intrinsics"]["head"]["backend_config"]["use_shadows"] is None
    assert [r["frame"] for r in records] == [0, 1]
    assert records[1]["sim_time"] == 0.02
    assert records[0]["camera"] == "head"


def _decode(path: Path) -> list[npt.NDArray[np.uint8]]:
    capture = cv2.VideoCapture(str(path))
    frames: list[npt.NDArray[np.uint8]] = []
    try:
        while True:
            valid, bgr = capture.read()
            if not valid:
                return frames
            frames.append(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8))
    finally:
        capture.release()


def _rows(out: Path) -> list[dict[str, Any]]:
    """The index's frame rows (every line after the header)."""
    return [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()][1:]


def _videos(out: Path) -> dict[str, Any]:
    """``videos.json``: how each mp4 stream ended."""
    result: dict[str, Any] = json.loads((out / "videos.json").read_text())
    return result


def _probe(path: Path) -> tuple[float, int]:
    """(fps, frame count) of a video as a player sees it."""
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _run_mp4(
    tmp_path: Path, n: int, dt: float = 0.02, cam: CameraSensorConfig | None = None, **cfg: Any
) -> tuple[FrameWriterPlugin, _FakeSimulator]:
    """Publish ``n`` distinct 32x32 frames stamped ``dt`` apart (render rate: 50 Hz control / decimation)."""
    sim = _FakeSimulator({"head": cam or _cam()}, {("head", "rgb"): _rgb(h=32, w=32)})
    plugin = _make(FrameWriterPluginConfig(output_dir=str(tmp_path), rgb_format="mp4", **cfg), sim)
    for i in range(n):
        sim._frames[("head", "rgb")] = np.full((1, 32, 32, 3), (10 * i) % 250, dtype=np.uint8)
        sim._t = i * dt
        _step(sim, {"head"})
    return plugin, sim


requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@requires_ffmpeg
def test_mp4_mode_streams_frames_and_publishes_at_stop(tmp_path: Path) -> None:
    plugin, sim = _run_mp4(tmp_path, 5)
    assert not list(tmp_path.glob("**/*.png"))  # nothing written per frame
    # Encoding is under way while the episode runs; no finished-looking video exists until stop.
    assert plugin._encoders[("head", "rgb", 0)].frames == 5
    assert plugin._encoders[("head", "rgb", 0)]._proc.poll() is None
    assert not (tmp_path / "head_rgb_env0.mp4").exists()
    sim.hooks.emit(Phase.CLOSE)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["head_rgb_env0.mp4", "index.jsonl", "videos.json"]
    assert _probe(tmp_path / "head_rgb_env0.mp4") == (pytest.approx(50.0, rel=1e-3), 5)
    decoded = _decode(tmp_path / "head_rgb_env0.mp4")
    assert [float(f.mean()) for f in decoded] == pytest.approx([0, 10, 20, 30, 40], abs=3)
    # index.jsonl records every frame against the video path; videos.json says how the video ended
    records = _rows(tmp_path)
    assert [r["frame"] for r in records] == list(range(5))
    assert all(r["path"] == "head_rgb_env0.mp4" for r in records)
    assert _videos(tmp_path) == {"head_rgb_env0": {"status": "complete", "path": "head_rgb_env0.mp4", "frames": 5}}


@requires_ffmpeg
def test_mp4_stream_that_never_rendered_is_recorded_empty(tmp_path: Path) -> None:
    # Only "head" renders, so the wanted wrist stream ends with no frames and no encoder.
    sim = _FakeSimulator({"head": _cam(), "wrist": _cam()}, {("head", "rgb"): _rgb(h=32, w=32)})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), rgb_format="mp4"), sim)
    for i in range(3):
        sim._t = i * 0.02
        _step(sim, {"head"})
    sim.hooks.emit(Phase.CLOSE)
    assert _videos(tmp_path)["wrist_rgb_env0"] == {"status": "empty", "path": None, "frames": 0}
    assert _videos(tmp_path)["head_rgb_env0"]["status"] == "complete"


@requires_ffmpeg
def test_mp4_rerun_into_the_same_directory_drops_the_earlier_outputs(tmp_path: Path) -> None:
    # A finished video and videos.json from an earlier run must not survive into a run that never
    # finishes, where they would read as this run's output.
    plugin, sim = _run_mp4(tmp_path, 3)
    sim.hooks.emit(Phase.CLOSE)
    assert (tmp_path / "head_rgb_env0.mp4").exists()
    plugin, _ = _run_mp4(tmp_path, 2)  # still running: no stop yet
    assert not (tmp_path / "head_rgb_env0.mp4").exists()
    assert not (tmp_path / "videos.json").exists()
    plugin._encoders[("head", "rgb", 0)].discard()


@requires_ffmpeg
def test_mp4_fps_follows_camera_decimation(tmp_path: Path) -> None:
    # update_decimation=2 at a 50 Hz control rate renders at 25 Hz: frames 0.04 s apart.
    cam = CameraSensorConfig(mount=_MOUNT, data_types=["rgb"], update_decimation=2)
    plugin, sim = _run_mp4(tmp_path, 6, dt=0.04, cam=cam)
    assert plugin._encoders[("head", "rgb", 0)].fps == pytest.approx(25.0)
    sim.hooks.emit(Phase.CLOSE)
    assert _probe(tmp_path / "head_rgb_env0.mp4") == (pytest.approx(25.0, rel=1e-3), 6)


@requires_ffmpeg
def test_mp4_warns_when_sim_time_disagrees_with_render_rate(tmp_path: Path) -> None:
    # Opened at the 50 Hz render rate, but the stamps say 20 Hz: the mismatch is flagged and the
    # video keeps the render rate.
    _, sim = _run_mp4(tmp_path, 6, dt=0.05)
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING")
    try:
        sim.hooks.emit(Phase.CLOSE)
    finally:
        logger.remove(sink)
    assert any("20 fps of sim time, not the 50 fps render rate" in w for w in warnings)
    assert _probe(tmp_path / "head_rgb_env0.mp4") == (pytest.approx(50.0, rel=1e-3), 6)


@requires_ffmpeg
@pytest.mark.parametrize("n", [0, 1])
def test_mp4_stream_under_two_frames_writes_nothing(tmp_path: Path, n: int) -> None:
    _, sim = _run_mp4(tmp_path, n)
    sim.hooks.emit(Phase.CLOSE)
    assert not list(tmp_path.glob("*.mp4*"))
    if n:  # with no frame at all, start never runs and nothing is written
        assert _videos(tmp_path) == {"head_rgb_env0": {"status": "empty", "path": None, "frames": n}}


@requires_ffmpeg
def test_mp4_ram_stays_flat_over_a_long_episode(tmp_path: Path) -> None:
    # 400 distinct 160x120 frames are 23 MB of pixels; a writer that buffers them shows it in the peak.
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): _rgb(h=120, w=160)})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), rgb_format="mp4"), sim)

    def publish(frames: range) -> None:
        for i in frames:
            sim._frames[("head", "rgb")] = np.full((1, 120, 160, 3), i % 250, dtype=np.uint8)
            sim._t = i * 0.02
            _step(sim, {"head"})

    tracemalloc.start()
    try:
        publish(range(20))  # encoder open, index and caches warm
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        publish(range(20, 400))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    sim.hooks.emit(Phase.CLOSE)
    assert peak - baseline < 2_000_000
    assert _probe(tmp_path / "head_rgb_env0.mp4")[1] == 400


@requires_ffmpeg
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143), SystemExit(1)])
def test_mp4_interrupted_teardown_leaves_detectable_partial(tmp_path: Path, interrupt: BaseException) -> None:
    # CLOSE runs while the interrupt unwinds (run_sim's teardown is in __exit__/except), so the
    # truncated episode must stay under its .partial name. SystemExit(143) is SIGTERM's (128 + 15).
    _, sim = _run_mp4(tmp_path, 4)

    def interrupted_run() -> None:
        try:
            raise interrupt
        finally:
            sim.hooks.emit(Phase.CLOSE)

    with pytest.raises(type(interrupt)):
        interrupted_run()
    assert not (tmp_path / "head_rgb_env0.mp4").exists()
    partial = tmp_path / "head_rgb_env0.mp4.partial"
    assert len(_decode(partial)) == 4  # finalized by ffmpeg, so the frames are recoverable
    assert _videos(tmp_path) == {
        "head_rgb_env0": {"status": "partial", "path": "head_rgb_env0.mp4.partial", "frames": 4}
    }


@requires_ffmpeg
@pytest.mark.parametrize("code", [0, None])
def test_mp4_clean_system_exit_teardown_publishes(tmp_path: Path, code: int | None) -> None:
    # sys.exit(0) / sys.exit() ends a run that succeeded: its videos are complete and must publish.
    _, sim = _run_mp4(tmp_path, 4)

    def exiting_run() -> None:
        try:
            raise SystemExit(code)
        finally:
            sim.hooks.emit(Phase.CLOSE)

    with pytest.raises(SystemExit):
        exiting_run()
    assert not (tmp_path / "head_rgb_env0.mp4.partial").exists()
    assert len(_decode(tmp_path / "head_rgb_env0.mp4")) == 4


@requires_ffmpeg
def test_mp4_encoder_death_disables_stream_and_teardown_finishes(tmp_path: Path) -> None:
    plugin, sim = _run_mp4(tmp_path, 3)
    encoder = plugin._encoders[("head", "rgb", 0)]
    encoder._proc.kill()
    encoder._proc.wait()
    _step(sim, {"head"})  # the write hits a dead pipe
    _step(sim, {"head"})
    assert ("head", "rgb", 0) in plugin._video_failed
    errors: list[str] = []
    sink = logger.add(errors.append, level="ERROR")
    try:
        sim.hooks.emit(Phase.CLOSE)  # an encode failure must not stop the CLOSE chain
    finally:
        logger.remove(sink)
    assert any("mp4 encoding failed for head_rgb_env0: ffmpeg" in e for e in errors)
    assert not (tmp_path / "head_rgb_env0.mp4").exists()
    # The dropped frames get no index rows, and videos.json marks the stream failed.
    assert [r["frame"] for r in _rows(tmp_path)] == [0, 1, 2]
    assert _videos(tmp_path)["head_rgb_env0"]["status"] == "failed"
    assert _videos(tmp_path)["head_rgb_env0"]["frames"] == 3


@requires_ffmpeg
def test_mp4_nvenc_falls_back_to_libx264(tmp_path: Path) -> None:
    # 32x32 is below NVENC's minimum frame size, so this falls back on every host, GPU or not.
    plugin, sim = _run_mp4(tmp_path, 4, video_codec="h264_nvenc")
    assert plugin._encoders[("head", "rgb", 0)].codec == "libx264"
    sim.hooks.emit(Phase.CLOSE)
    assert _probe(tmp_path / "head_rgb_env0.mp4")[1] == 4


def test_mp4_mode_depth_still_writes_files(tmp_path: Path) -> None:
    depth = np.full((1, 4, 4, 1), 2.0, dtype=np.float32)
    sim = _FakeSimulator({"head": _cam(("rgb", "depth"))}, {("head", "rgb"): _rgb(), ("head", "depth"): depth})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), rgb_format="mp4"), sim)
    _step(sim, {"head"})
    assert (tmp_path / "head_depth_env0" / "000000.png").exists()  # depth unaffected by mp4 mode


def test_rgb_png_round_trips_exact(tmp_path: Path) -> None:
    frames = _rgb()
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): frames})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path)), sim)
    _step(sim, {"head"})
    bgr = cv2.imread(str(tmp_path / "head_rgb_env0" / "000000.png"))
    assert bgr is not None, "imread returned None: the plugin did not write a readable PNG"
    written = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(written, frames[0])


def test_depth_png16_millimeters_and_inf_saturation(tmp_path: Path) -> None:
    depth = np.full((1, 2, 2, 1), 1.5, dtype=np.float32)
    depth[0, 0, 0, 0] = np.inf  # no-hit
    sim = _FakeSimulator({"d": _cam(("depth",))}, {("d", "depth"): depth})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path)), sim)
    _step(sim, {"d"})
    written = cv2.imread(str(tmp_path / "d_depth_env0" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert written is not None  # imread returns None on a missing/invalid file (typed Optional)
    assert written.dtype == np.uint16
    assert written[0, 0] == np.iinfo(np.uint16).max
    assert written[1, 1] == 1500


def test_depth_npy_exact_meters(tmp_path: Path) -> None:
    depth = np.array([[[[0.5], [np.inf]], [[2.25], [3.0]]]], dtype=np.float32)
    sim = _FakeSimulator({"d": _cam(("depth",))}, {("d", "depth"): depth})
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), depth_format="npy"), sim)
    _step(sim, {"d"})
    written = np.load(tmp_path / "d_depth_env0" / "000000.npy")
    np.testing.assert_array_equal(written, depth[0, ..., 0])


def test_camera_and_modality_subsetting(tmp_path: Path) -> None:
    sim = _FakeSimulator(
        {"head": _cam(("rgb", "depth")), "wrist": _cam()},
        {("head", "rgb"): _rgb(), ("head", "depth"): np.ones((1, 4, 6, 1), np.float32), ("wrist", "rgb"): _rgb()},
    )
    plugin = _make(FrameWriterPluginConfig(output_dir=str(tmp_path), cameras=["head"], modalities=["rgb"]), sim)
    assert plugin.wanted_streams() == {("head", "rgb", 0)}
    _step(sim, {"head", "wrist"})
    assert (tmp_path / "head_rgb_env0" / "000000.png").exists()
    assert not (tmp_path / "wrist_rgb_env0").exists()


def test_multi_env_streams(tmp_path: Path) -> None:
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): _rgb(n=3)}, num_envs=3)
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), env_ids=[0, 2]), sim)
    _step(sim, {"head"})
    assert (tmp_path / "head_rgb_env0" / "000000.png").exists()
    assert (tmp_path / "head_rgb_env2" / "000000.png").exists()
    assert not (tmp_path / "head_rgb_env1").exists()


def test_default_env_ids_captures_all_envs(tmp_path: Path) -> None:
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): _rgb(n=3)}, num_envs=3)
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path)), sim)  # env_ids=None -> all envs
    _step(sim, {"head"})
    for env in range(3):
        assert (tmp_path / f"head_rgb_env{env}" / "000000.png").exists()


def test_empty_stream_selection_fails_loud(tmp_path: Path) -> None:
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): _rgb()})
    with pytest.raises(ValueError, match="no streams"):
        _make(FrameWriterPluginConfig(output_dir=str(tmp_path), modalities=["depth"]), sim)


# ----- episodic mode, driven by real queue playback -----


class _PlaybackWriterSim(_FakeSimulator):
    """A fake that both plugins can drive: playback writes root states, each env's camera shows its root x."""

    def __init__(self, num_envs: int, h: int = 32, w: int = 32) -> None:
        super().__init__({"head": _cam()}, {}, num_envs=num_envs)
        from holosoma.simulator.shared.scene_types import EnvOriginsScene

        self.num_envs = num_envs
        self.num_dof = 2
        self.sim_device = "cpu"
        self.dof_pos = torch.zeros(num_envs, 2)
        self.dof_vel = torch.zeros(num_envs, 2)
        self.robot_root_states = torch.zeros(num_envs, 13)
        self.scene = EnvOriginsScene(torch.zeros(num_envs, 3), "cpu")
        self.shutdown_requested = False
        self.installed_plugins: dict[str, Any] = {}
        self._shape = (h, w)

    def request_shutdown(self, reason: str = "") -> None:
        self.shutdown_requested = True

    def get_camera_data(self, name: str, data_type: str = "rgb", env_ids: Any = None, device: Any = None) -> Any:
        x = self.robot_root_states[:, 0].round().to(torch.uint8)  # each env's frame encodes its root x
        return x[:, None, None, None].expand(self.num_envs, *self._shape, 3).clone()


def _clip_files(tmp_path: Path, lengths: list[int]) -> list[str]:
    """Clip i: ``lengths[i]`` rows, root x = 50 * i + row (wxyz identity quat), 2 joints."""
    files = []
    for i, rows in enumerate(lengths):
        mat = np.zeros((rows, 9), dtype=np.float32)
        mat[:, 0] = 50 * i + np.arange(rows)
        mat[:, 3] = 1.0
        path = tmp_path / f"clip{i}.npy"
        np.save(path, mat)
        files.append(str(path))
    return files


def _run_episodic(
    tmp_path: Path, lengths: list[int], num_envs: int, assignment: str = "queue", settle: int = 0, **writer: Any
) -> Path:
    from holosoma.config_types.plugin import MotionPlaybackPluginConfig

    out = tmp_path / "out"
    sim = _PlaybackWriterSim(num_envs)
    play_cfg = MotionPlaybackPluginConfig(
        motion_files=_clip_files(tmp_path, lengths),
        fps=50.0,
        env_assignment=assignment,  # type: ignore[arg-type]
        settle_frames=settle,
    )
    sim.installed_plugins["play"] = play_cfg.get_cls()(play_cfg, sim)
    sim.installed_plugins["out"] = _make(FrameWriterPluginConfig(output_dir=str(out), episodic=True, **writer), sim)
    for env in range(num_envs):  # run_sim's startup boundary
        sim.hooks.emit(Phase.EPISODE_START, env)
    frame = 0
    while not sim.shutdown_requested and frame < 100:
        sim.hooks.emit(Phase.FRAME_BEGIN)
        sim.hooks.emit(Phase.PRE_STEP)
        sim._t = frame * 0.02
        _step(sim, {"head"})
        frame += 1
    sim.hooks.emit(Phase.CLOSE)
    return out


def _episode_values(out: Path, name: str) -> list[int]:
    stream = out / name / "head_rgb"
    return [int(cv2.imread(str(p))[0, 0, 0]) for p in sorted(stream.iterdir())]


def test_episodic_writes_one_dir_per_episode_with_its_own_frames(tmp_path: Path) -> None:
    out = _run_episodic(tmp_path, [3, 5, 2, 4], num_envs=2)
    assert sorted(p.name for p in out.iterdir()) == ["clip0", "clip1", "clip2", "clip3"]
    for i, rows in enumerate([3, 5, 2, 4]):
        # Every row of the clip, once, in order; nothing from another clip or from an idle env's hold.
        assert _episode_values(out, f"clip{i}") == [50 * i + r for r in range(rows)]
        lines = [json.loads(line) for line in (out / f"clip{i}" / "index.jsonl").read_text().splitlines()]
        assert lines[0]["episode"] == f"clip{i}"
        assert [r["frame"] for r in lines[1:]] == list(range(rows))
        assert all(r["path"].startswith("head_rgb/") for r in lines[1:])
    # clip0 (env 0) frees first and env 0 takes clip2; clip1 stays on env 1; clip3 goes to env 0.
    envs = {i: json.loads((out / f"clip{i}" / "index.jsonl").read_text().splitlines()[0])["env_id"] for i in range(4)}
    assert envs == {0: 0, 1: 1, 2: 0, 3: 0}


def test_episodic_env_without_a_clip_writes_nothing(tmp_path: Path) -> None:
    out = _run_episodic(tmp_path, [3], num_envs=3)
    assert [p.name for p in out.iterdir()] == ["clip0"]


def test_episodic_shared_playback_qualifies_repeated_names_by_env(tmp_path: Path) -> None:
    out = _run_episodic(tmp_path, [3], num_envs=2, assignment="shared")
    assert sorted(p.name for p in out.iterdir()) == ["clip0", "clip0_env1"]
    assert _episode_values(out, "clip0") == _episode_values(out, "clip0_env1") == [0, 1, 2]


def test_episodic_without_playback_numbers_episodes(tmp_path: Path) -> None:
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): _rgb(n=2)}, num_envs=2)
    _make(FrameWriterPluginConfig(output_dir=str(tmp_path), episodic=True), sim)
    for env in (0, 1):
        sim.hooks.emit(Phase.EPISODE_START, env)
    _step(sim, {"head"})
    sim.hooks.emit(Phase.EPISODE_END, 0)
    sim.hooks.emit(Phase.EPISODE_START, 0)
    _step(sim, {"head"})
    sim.hooks.emit(Phase.CLOSE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode_00000", "episode_00001", "episode_00002"]
    assert (tmp_path / "episode_00002" / "head_rgb" / "000000.png").exists()  # env 0's second episode


def test_episodic_mp4_finalizes_each_episode_at_its_end(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available (create_video h264 shells out to it)")
    out = _run_episodic(tmp_path, [6, 9, 5], num_envs=2, rgb_format="mp4")
    for i, rows in enumerate([6, 9, 5]):
        video = out / f"clip{i}" / "head_rgb.mp4"
        assert video.exists()
        assert not (out / f"clip{i}" / "head_rgb.mp4.partial").exists()
        videos = json.loads((out / f"clip{i}" / "videos.json").read_text())
        assert videos == {"head_rgb": {"status": "complete", "path": "head_rgb.mp4", "frames": rows}}
        capture = cv2.VideoCapture(str(video))
        assert (capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) == (50.0, rows)
        capture.release()


def test_episodic_skips_playback_settle_frames(tmp_path: Path) -> None:
    # settle_frames holds each new clip's first frame after a switch; the first of those is the frame a
    # renderer can still show the previous clip in. Skipped, every episode keeps exactly its rows.
    out = _run_episodic(tmp_path, [3, 4, 3], num_envs=2, settle=2)
    assert _episode_values(out, "clip0") == [0, 1, 2]
    assert _episode_values(out, "clip1") == [50, 51, 52, 53]
    assert _episode_values(out, "clip2") == [100, 101, 102]
