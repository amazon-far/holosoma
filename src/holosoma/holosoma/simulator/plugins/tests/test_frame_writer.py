"""Unit tests for FrameWriterPlugin (pure, no simulator backend; real cv2 + tmp dir I/O).

Reuses the fake-simulator pattern from test_camera_consumer: the plugin is constructed against a
minimal stand-in and driven by emitting FRAME_END. Covers: stream selection (all cameras/modalities
by default, config subsetting), on-disk layout (per-stream dirs, numbered frames, index.jsonl with
an intrinsics header), RGB round-trip through PNG, and both depth formats.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

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


class _FakeSensorManager:
    def __init__(self) -> None:
        self.last_due: set[str] = set()


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
        self.sensor_config = sensors_config
        self.sensor_manager = _FakeSensorManager()
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

    assert sorted(p.name for p in tmp_path.iterdir()) == ["head_rgb_env0.mp4", "index.jsonl"]
    assert _probe(tmp_path / "head_rgb_env0.mp4") == (pytest.approx(50.0, rel=1e-3), 5)
    decoded = _decode(tmp_path / "head_rgb_env0.mp4")
    assert [float(f.mean()) for f in decoded] == pytest.approx([0, 10, 20, 30, 40], abs=3)
    # index.jsonl records every frame against the video path
    records = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()][1:]
    assert [r["frame"] for r in records] == list(range(5))
    assert all(r["path"] == "head_rgb_env0.mp4" for r in records)


@requires_ffmpeg
def test_mp4_fps_follows_camera_decimation(tmp_path: Path) -> None:
    # update_decimation=2 at a 50 Hz control rate renders at 25 Hz: frames 0.04 s apart, no retiming.
    cam = CameraSensorConfig(mount=_MOUNT, data_types=["rgb"], update_decimation=2)
    plugin, sim = _run_mp4(tmp_path, 6, dt=0.04, cam=cam)
    assert plugin._encoders[("head", "rgb", 0)].fps == pytest.approx(25.0)
    sim.hooks.emit(Phase.CLOSE)
    assert _probe(tmp_path / "head_rgb_env0.mp4") == (pytest.approx(25.0, rel=1e-3), 6)


@requires_ffmpeg
def test_mp4_retimed_when_sim_time_disagrees_with_render_rate(tmp_path: Path) -> None:
    # Opened at the 50 Hz render rate, but the stamps say 20 Hz: the video must play at sim speed.
    plugin, sim = _run_mp4(tmp_path, 6, dt=0.05)
    assert plugin._encoders[("head", "rgb", 0)].fps == pytest.approx(50.0)
    sim.hooks.emit(Phase.CLOSE)
    fps, count = _probe(tmp_path / "head_rgb_env0.mp4")
    assert count == 6
    assert fps == pytest.approx(20.0, rel=1e-3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["head_rgb_env0.mp4", "index.jsonl"]


@requires_ffmpeg
@pytest.mark.parametrize("n", [0, 1])
def test_mp4_stream_under_two_frames_writes_nothing(tmp_path: Path, n: int) -> None:
    _, sim = _run_mp4(tmp_path, n)
    sim.hooks.emit(Phase.CLOSE)
    assert not list(tmp_path.glob("*.mp4*"))


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
def test_mp4_encoder_death_disables_stream_and_stop_raises(tmp_path: Path) -> None:
    plugin, sim = _run_mp4(tmp_path, 3)
    encoder = plugin._encoders[("head", "rgb", 0)]
    encoder._proc.kill()
    encoder._proc.wait()
    _step(sim, {"head"})  # the write hits a dead pipe
    _step(sim, {"head"})
    assert ("head", "rgb", 0) in plugin._video_failed
    with pytest.raises(RuntimeError, match="head_rgb_env0: ffmpeg"):
        sim.hooks.emit(Phase.CLOSE)
    assert not (tmp_path / "head_rgb_env0.mp4").exists()


@requires_ffmpeg
def test_mp4_nvenc_falls_back_to_libx264(tmp_path: Path) -> None:
    # 32x32 is below NVENC's minimum frame size, so this falls back on every host, GPU or not.
    plugin, sim = _run_mp4(tmp_path, 4, video_codec="h264_nvenc")
    assert plugin._encoders[("head", "rgb", 0)].codec == "libx264"
    sim.hooks.emit(Phase.CLOSE)
    assert _probe(tmp_path / "head_rgb_env0.mp4")[1] == 4


@requires_ffmpeg
def test_mp4_encoder_options_reach_ffmpeg(tmp_path: Path) -> None:
    # A keyframe every 2 frames: decode the stream's picture types to see -g applied.
    _, sim = _run_mp4(tmp_path, 6, video_gop=2, video_crf=10, video_preset="ultrafast")
    sim.hooks.emit(Phase.CLOSE)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame", "-of", "csv=p=0"]
        + [str(tmp_path / "head_rgb_env0.mp4")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert [line.split(",")[0] for line in out.stdout.split()] == ["1", "0", "1", "0", "1", "0"]


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
