"""Unit tests for FrameWriterPlugin (pure, no simulator backend; real cv2 + tmp dir I/O).

Reuses the fake-simulator pattern from test_camera_consumer: the plugin is constructed against a
minimal stand-in and driven by emitting FRAME_END. Covers: stream selection (all cameras/modalities
by default, config subsetting), on-disk layout (per-stream dirs, numbered frames, index.jsonl with
an intrinsics header), RGB round-trip through PNG, and both depth formats.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

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


def test_mp4_mode_buffers_and_encodes_at_stop(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available (create_video h264 shells out to it)")
    frames = _rgb(h=32, w=32)
    sim = _FakeSimulator({"head": _cam()}, {("head", "rgb"): frames})
    plugin = _make(FrameWriterPluginConfig(output_dir=str(tmp_path), rgb_format="mp4"), sim)
    for i in range(5):
        sim._t = i * 0.02  # 50Hz sim-time stamps -> encoded fps 50
        _step(sim, {"head"})
    assert not list(tmp_path.glob("**/*.png"))  # buffered, nothing written per frame
    assert len(plugin._video_frames[("head", "rgb", 0)]) == 5
    sim.hooks.emit(Phase.CLOSE)
    videos = list(tmp_path.glob("head_rgb_env0.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 0
    # index.jsonl records every frame against the video path
    records = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()][1:]
    assert [r["frame"] for r in records] == list(range(5))
    assert all(r["path"] == "head_rgb_env0.mp4" for r in records)


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
    written = cv2.cvtColor(cv2.imread(str(tmp_path / "head_rgb_env0" / "000000.png")), cv2.COLOR_BGR2RGB)
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
