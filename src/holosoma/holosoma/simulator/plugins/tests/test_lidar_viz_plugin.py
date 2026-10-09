# mypy: disable-error-code="no-untyped-def"
"""Production rendering checks for the optional local LiDAR point-cloud visualizer."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.image as mpimg
import numpy as np
import numpy.testing as np_test
import numpy.typing as npt
import pytest
from mpl_toolkits.mplot3d.axes3d import Axes3D

from holosoma.config_types.plugin import LidarVizPluginConfig
from holosoma.config_types.sensor import CustomLidarRayPatternConfig, LidarSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.viz.lidar_viz_plugin import LidarVizPlugin
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _TrainingConfig:
    num_envs = 1


class _VideoConfig:
    save_dir = None


class _FakeSimulator:
    def __init__(self) -> None:
        lidar = LidarSensorConfig(
            mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
            pattern=CustomLidarRayPatternConfig(
                ray_directions=[
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                    [-1.0, 0.0, -1.0],
                ],
            ),
        )
        self.hooks = HookRegistry()
        self.sensor_config = {"front/scan": lidar}
        self.training_config = _TrainingConfig()
        self.video_config = _VideoConfig()
        self.headless = True
        self.sensor_manager = SensorManager("cpu", control_hz=50.0)
        self.sensor_manager.register_lidar("front/scan", lidar)
        self._time = 1.25

    def time(self) -> float:
        return self._time

    def get_lidar_data(
        self,
        name: str,
        data_type: str = "points",
        env_ids: Any = None,
        device: Any = None,
    ) -> torch.Tensor:
        return self.sensor_manager.get_lidar(name).buffer_on(data_type, device)


def _emit_fresh_scan(simulator: _FakeSimulator, points: npt.NDArray[np.float32]) -> None:
    simulator.sensor_manager.get_lidar("front/scan").set_buffer("points", torch.from_numpy(points))
    simulator.sensor_manager.collect_lidars_due()
    simulator.hooks.emit(Phase.FRAME_END)


def test_live_window_uses_headless_state_at_plugin_start(monkeypatch) -> None:
    simulator = _FakeSimulator()
    simulator.headless = False
    monkeypatch.setenv("DISPLAY", ":99")
    plugin = LidarVizPluginConfig(live_window=True).get_cls()(
        LidarVizPluginConfig(live_window=True),
        simulator,
    )

    simulator.headless = True
    plugin.start()

    assert not plugin._show_live
    assert plugin._figure_cls is None


def test_live_window_displays_offscreen_render_through_opencv(monkeypatch) -> None:
    simulator = _FakeSimulator()
    simulator.headless = False
    monkeypatch.setenv("DISPLAY", ":99")
    shown: list[tuple[str, npt.NDArray[np.uint8]]] = []
    destroyed: list[str] = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append((name, image.copy())))
    monkeypatch.setattr(cv2, "pollKey", lambda: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda _name, _prop: 1.0)
    monkeypatch.setattr(cv2, "destroyWindow", destroyed.append)

    config = LidarVizPluginConfig(live_window=True)
    config.get_cls()(config, simulator)
    points = np.array([[[0.0, 1.0, -1.0], [1.0, 2.0, -2.0], [4.0, 5.0, -5.0]]], dtype=np.float32)
    _emit_fresh_scan(simulator, points)

    assert len(shown) == 1
    window, image = shown[0]
    assert window == "holosoma LiDAR: front/scan env 0"
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    simulator.hooks.emit(Phase.CLOSE)
    assert destroyed == [window]


def test_saves_numbered_finite_sensor_xyz_with_deterministic_subsampling(tmp_path: Path, monkeypatch) -> None:
    simulator = _FakeSimulator()
    config = LidarVizPluginConfig(
        save_scans=True,
        save_dir=str(tmp_path),
        lidars=["front/scan"],
        max_points=2,
        point_size=3.5,
        view_elevation=15.0,
        view_azimuth=-45.0,
    )
    config.get_cls()(config, simulator)
    captured: list[npt.NDArray[np.float64]] = []
    colors: list[npt.NDArray[np.float64]] = []
    marker_sizes: list[float] = []
    views: list[tuple[float | None, float | None]] = []
    original_scatter = Axes3D.scatter
    original_view_init = Axes3D.view_init

    def capture_scatter(self, xs, ys, zs=0, *args, **kwargs):
        captured.append(np.asarray(np.column_stack((xs, ys, zs)), dtype=np.float64))
        colors.append(np.asarray(kwargs["c"], dtype=np.float64))
        marker_sizes.append(kwargs["s"])
        return original_scatter(self, xs, ys, zs, *args, **kwargs)

    def capture_view_init(self, elev=None, azim=None, *args, **kwargs):
        views.append((elev, azim))
        return original_view_init(self, elev, azim, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "scatter", capture_scatter)
    monkeypatch.setattr(Axes3D, "view_init", capture_view_init)
    points = np.array(
        [
            [
                [0.0, 1.0, -1.0],
                [1.0, 2.0, -2.0],
                [np.nan, 3.0, -3.0],
                [4.0, 5.0, -5.0],
            ]
        ],
        dtype=np.float32,
    )
    _emit_fresh_scan(simulator, points)
    simulator._time = 1.27
    _emit_fresh_scan(simulator, points)

    expected = points[0, [0, 3]]
    assert len(captured) == 2
    np_test.assert_allclose(captured[0], expected)
    np_test.assert_allclose(captured[1], expected)
    np_test.assert_allclose(colors[0], np.linalg.norm(expected, axis=1))
    np_test.assert_allclose(colors[1], np.linalg.norm(expected, axis=1))
    assert marker_sizes == [3.5, 3.5]
    assert views.count((15.0, -45.0)) == 2
    images = sorted(tmp_path.glob("front_scan/env_000/*.png"))
    assert [image.name for image in images] == ["000000.png", "000001.png"]
    for image_path in images:
        image = np.asarray(mpimg.imread(image_path))
        assert image.ndim == 3 and image.shape[0] >= 900 and image.shape[1] >= 1_000
        assert np.isfinite(image).all() and float(image.max()) > float(image.min())

    simulator.hooks.emit(Phase.CLOSE)


def test_default_stream_selection_includes_each_configured_lidar() -> None:
    simulator = _FakeSimulator()
    config = LidarVizPluginConfig(save_scans=True, save_dir="unused")
    plugin: LidarVizPlugin = config.get_cls()(config, simulator)

    assert plugin.wanted_streams() == {("front/scan", 0)}


def test_record_video_encodes_each_fresh_scan_with_simulation_time_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _FakeSimulator()
    config = LidarVizPluginConfig(record_video=True, save_dir=str(tmp_path), lidars=["front/scan"])
    captured: dict[str, object] = {}

    def capture_video(frames, *, fps, save_dir, output_format, wandb_logging):
        captured["frames"] = frames
        captured["fps"] = fps
        captured["save_dir"] = save_dir
        captured["output_format"] = output_format
        captured["wandb_logging"] = wandb_logging

    rendered_clouds: list[npt.NDArray[np.float32]] = []
    original_render = LidarVizPlugin._render

    def capture_render(plugin, finite_points, title):
        rendered_clouds.append(finite_points.copy())
        return original_render(plugin, finite_points, title)

    monkeypatch.setattr("holosoma.simulator.plugins.viz.lidar_viz_plugin.create_video", capture_video)
    monkeypatch.setattr(LidarVizPlugin, "_render", capture_render)
    config.get_cls()(config, simulator)
    points = np.array(
        [
            [
                [0.0, 1.0, -1.0],
                [1.0, 2.0, -2.0],
                [np.nan, 3.0, -3.0],
                [4.0, 5.0, -5.0],
            ]
        ],
        dtype=np.float32,
    )
    later_points = points.copy()
    later_points[0, 1] = [-2.0, 3.0, -4.0]

    _emit_fresh_scan(simulator, points)
    simulator._time = 1.75
    _emit_fresh_scan(simulator, later_points)
    simulator.hooks.emit(Phase.CLOSE)

    frames = captured["frames"]
    assert isinstance(frames, np.ndarray)
    assert frames.shape[0] == 2 and frames.shape[-1] == 3
    assert frames.dtype == np.uint8 and np.isfinite(frames).all()
    assert float(frames.max()) > float(frames.min())
    assert captured["fps"] == pytest.approx(2.0)
    assert captured["save_dir"] == tmp_path / "front_scan" / "env_000"
    assert captured["output_format"] == "h264"
    assert captured["wandb_logging"] is False
    assert len(rendered_clouds) == 2
    np_test.assert_allclose(rendered_clouds[0], points[0, [0, 1, 3]])
    np_test.assert_allclose(rendered_clouds[1], later_points[0, [0, 1, 3]])
    assert not np.array_equal(rendered_clouds[0], rendered_clouds[1])


def test_record_video_writes_a_decodable_mp4(tmp_path: Path) -> None:
    """The configured video path must produce a real file that OpenCV can decode."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available (create_video h264 shells out to it)")
    simulator = _FakeSimulator()
    config = LidarVizPluginConfig(record_video=True, save_dir=str(tmp_path), lidars=["front/scan"])
    config.get_cls()(config, simulator)
    first = np.array([[[0.0, 1.0, -1.0], [1.0, 2.0, -2.0], [4.0, 5.0, -5.0]]], dtype=np.float32)
    second = first.copy()
    second[0, 1] = [-2.0, 3.0, -4.0]

    _emit_fresh_scan(simulator, first)
    simulator._time = 1.75
    _emit_fresh_scan(simulator, second)
    simulator.hooks.emit(Phase.CLOSE)

    videos = list(tmp_path.glob("front_scan/env_000/*.mp4"))
    assert len(videos) == 1
    assert videos[0].stat().st_size > 0
    capture = cv2.VideoCapture(str(videos[0]))
    decoded: list[npt.NDArray[np.uint8]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(np.asarray(frame, dtype=np.uint8))
    capture.release()
    assert len(decoded) == 2
    assert all(frame.ndim == 3 and frame.shape[2] == 3 for frame in decoded)
    assert not np.array_equal(decoded[0], decoded[1])
