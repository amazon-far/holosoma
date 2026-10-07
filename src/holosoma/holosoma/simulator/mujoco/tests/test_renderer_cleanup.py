"""Focused tests for explicit MuJoCo renderer cleanup."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("mujoco")

from holosoma.config_types.sensor import MujocoCameraConfig
from holosoma.config_types.video import VideoConfig
from holosoma.simulator.mujoco import mujoco as mujoco_module
from holosoma.simulator.mujoco.backends import classic_backend as classic_backend_module
from holosoma.simulator.mujoco.backends.base import apply_sensor_scene_flags
from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend
from holosoma.simulator.mujoco.mujoco import MuJoCo
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

pytestmark = pytest.mark.no_sim


class _Renderer:
    def __init__(self, *, fail: bool = False, events: list[str] | None = None, label: str = "") -> None:
        self.closed = 0
        self.fail = fail
        self.events = events
        self.label = label

    def close(self) -> None:
        self.closed += 1
        if self.events is not None:
            self.events.append(self.label)
        if self.fail:
            raise RuntimeError("close failed")


def _mujoco_simulator() -> MuJoCo:
    simulator_config = SimpleNamespace(
        debug_viz=False,
        sim=SimpleNamespace(kinematic_playback=False, fps=200.0, control_decimation_steps=4),
    )
    tyro_config = SimpleNamespace(
        training=SimpleNamespace(num_envs=1),
        simulator=simulator_config,
        scene=SimpleNamespace(),
        sensors={},
        robot=SimpleNamespace(),
        logger=SimpleNamespace(video=VideoConfig(enabled=False), headless_recording=False),
        plugin={},
        experiment_dir=None,
    )
    return MuJoCo(tyro_config, terrain_manager=SimpleNamespace(), device="cpu")  # type: ignore[arg-type]


def test_mujoco_closes_viewer_before_backend_once(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = _mujoco_simulator()
    events: list[str] = []
    backend = _Renderer(events=events, label="backend")
    viewer = _Renderer(events=events, label="viewer")
    simulator.backend = backend  # type: ignore[assignment]
    simulator.headless = False
    simulator.root_model = object()
    simulator.root_data = object()
    simulator.debug_viz_enabled = False
    viewer.opt = object()  # type: ignore[attr-defined]
    monkeypatch.setattr(mujoco_module, "ViewerInputController", lambda _sim: SimpleNamespace(on_key=None))
    monkeypatch.setattr(mujoco_module.mujoco.viewer, "launch_passive", lambda *_args, **_kwargs: viewer)
    monkeypatch.setattr(mujoco_module, "apply_sensor_scene_flags", lambda *_args, **_kwargs: None)

    simulator.setup_viewer()
    simulator.close()
    simulator.close()

    assert events == ["viewer", "backend"]
    assert viewer.closed == 1
    assert backend.closed == 1
    assert cast("object", simulator.backend) is backend


def test_classic_backend_closes_every_renderer_and_is_idempotent() -> None:
    backend = object.__new__(ClassicBackend)
    backend._closed = False
    rgb = _Renderer()
    depth = _Renderer()
    backend._cam_ids = {"head": 1}
    backend._mj_renderers = {"head": {"rgb": rgb, "depth": depth}}

    backend.close()
    backend.close()

    assert rgb.closed == 1
    assert depth.closed == 1
    assert backend._mj_renderers == {}
    assert backend._cam_ids == {}


def test_classic_backend_continues_closing_after_renderer_failure() -> None:
    backend = object.__new__(ClassicBackend)
    backend._closed = False
    failed = _Renderer(fail=True)
    healthy = _Renderer()
    backend._cam_ids = {"head": 1}
    backend._mj_renderers = {"head": {"rgb": failed, "depth": healthy}}

    with pytest.raises(RuntimeError, match="head/rgb"):
        backend.close()

    assert failed.closed == 1
    assert healthy.closed == 1
    assert backend._mj_renderers == {}


@pytest.mark.mujoco_classic
@pytest.mark.parametrize(
    ("initial_size", "camera_sizes", "expected_size"),
    [
        pytest.param(
            (640, 480),
            ((1920, 720), (1280, 1080)),
            (1920, 1080),
            id="largest-dimensions-across-cameras",
        ),
        pytest.param(
            (2560, 480),
            ((1920, 1080),),
            (2560, 1080),
            id="preserve-larger-existing-width",
        ),
        pytest.param(
            (640, 1440),
            ((1920, 1080),),
            (1920, 1440),
            id="preserve-larger-existing-height",
        ),
    ],
)
def test_classic_backend_sizes_framebuffer_for_mounted_cameras(
    monkeypatch: pytest.MonkeyPatch,
    initial_size: tuple[int, int],
    camera_sizes: tuple[tuple[int, int], ...],
    expected_size: tuple[int, int],
) -> None:
    created_sizes: list[tuple[int, int]] = []

    class _CameraRenderer:
        def __init__(self, _model: Any, *, height: int, width: int) -> None:
            created_sizes.append((width, height))

        def enable_depth_rendering(self) -> None:
            return None

    backend = object.__new__(ClassicBackend)
    backend._closed = False
    backend.model = SimpleNamespace(
        vis=SimpleNamespace(
            global_=SimpleNamespace(offwidth=initial_size[0], offheight=initial_size[1]),
        ),
    )
    backend._cam_ids = {}
    backend._mj_renderers = {}
    monkeypatch.setattr(classic_backend_module.mujoco, "mj_name2id", lambda *_args: 7)
    monkeypatch.setattr(classic_backend_module.mujoco, "Renderer", _CameraRenderer)
    cameras = [
        SimpleNamespace(
            name=f"camera_{index}",
            config=SimpleNamespace(width=width, height=height, data_types=["depth"]),
        )
        for index, (width, height) in enumerate(camera_sizes)
    ]

    backend.create_renderers([cast("Any", camera) for camera in cameras])

    assert (backend.model.vis.global_.offwidth, backend.model.vis.global_.offheight) == expected_size
    assert created_sizes == list(camera_sizes)


def test_video_cleanup_closes_renderer_even_if_shared_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = object.__new__(MuJoCoVideoRecorder)
    renderer = _Renderer()
    recorder._renderer = renderer
    recorder._camera = object()
    recorder.recording_thread = None

    def _fail_cleanup(_self: Any) -> None:
        raise RuntimeError("shared cleanup failed")

    monkeypatch.setattr(VideoRecorderInterface, "cleanup", _fail_cleanup)
    with pytest.raises(RuntimeError, match=r"shared cleanup failed"):
        recorder.cleanup()

    assert renderer.closed == 1
    assert recorder._renderer is None
    assert recorder._camera is None  # type: ignore[unreachable]  # cleanup mutates the narrowed attribute.


def test_sensor_scene_flags_show_robot_visual_group_only() -> None:
    option = apply_sensor_scene_flags(False)

    assert [index for index, enabled in enumerate(option.geomgroup) if enabled] == [0, 1, 2, 4]


def test_classic_camera_render_uses_configured_sensor_scene_option() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    render_data = object()
    scene_option = object()
    update_calls: list[tuple[Any, int, Any]] = []
    buffers: dict[str, Any] = {}

    class _CameraRenderer:
        def update_scene(self, data: Any, *, camera: int, scene_option: Any) -> None:
            update_calls.append((data, camera, scene_option))

        def render(self) -> Any:
            return frame

    backend = object.__new__(ClassicBackend)
    backend.model = SimpleNamespace(
        vis=SimpleNamespace(map=SimpleNamespace(zfar=100.0)),
        stat=SimpleNamespace(extent=1.0),
    )
    backend.data = render_data
    backend.device = "cpu"
    backend._cam_ids = {"head": 7}
    backend._mj_renderers = {"head": {"rgb": _CameraRenderer()}}
    backend._sensor_scene_option = scene_option

    def _set_buffer(data_type: str, value: Any) -> None:
        buffers[data_type] = value

    camera = SimpleNamespace(
        name="head",
        config=SimpleNamespace(data_types=["rgb"]),
        set_buffer=_set_buffer,
    )

    backend.render_cameras(cast("Any", [camera]))

    assert update_calls == [(render_data, 7, scene_option)]
    assert buffers["rgb"].shape == (1, 2, 3, 3)
    assert np.array_equal(buffers["rgb"].numpy()[0], frame)


def test_warp_camera_context_enables_literal_sensor_geom_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    warp_backend = importlib.import_module("holosoma.simulator.mujoco.backends.warp_backend")

    context = object()
    context_calls: list[tuple[Any, dict[str, Any]]] = []

    class _ScopedDevice:
        def __init__(self, device: Any) -> None:
            self.device = device

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    def _create_render_context(model: Any, **kwargs: Any) -> Any:
        context_calls.append((model, kwargs))
        return context

    monkeypatch.setattr(warp_backend.mujoco, "mj_name2id", lambda *_args: 0)
    monkeypatch.setattr(warp_backend.mjw, "create_render_context", _create_render_context)
    monkeypatch.setattr(warp_backend.wp, "ScopedDevice", _ScopedDevice)
    monkeypatch.setattr(warp_backend.wp, "zeros", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        warp_backend.wp,
        "get_stream",
        lambda _device: SimpleNamespace(cuda_stream=object()),
    )
    monkeypatch.setattr(warp_backend.torch.cuda, "ExternalStream", lambda _stream: object())

    backend = object.__new__(warp_backend.WarpBackend)
    backend._closed = False
    backend.model = SimpleNamespace(ncam=1)
    backend.num_envs = 2
    backend.mjw_device = object()
    camera = SimpleNamespace(
        name="head",
        config=SimpleNamespace(
            data_types=["rgb"],
            height=2,
            width=3,
            far=100.0,
            mujoco=MujocoCameraConfig(),
        ),
    )

    backend.create_renderers([camera])

    assert len(context_calls) == 1
    assert context_calls[0][0] is backend.model
    assert context_calls[0][1]["enabled_geom_groups"] == [0, 1, 2, 4]
    assert backend._render_context is context


def test_video_capture_uses_configured_scene_option() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frames: list[Any] = []
    scene_options: list[Any] = []

    class _CaptureRenderer:
        def update_scene(self, *_args: Any, **kwargs: Any) -> None:
            scene_options.append(kwargs["scene_option"])

        def render(self) -> Any:
            return frame

    recorder = cast("Any", object.__new__(MuJoCoVideoRecorder))
    recorder.simulator = SimpleNamespace(
        backend=SimpleNamespace(get_render_data=lambda world_id=0: SimpleNamespace(world_id=world_id))
    )
    recorder._renderer = _CaptureRenderer()
    recorder._camera = object()
    recorder.scene_option = object()
    recorder._update_camera_position = lambda _camera: None
    recorder._apply_command_overlay = lambda image: image
    recorder._add_frame = frames.append

    recorder._capture_frame_impl()

    assert scene_options == [recorder.scene_option]
    assert len(frames) == 1
    assert frames[0] is frame
