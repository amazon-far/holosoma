# mypy: disable-error-code="arg-type,assignment,attr-defined,name-defined,no-untyped-def,var-annotated"
"""Focused tests for the native MuJoCo LiDAR ray-query wrapper."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from holosoma.config_types.sensor import (  # noqa: E402
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    LidarBodyFilterConfig,
    LidarSensorConfig,
    MujocoLidarConfig,
    SensorMountConfig,
)
from holosoma.simulator.mujoco import sensor_setup  # noqa: E402
from holosoma.simulator.mujoco.geom_groups import (  # noqa: E402
    ROBOT_COLLISION_GEOM_GROUP,
    ROBOT_VISUAL_GEOM_GROUP,
)
from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager  # noqa: E402
from holosoma.simulator.mujoco.sensor_setup import (  # noqa: E402
    _cast_cpu,
    _effective_geom_groups,
    _render_warp,
    _resolve_body_exclude,
    _warp_geom_groups,
)
from holosoma.simulator.shared.sensor_manager import SensorManager  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.mujoco


def _box_model(*, group: int = 0) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
          <geom name="box" type="box" group="{group}" pos="2 0 0" size="0.1 0.2 0.3"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _robot_filter_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <asset>
            <material name="shared" rgba="0.3 0.4 0.5 1"/>
          </asset>
          <worldbody>
            <site name="holosoma_lidar_mount_scan"/>
            <site name="holosoma_lidar_mount_unfiltered"/>
            <geom name="disabled_x" type="box" group="0" pos="0.25 0 0" size="0.05 0.05 0.05"/>
            <geom name="disabled_y" type="box" group="0" pos="0 0.25 0" size="0.05 0.05 0.05"/>
            <body name="unit_torso_link">
              <geom name="robot_x" type="box" group="1" material="shared"
                    pos="0.5 0 0" size="0.1 0.1 0.1"/>
              <body pos="0 0.5 0">
                <joint name="unnamed_body_joint" type="hinge" axis="0 0 1"/>
                <geom name="robot_y" type="box" group="3" material="shared"
                      size="0.1 0.1 0.1"/>
              </body>
            </body>
            <body name="robot_decoy_x">
              <geom name="target_x" type="box" group="1" material="shared"
                    pos="2 0 0" size="0.1 0.1 0.1"/>
            </body>
            <body name="scene_y">
              <geom name="target_y" type="box" group="3" material="shared"
                    pos="0 2 0" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    scene_manager.robot_spec_meta = ActorSpecMeta(prefix="unit_", root_body="torso_link")
    model = scene_manager.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_mujoco_uses_warp_device_clock_instead_of_stale_render_snapshot() -> None:
    backend = SimpleNamespace(physics_time=lambda: 1.25)
    sim = object.__new__(MuJoCo)
    sim.backend = backend
    sim.root_data = SimpleNamespace(time=0.0)

    assert sim._physics_time() == pytest.approx(1.25)


def test_warp_geom_groups_use_the_public_six_component_warp_vector() -> None:
    vector_requests = []

    def vector(*, length: int, dtype):
        vector_requests.append((length, dtype))
        return lambda *components: components

    class PublicWarp:
        float32 = object()
        types = SimpleNamespace(vector=vector)

    record = SimpleNamespace(
        name="scan",
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            body_filter=LidarBodyFilterConfig(target_kind="robot"),
            mujoco=MujocoLidarConfig(geom_groups=[False, True, False, True, True, True]),
        ),
    )
    assert _warp_geom_groups(record, PublicWarp) == (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert vector_requests == [(6, PublicWarp.float32)]


def test_effective_geom_groups_compact_robot_categories_and_preserve_all_false() -> None:
    configured = MujocoLidarConfig(geom_groups=[False, True, False, True, False, False])
    world_mount = SensorMountConfig(target_kind="world")

    unfiltered = LidarSensorConfig(
        mount=world_mount,
        body_filter=LidarBodyFilterConfig(target_kind="none"),
        mujoco=configured,
    )
    explicit = LidarSensorConfig(
        mount=world_mount,
        body_filter=LidarBodyFilterConfig(target_kind="actor", target="target"),
        mujoco=configured,
    )
    robot = LidarSensorConfig(
        mount=world_mount,
        body_filter=LidarBodyFilterConfig(target_kind="robot"),
        mujoco=configured,
    )
    all_false = LidarSensorConfig(
        mount=world_mount,
        body_filter=LidarBodyFilterConfig(target_kind="none"),
        mujoco=MujocoLidarConfig(geom_groups=[False] * 6),
    )

    assert _effective_geom_groups(unfiltered) == (False, True, False, True, True, True)
    assert _effective_geom_groups(explicit) == (False, True, False, True, True, True)
    assert _effective_geom_groups(robot) == (False, True, False, True, False, False)
    assert _effective_geom_groups(all_false) == (False,) * 6


def test_scene_manager_compile_tags_robot_groups_after_inertia_compilation() -> None:
    def build_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.inertiagrouprange = [1, 1]
        robot = spec.worldbody.add_body(name="unit_torso_link")
        robot.add_freejoint()
        robot.add_geom(
            name="robot_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=1,
            size=[0.2, 0.1, 0.1],
            density=400.0,
        )
        robot.add_geom(
            name="robot_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            group=3,
            size=[0.3, 0.0, 0.0],
            density=4000.0,
        )
        scene = spec.worldbody.add_body(name="scene_body")
        scene.add_geom(
            name="scene_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=2,
            size=[0.1, 0.1, 0.1],
        )
        return spec

    baseline_model = build_spec().compile()
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = build_spec()
    scene_manager.robot_spec_meta = ActorSpecMeta(prefix="unit_", root_body="torso_link")

    tagged_model = scene_manager.compile()

    assert tagged_model.geom_group[tagged_model.geom("robot_visual").id] == ROBOT_VISUAL_GEOM_GROUP
    assert tagged_model.geom_group[tagged_model.geom("robot_collision").id] == ROBOT_COLLISION_GEOM_GROUP
    assert tagged_model.geom_group[tagged_model.geom("scene_visual").id] == 2
    tagged_robot_body = tagged_model.body("unit_torso_link").id
    baseline_robot_body = baseline_model.body("unit_torso_link").id
    assert tagged_model.body_mass[tagged_robot_body] == baseline_model.body_mass[baseline_robot_body]
    np.testing.assert_array_equal(
        tagged_model.body_inertia[tagged_robot_body],
        baseline_model.body_inertia[baseline_robot_body],
    )


def test_scene_manager_compile_rejects_nonrobot_reserved_geom_groups() -> None:
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    scene_manager.world_spec.worldbody.add_geom(
        name="scene_reserved",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=ROBOT_VISUAL_GEOM_GROUP,
        size=[0.1, 0.1, 0.1],
    )
    scene_manager.robot_spec_meta = None

    with pytest.raises(ValueError, match=r"reserved.*nonrobot geometry"):
        scene_manager.compile()


def test_mjwarp_shares_one_refitted_context_across_all_due_lidars(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeWarpBackend:
        def __init__(self) -> None:
            self.prepare_calls: list[list[int]] = []

        def prepare_lidar_ray_context(self, enabled_geom_groups: list[int]) -> object:
            self.prepare_calls.append(enabled_geom_groups)
            return context

    context = object()
    backend = _FakeWarpBackend()
    first = SimpleNamespace(
        name="first",
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            body_filter=LidarBodyFilterConfig(target_kind="robot"),
            mujoco=MujocoLidarConfig(geom_groups=[True] * 6),
        ),
    )
    second = SimpleNamespace(
        name="second",
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            body_filter=LidarBodyFilterConfig(target_kind="none"),
            mujoco=MujocoLidarConfig(geom_groups=[False, True, False, True, False, False]),
        ),
    )

    class _Manager:
        lidars = [first, second]

        def collect_lidars_due(self):
            return [first, second]

    sim = SimpleNamespace(
        sensor_manager=_Manager(),
        backend=backend,
        time=lambda: 1.25,
    )
    rendered = []

    def capture_render(simulator, record, render_backend, ray_context, sim_time) -> None:
        rendered.append((record.name, render_backend, ray_context, sim_time))

    monkeypatch.setattr(sensor_setup, "WarpBackend", _FakeWarpBackend)
    monkeypatch.setattr(sensor_setup, "_render_warp", capture_render)

    sensor_setup.render_lidars(sim)

    assert backend.prepare_calls == [[0, 1, 2, 3, 4, 5]]
    assert rendered == [
        ("first", backend, context, 1.25),
        ("second", backend, context, 1.25),
    ]


def test_mjwarp_lidar_context_reuses_configured_geometry_bvh(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco_warp")
    from holosoma.simulator.mujoco.backends import warp_backend

    active_capture = None

    class _ScopedDevice:
        def __init__(self, device) -> None:
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class _Capture:
        def __init__(self) -> None:
            self.graph = SimpleNamespace(launch=None)

        def __enter__(self):
            nonlocal active_capture
            active_capture = self
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            nonlocal active_capture
            active_capture = None

    launched = []
    replayed_refits = []

    def capture_launch(graph) -> None:
        launched.append(graph)
        assert graph.launch is not None
        graph.launch()

    class _Warp:
        ScopedDevice = _ScopedDevice
        ScopedCapture = _Capture

    _Warp.capture_launch = staticmethod(capture_launch)

    created = []
    refits = []

    class _MujocoWarp:
        @staticmethod
        def create_render_context(model, **kwargs):
            context = SimpleNamespace(kwargs=kwargs)
            created.append((model, context))
            return context

        @staticmethod
        def refit_bvh(model, data, context) -> None:
            assert active_capture is not None
            refits.append((model, data, context))
            active_capture.graph.launch = lambda: replayed_refits.append((model, data, context))

    backend = object.__new__(warp_backend.WarpBackend)
    backend.mjw_device = "cuda:0"
    backend.model = SimpleNamespace(ncam=3)
    backend.num_envs = 2
    backend.mjw_model = object()
    backend.mjw_data = object()
    backend._lidar_render_context = None
    backend._lidar_render_context_groups = None
    backend._sensor_refit_graphs = {}
    backend._sensor_refit_contexts = set()
    monkeypatch.setattr(warp_backend, "wp", _Warp)
    monkeypatch.setattr(warp_backend, "mjw", _MujocoWarp)

    first = backend.prepare_lidar_ray_context([0, 1, 2])
    second = backend.prepare_lidar_ray_context([0, 1, 2])
    backend.begin_sensor_render()
    third = backend.prepare_lidar_ray_context([0, 1, 2])

    assert first is second is third
    assert len(created) == 1
    assert created[0][1].kwargs == {
        "nworld": 2,
        "enabled_geom_groups": [0, 1, 2],
        "cam_active": [False, False, False],
        "use_textures": False,
        "use_shadows": False,
    }
    assert refits == [(backend.mjw_model, backend.mjw_data, first)]
    graph = backend._sensor_refit_graphs[id(first)]
    assert launched == [graph, graph]
    assert replayed_refits == [
        (backend.mjw_model, backend.mjw_data, first),
        (backend.mjw_model, backend.mjw_data, first),
    ]


def test_mjwarp_lidar_reuses_camera_bvh_without_duplicate_same_frame_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco_warp")
    from holosoma.simulator.mujoco import sensor_setup
    from holosoma.simulator.mujoco.backends import warp_backend

    class _ScopedDevice:
        def __init__(self, device) -> None:
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    launched = []

    class _Warp:
        ScopedDevice = _ScopedDevice

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def capture_launch(graph) -> None:
            launched.append(graph)

    class _MujocoWarp:
        @staticmethod
        def create_render_context(*_args, **_kwargs):
            raise AssertionError("visual-group LiDAR should reuse the camera context")

        @staticmethod
        def refit_bvh(*_args, **_kwargs) -> None:
            raise AssertionError("camera rendering already refitted this context")

    class _Mask:
        def assign(self, _value) -> None:
            return None

    camera_context = SimpleNamespace(render_rgb=_Mask(), render_depth=_Mask())
    camera_graph = object()
    render_stream = object()
    backend = object.__new__(warp_backend.WarpBackend)
    backend.device = "cuda:0"
    backend.mjw_device = "cuda:0"
    backend.model = SimpleNamespace(ncam=1)
    backend._render_context = camera_context
    backend._render_graph = camera_graph
    backend._torch_render_stream = render_stream
    backend._cam_ids = {"camera": 0}
    backend._render_rgb_out = {}
    backend._render_depth_out = {}
    backend._lidar_render_context = None
    backend._lidar_render_context_groups = None
    backend._sensor_refit_graphs = {}
    backend._sensor_refit_contexts = {-1}

    camera = SimpleNamespace(name="camera", config=SimpleNamespace(data_types=[]))
    lidar = SimpleNamespace(
        name="lidar",
        config=LidarSensorConfig(mount=SensorMountConfig(target_kind="world")),
    )

    class _Manager:
        lidars = [lidar]

        def collect_due(self):
            return [camera]

        def collect_lidars_due(self):
            return [lidar]

    rendered = []
    sim = SimpleNamespace(
        sensor_manager=_Manager(),
        backend=backend,
        time=lambda: 1.25,
    )
    monkeypatch.setattr(warp_backend, "wp", _Warp)
    monkeypatch.setattr(warp_backend, "mjw", _MujocoWarp)
    monkeypatch.setattr(
        warp_backend.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(wait_stream=lambda _stream: None),
    )
    monkeypatch.setattr(warp_backend.torch.cuda, "stream", lambda _stream: _ScopedDevice(_stream))
    monkeypatch.setattr(
        sensor_setup,
        "_render_warp",
        lambda simulator, record, render_backend, context, sim_time: rendered.append(
            (simulator, record, render_backend, context, sim_time)
        ),
    )

    sensor_setup.render_sensors(sim)

    assert launched == [camera_graph]
    assert backend._sensor_refit_contexts == {id(camera_context)}
    assert rendered == [(sim, lidar, backend, camera_context, 1.25)]


def test_mjwarp_captures_distinct_reusable_ray_graphs_per_lidar(monkeypatch: pytest.MonkeyPatch) -> None:
    capture_count = 0
    launched_graphs = []
    ray_calls = 0

    class _ScopedDevice:
        def __init__(self, device) -> None:
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class _Capture:
        def __init__(self) -> None:
            self.graph = object()

        def __enter__(self):
            nonlocal capture_count
            capture_count += 1
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    fake_warp = ModuleType("warp")
    fake_warp.ScopedDevice = _ScopedDevice
    fake_warp.ScopedCapture = _Capture
    fake_warp.to_torch = lambda tensor: tensor

    def from_torch(tensor, dtype=None):
        assert dtype in (None, fake_warp.vec3)
        return tensor

    fake_warp.from_torch = from_torch
    fake_warp.vec3 = object()
    fake_warp.float32 = object()

    def vector(*, length: int, dtype):
        assert length == 6
        assert dtype is fake_warp.float32
        return lambda *components: components

    fake_warp.types = SimpleNamespace(vector=vector)
    fake_warp.capture_launch = lambda graph: launched_graphs.append(graph)
    fake_warp.get_stream = lambda _device: SimpleNamespace(cuda_stream=17)

    fake_mujoco_warp = ModuleType("mujoco_warp")

    def rays(*args, **kwargs) -> None:
        nonlocal ray_calls
        ray_calls += 1

    fake_mujoco_warp.rays = rays
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setitem(sys.modules, "mujoco_warp", fake_mujoco_warp)
    external_streams = []
    consumer_streams = []
    producer_waits = []
    waited_streams = []

    class _ExternalStream:
        def wait_stream(self, stream) -> None:
            producer_waits.append((self, stream))

    class _ConsumerStream:
        def wait_stream(self, stream) -> None:
            waited_streams.append(stream)

    def external_stream(cuda_stream):
        assert cuda_stream == 17
        stream = _ExternalStream()
        external_streams.append(stream)
        return stream

    def current_stream(_device):
        stream = _ConsumerStream()
        consumer_streams.append(stream)
        return stream

    monkeypatch.setattr(sensor_setup.torch.cuda, "ExternalStream", external_stream)
    monkeypatch.setattr(sensor_setup.torch.cuda, "stream", lambda stream: _ScopedDevice(stream))
    monkeypatch.setattr(sensor_setup.torch.cuda, "current_stream", current_stream)

    first = SimpleNamespace(
        name="first",
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            mujoco=MujocoLidarConfig(geom_groups=[True, False, False, False, False, False]),
        ),
        backend_cache={
            "mujoco_site_id": 0,
            "mujoco_body_exclude_id": 4,
        },
        pattern_at=lambda _: SimpleNamespace(directions=torch.tensor([[0.0, 0.0, -1.0]])),
    )
    second = SimpleNamespace(
        name="second",
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            mujoco=MujocoLidarConfig(geom_groups=[False, True, False, False, False, False]),
        ),
        backend_cache={
            "mujoco_site_id": 1,
            "mujoco_body_exclude_id": 7,
        },
        pattern_at=lambda _: SimpleNamespace(directions=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])),
    )
    backend = SimpleNamespace(
        mjw_device="cpu",
        mjw_model=object(),
        mjw_data=SimpleNamespace(
            site_xpos=torch.zeros((1, 2, 3)),
            site_xmat=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, -1, -1),
        ),
    )
    sim = SimpleNamespace(
        root_model=SimpleNamespace(site_bodyid=np.array([1, 2])),
        num_envs=1,
        sim_device="cpu",
        env_origins=torch.zeros((1, 3)),
    )
    monkeypatch.setattr(sensor_setup, "finalize_lidar_returns", lambda *_args, **_kwargs: None)

    sensor_setup._render_warp(sim, first, backend, object(), 0.0)
    sensor_setup._render_warp(sim, second, backend, object(), 0.0)
    first_workspace = first.backend_cache["mujoco_warp_rays"]
    second_workspace = second.backend_cache["mujoco_warp_rays"]
    assert first_workspace is not second_workspace
    assert first_workspace.directions.shape == (1, 1, 3)
    assert second_workspace.directions.shape == (1, 2, 3)
    assert first_workspace.geom_group == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert second_workspace.geom_group == (0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    torch.testing.assert_close(first_workspace.excludes, torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(second_workspace.excludes, torch.tensor([7, 7], dtype=torch.int32))
    assert first_workspace.graph is not second_workspace.graph

    sensor_setup._render_warp(sim, first, backend, object(), 0.1)
    sensor_setup._render_warp(sim, second, backend, object(), 0.1)

    assert capture_count == 2
    assert ray_calls == 2
    assert launched_graphs == [
        first_workspace.graph,
        second_workspace.graph,
        first_workspace.graph,
        second_workspace.graph,
    ]
    assert external_streams == [first_workspace.torch_stream, second_workspace.torch_stream]
    assert producer_waits == [
        (first_workspace.torch_stream, consumer_streams[0]),
        (second_workspace.torch_stream, consumer_streams[1]),
        (first_workspace.torch_stream, consumer_streams[2]),
        (second_workspace.torch_stream, consumer_streams[3]),
    ]
    assert waited_streams == [
        first_workspace.torch_stream,
        second_workspace.torch_stream,
        first_workspace.torch_stream,
        second_workspace.torch_stream,
    ]


def test_mjwarp_replayed_graph_reads_current_directions_and_updates_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured ray graph must consume current buffers, not directions from its capture tick."""
    active_capture = None
    capture_count = 0
    ray_calls = 0

    class _ScopedDevice:
        def __init__(self, device) -> None:
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class _Graph:
        launch = None

    class _Capture:
        def __init__(self) -> None:
            self.graph = _Graph()

        def __enter__(self):
            nonlocal active_capture, capture_count
            capture_count += 1
            active_capture = self
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            nonlocal active_capture
            active_capture = None

    fake_warp = ModuleType("warp")
    fake_warp.ScopedDevice = _ScopedDevice
    fake_warp.ScopedCapture = _Capture
    fake_warp.to_torch = lambda tensor: tensor

    def from_torch(tensor, dtype=None):
        assert dtype in (None, fake_warp.vec3)
        return tensor

    fake_warp.from_torch = from_torch
    fake_warp.vec3 = object()
    fake_warp.float32 = object()

    def vector(*, length, dtype):
        assert length == 6
        assert dtype is fake_warp.float32
        return lambda *components: components

    fake_warp.types = SimpleNamespace(vector=vector)
    fake_warp.capture_launch = lambda graph: graph.launch()
    fake_warp.get_stream = lambda _device: SimpleNamespace(cuda_stream=23)

    fake_mujoco_warp = ModuleType("mujoco_warp")

    def rays(
        _model,
        _data,
        _origins,
        directions,
        _geom_group,
        _include_static,
        _excludes,
        distances,
        geom_ids,
        _normals,
        *,
        rc,
    ) -> None:
        nonlocal ray_calls
        ray_calls += 1
        assert active_capture is not None

        def launch() -> None:
            distances.copy_(2.0 + directions[..., 0])
            geom_ids.fill_(7)

        active_capture.graph.launch = launch

    fake_mujoco_warp.rays = rays
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setitem(sys.modules, "mujoco_warp", fake_mujoco_warp)
    external_streams = []
    consumer_streams = []
    producer_waits = []
    waited_streams = []

    class _ExternalStream:
        def wait_stream(self, stream) -> None:
            producer_waits.append((self, stream))

    class _ConsumerStream:
        def wait_stream(self, stream) -> None:
            waited_streams.append(stream)

    def external_stream(cuda_stream):
        assert cuda_stream == 23
        stream = _ExternalStream()
        external_streams.append(stream)
        return stream

    def current_stream(_device):
        stream = _ConsumerStream()
        consumer_streams.append(stream)
        return stream

    monkeypatch.setattr(sensor_setup.torch.cuda, "ExternalStream", external_stream)
    monkeypatch.setattr(sensor_setup.torch.cuda, "stream", lambda stream: _ScopedDevice(stream))
    monkeypatch.setattr(sensor_setup.torch.cuda, "current_stream", current_stream)

    class _Record:
        name = "moving_pattern"
        config = LidarSensorConfig(mount=SensorMountConfig(target_kind="world"))

        def __init__(self) -> None:
            self.backend_cache = {
                "mujoco_site_id": 0,
                "mujoco_body_exclude_id": -1,
            }
            self.buffers = {}

        def pattern_at(self, sim_time: float):
            direction = [0.0, 0.0, -1.0] if sim_time < 0.5 else [1.0, 0.0, 0.0]
            return SimpleNamespace(directions=torch.tensor([direction], dtype=torch.float32))

        def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
            self.buffers[output] = tensor.clone()

    record = _Record()
    backend = SimpleNamespace(
        mjw_device="cpu",
        mjw_model=object(),
        mjw_data=SimpleNamespace(
            site_xpos=torch.zeros((1, 1, 3)),
            site_xmat=torch.eye(3).reshape(1, 1, 3, 3),
        ),
    )
    sim = SimpleNamespace(
        root_model=SimpleNamespace(site_bodyid=np.array([0])),
        num_envs=1,
        sim_device="cpu",
        env_origins=torch.zeros((1, 3)),
    )

    sensor_setup._render_warp(sim, record, backend, object(), 0.0)
    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[2.0]]))
    torch.testing.assert_close(record.buffers["points"], torch.tensor([[[0.0, 0.0, -2.0]]]))

    sensor_setup._render_warp(sim, record, backend, object(), 1.0)
    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[3.0]]))
    torch.testing.assert_close(record.buffers["points"], torch.tensor([[[3.0, 0.0, 0.0]]]))
    assert record.buffers["geom_ids"].item() == 7
    assert capture_count == 1
    assert ray_calls == 1
    workspace = record.backend_cache["mujoco_warp_rays"]
    assert external_streams == [workspace.torch_stream]
    assert producer_waits == [
        (workspace.torch_stream, consumer_streams[0]),
        (workspace.torch_stream, consumer_streams[1]),
    ]
    assert waited_streams == [workspace.torch_stream, workspace.torch_stream]


def test_multi_ray_hit_miss_and_cutoff() -> None:
    model, data = _box_model()
    distances, geom_ids = _cast_cpu(
        model,
        data,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        geom_groups=np.ones(6, dtype=np.uint8),
        include_static=True,
        body_exclude=-1,
        far=10.0,
    )
    assert distances[0] == pytest.approx(1.9)
    assert geom_ids[0] == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    assert distances[1] == -1.0
    assert geom_ids[1] == -1

    clipped, _ = _cast_cpu(
        model,
        data,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0]]),
        geom_groups=np.ones(6, dtype=np.uint8),
        include_static=True,
        body_exclude=-1,
        far=1.0,
    )
    assert clipped[0] == -1.0


def test_geom_group_filter_excludes_matching_geometry() -> None:
    model, data = _box_model(group=1)
    distances, geom_ids = _cast_cpu(
        model,
        data,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0]]),
        geom_groups=np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        include_static=True,
        body_exclude=-1,
        far=10.0,
    )

    assert distances[0] == -1.0
    assert geom_ids[0] == -1


def test_body_filter_recovers_return_behind_enclosing_body() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="sensor_shell">
              <geom name="shell" type="box" size="0.25 0.25 0.25"/>
            </body>
            <body name="target_body" pos="2 0 0">
              <geom name="target" type="box" size="0.1 0.2 0.3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    direction = np.array([[1.0, 0.0, 0.0]])
    kwargs = {
        "geom_groups": np.ones(6, dtype=np.uint8),
        "include_static": True,
        "far": 10.0,
    }

    unfiltered_distances, unfiltered_geom_ids = _cast_cpu(
        model,
        data,
        np.zeros(3),
        direction,
        body_exclude=-1,
        **kwargs,
    )
    filtered_distances, filtered_geom_ids = _cast_cpu(
        model,
        data,
        np.zeros(3),
        direction,
        body_exclude=model.body("sensor_shell").id,
        **kwargs,
    )

    assert unfiltered_distances[0] == pytest.approx(0.25)
    assert unfiltered_geom_ids[0] == model.geom("shell").id
    assert filtered_distances[0] == pytest.approx(1.9)
    assert filtered_geom_ids[0] == model.geom("target").id


def test_body_filter_resolves_mount_none_actor_and_rejects_missing_target() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="unit_torso_link">
              <site name="holosoma_lidar_mount_scan"/>
            </body>
            <body name="target_body">
              <site name="holosoma_lidar_filter_scan"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    site_id = model.site("holosoma_lidar_mount_scan").id
    sim = SimpleNamespace(root_model=model)
    mount = SensorMountConfig(target_kind="robot_link", target="torso_link")

    assert (
        _resolve_body_exclude(sim, "scan", LidarSensorConfig(mount=mount), site_id) == model.body("unit_torso_link").id
    )
    assert (
        _resolve_body_exclude(
            sim,
            "scan",
            LidarSensorConfig(mount=mount, body_filter=LidarBodyFilterConfig(target_kind="none")),
            site_id,
        )
        == -1
    )
    assert (
        _resolve_body_exclude(
            sim,
            "scan",
            LidarSensorConfig(
                mount=mount,
                body_filter=LidarBodyFilterConfig(target_kind="actor", target="target"),
            ),
            site_id,
        )
        == model.body("target_body").id
    )
    with pytest.raises(RuntimeError, match="marker"):
        _resolve_body_exclude(
            sim,
            "missing",
            LidarSensorConfig(
                mount=mount,
                body_filter=LidarBodyFilterConfig(target_kind="actor", target="missing"),
            ),
            site_id,
        )


def test_registered_robot_filter_excludes_reserved_groups_and_none_preserves_authored_groups() -> None:
    model, data = _robot_filter_model()
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        near=0.01,
        far=10.0,
        body_filter=LidarBodyFilterConfig(target_kind="robot"),
        mujoco=MujocoLidarConfig(geom_groups=[False, True, False, True, False, False]),
    )
    unfiltered = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        near=0.01,
        far=10.0,
        body_filter=LidarBodyFilterConfig(target_kind="none"),
        mujoco=MujocoLidarConfig(geom_groups=[False, True, False, True, False, False]),
    )
    manager = SensorManager("cpu", control_hz=10.0)
    sim = SimpleNamespace(
        root_model=model,
        sensor_config={"scan": lidar, "unfiltered": unfiltered},
        sensor_manager=manager,
        backend=SimpleNamespace(get_render_data=lambda world_id=0: data if world_id == 0 else None),
        sim_device="cpu",
        num_envs=1,
        env_origins=torch.zeros((1, 3)),
        time=lambda: 0.0,
    )
    original_matid = model.geom_matid.copy()
    original_rgba = model.geom_rgba.copy()

    sensor_setup.register_lidars(sim, manager)
    sensor_setup.render_lidars(sim)

    record = manager.get_lidar("scan")
    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[1.9, 1.9]], dtype=torch.float32))
    torch.testing.assert_close(
        record.buffers["geom_ids"],
        torch.tensor([[model.geom("target_x").id, model.geom("target_y").id]], dtype=torch.int32),
    )
    unfiltered_record = manager.get_lidar("unfiltered")
    torch.testing.assert_close(
        unfiltered_record.buffers["ranges"],
        torch.tensor([[0.4, 0.4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        unfiltered_record.buffers["geom_ids"],
        torch.tensor([[model.geom("robot_x").id, model.geom("robot_y").id]], dtype=torch.int32),
    )
    assert record.backend_cache["mujoco_body_exclude_id"] == -1
    assert "mujoco_robot_geom_ids" not in record.backend_cache
    np.testing.assert_array_equal(model.geom_matid, original_matid)
    np.testing.assert_array_equal(model.geom_rgba, original_rgba)


def test_registered_explicit_body_filter_renders_target_behind_shell() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="unit_torso_link">
              <site name="holosoma_lidar_mount_scan"/>
              <body name="unit_head_link">
                <site name="holosoma_lidar_filter_scan"/>
                <geom name="shell" type="box" size="0.25 0.25 0.25"/>
              </body>
            </body>
            <body name="target_body" pos="2 0 0">
              <geom name="target" type="box" size="0.1 0.2 0.3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="torso_link"),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[1.0, 0.0, 0.0]]),
        near=0.01,
        far=10.0,
        body_filter=LidarBodyFilterConfig(target_kind="robot_link", target="head_link"),
    )
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.robot_spec_meta = ActorSpecMeta(
        prefix="unit_",
        root_body="torso_link",
        body_names=["head_link"],
    )
    manager = SensorManager("cpu", control_hz=10.0)
    sim = SimpleNamespace(
        root_model=model,
        sensor_config={"scan": lidar},
        scene_manager=scene_manager,
        sensor_manager=manager,
        backend=SimpleNamespace(get_render_data=lambda world_id=0: data if world_id == 0 else None),
        sim_device="cpu",
        num_envs=1,
        time=lambda: 0.0,
    )

    sensor_setup.register_lidars(sim, manager)
    sensor_setup.render_lidars(sim)

    record = manager.get_lidar("scan")
    assert record.backend_cache["mujoco_body_exclude_id"] == model.body("unit_head_link").id
    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[1.9]], dtype=torch.float32))
    torch.testing.assert_close(record.buffers["points"], torch.tensor([[[1.9, 0.0, 0.0]]]))
    torch.testing.assert_close(record.buffers["geom_ids"], torch.tensor([[model.geom("target").id]], dtype=torch.int32))


@pytest.mark.mujoco_warp
def test_mjwarp_body_filter_recovers_return_behind_enclosing_body() -> None:
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="sensor_shell">
              <site name="holosoma_lidar_mount_scan"/>
              <geom name="shell" type="box" size="0.25 0.25 0.25"/>
            </body>
            <body name="target_body" pos="2 0 0">
              <geom name="target" type="box" size="0.1 0.2 0.3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wp.init()
    device = wp.get_device()
    with wp.ScopedDevice(device):
        backend = SimpleNamespace(
            mjw_device=device,
            mjw_model=mjw.put_model(model),
            mjw_data=mjw.put_data(model, data),
        )
        ray_context = mjw.create_render_context(model, enabled_geom_groups=[0, 1, 2])

    device_name = str(device)
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="sensor_shell"),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[1.0, 0.0, 0.0]]),
        near=0.01,
        far=10.0,
    )
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.robot_spec_meta = ActorSpecMeta(prefix="", root_body="sensor_shell", body_names=[])
    manager = SensorManager(device_name, control_hz=10.0)
    sim = SimpleNamespace(
        root_model=model,
        sensor_config={"scan": lidar},
        scene_manager=scene_manager,
        sensor_manager=manager,
        backend=backend,
        sim_device=device_name,
        num_envs=1,
        env_origins=torch.zeros((1, 3), device=device_name),
    )
    sensor_setup.register_lidars(sim, manager)
    record = manager.get_lidar("scan")
    _render_warp(sim, record, backend, ray_context, 0.0)

    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[1.9]], device=device_name))
    torch.testing.assert_close(
        record.buffers["points"],
        torch.tensor([[[1.9, 0.0, 0.0]]], device=device_name),
    )
    torch.testing.assert_close(
        record.buffers["geom_ids"],
        torch.tensor([[model.geom("target").id]], dtype=torch.int32, device=device_name),
    )


@pytest.mark.mujoco_warp
def test_mjwarp_robot_filter_excludes_reserved_groups_and_preserves_scene_groups() -> None:
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    model, data = _robot_filter_model()
    wp.init()
    device = wp.get_device()
    with wp.ScopedDevice(device):
        backend = SimpleNamespace(
            mjw_device=device,
            mjw_model=mjw.put_model(model),
            mjw_data=mjw.put_data(model, data),
        )
        ray_context = mjw.create_render_context(model, enabled_geom_groups=[1, 3])

    device_name = str(device)
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        near=0.01,
        far=10.0,
        body_filter=LidarBodyFilterConfig(target_kind="robot"),
        mujoco=MujocoLidarConfig(geom_groups=[False, True, False, True, False, False]),
    )
    manager = SensorManager(device_name, control_hz=10.0)
    sim = SimpleNamespace(
        root_model=model,
        sensor_config={"scan": lidar},
        sensor_manager=manager,
        backend=backend,
        sim_device=device_name,
        num_envs=1,
        env_origins=torch.zeros((1, 3), device=device_name),
    )
    original_matid = wp.to_torch(backend.mjw_model.geom_matid).clone()
    original_rgba = wp.to_torch(backend.mjw_model.geom_rgba).clone()

    sensor_setup.register_lidars(sim, manager)
    record = manager.get_lidar("scan")
    _render_warp(sim, record, backend, ray_context, 0.0)
    captured_graph = record.backend_cache["mujoco_warp_rays"].graph
    _render_warp(sim, record, backend, ray_context, 0.1)

    assert record.backend_cache["mujoco_warp_rays"].geom_group == (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    torch.testing.assert_close(record.buffers["ranges"], torch.tensor([[1.9, 1.9]], device=device_name))
    torch.testing.assert_close(
        record.buffers["geom_ids"],
        torch.tensor(
            [[model.geom("target_x").id, model.geom("target_y").id]],
            dtype=torch.int32,
            device=device_name,
        ),
    )
    assert record.backend_cache["mujoco_warp_rays"].graph is captured_graph
    torch.testing.assert_close(wp.to_torch(backend.mjw_model.geom_matid), original_matid)
    torch.testing.assert_close(wp.to_torch(backend.mjw_model.geom_rgba), original_rgba)


def test_explicit_body_filter_resolves_non_default_robot_prefix() -> None:
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="torso_link"),
        body_filter=LidarBodyFilterConfig(target_kind="robot_link", target="head_link"),
    )
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    torso = scene_manager.world_spec.worldbody.add_body(name="unit_torso_link")
    torso.add_body(name="unit_head_link")
    scene_manager.robot_spec_meta = ActorSpecMeta(
        prefix="unit_",
        root_body="torso_link",
        body_names=["head_link"],
    )
    scene_manager.rigid_object_root_bodies = {}
    scene_manager.scene_file_bodies = {}

    scene_manager.add_lidar_sites({"scan": lidar})
    model = scene_manager.world_spec.compile()
    mount_site_id = model.site("holosoma_lidar_mount_scan").id
    filter_site_id = model.site("holosoma_lidar_filter_scan").id

    assert model.site_bodyid[mount_site_id] == model.body("unit_torso_link").id
    assert model.site_bodyid[filter_site_id] == model.body("unit_head_link").id


def test_filter_marker_follows_fixed_child_fused_into_dynamic_owner() -> None:
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="torso_link"),
        body_filter=LidarBodyFilterConfig(target_kind="robot_link", target="head_link"),
    )
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    scene_manager.world_spec.compiler.fusestatic = True
    torso = scene_manager.world_spec.worldbody.add_body(name="unit_torso_link")
    torso.add_freejoint()
    head = torso.add_body(name="unit_head_link")
    head.add_geom(name="shell", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.25, 0.25])
    target = scene_manager.world_spec.worldbody.add_body(name="target_body")
    target.pos = [2.0, 0.0, 0.0]
    target.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.2, 0.3])
    scene_manager.robot_spec_meta = ActorSpecMeta(
        prefix="unit_",
        root_body="torso_link",
        body_names=["head_link"],
    )
    scene_manager.rigid_object_root_bodies = {"target": "target_body"}
    scene_manager.scene_file_bodies = {}

    scene_manager.add_lidar_sites({"scan": lidar})
    model = scene_manager.world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mount_site_id = model.site("holosoma_lidar_mount_scan").id
    sim = SimpleNamespace(root_model=model)
    body_exclude = _resolve_body_exclude(sim, "scan", lidar, mount_site_id)

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "unit_head_link") == -1
    assert body_exclude == model.body("unit_torso_link").id
    distances, geom_ids = _cast_cpu(
        model,
        data,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0]]),
        geom_groups=np.ones(6, dtype=np.uint8),
        include_static=True,
        body_exclude=body_exclude,
        far=10.0,
    )
    assert distances[0] == pytest.approx(1.9)
    assert geom_ids[0] == model.geom("target").id


def test_sensor_mounts_use_captured_robot_prefix_for_cameras_and_lidars() -> None:
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    scene_manager.world_spec.worldbody.add_body(name="unit_torso_link")
    scene_manager.robot_spec_meta = ActorSpecMeta(prefix="unit_", root_body="torso_link")
    scene_manager.rigid_object_root_bodies = {}
    scene_manager.scene_file_bodies = {}
    mount = SensorMountConfig(target_kind="robot_link", target="torso_link")

    scene_manager.add_cameras({"view": CameraSensorConfig(mount=mount)}, robot_prefix="unit_")
    scene_manager.add_lidar_sites({"scan": LidarSensorConfig(mount=mount)})
    model = scene_manager.world_spec.compile()

    assert model.cam_bodyid[model.camera("view").id] == model.body("unit_torso_link").id
    assert model.site_bodyid[model.site("holosoma_lidar_mount_scan").id] == model.body("unit_torso_link").id

    with pytest.raises(ValueError, match="composed robot uses 'unit_'"):
        scene_manager.add_cameras({"bad": CameraSensorConfig(mount=mount)}, robot_prefix="robot_")


def test_lidar_marker_names_do_not_collide_with_filter_prefixed_sensor_names() -> None:
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    scene_manager.world_spec.worldbody.add_body(name="target_body")
    scene_manager.robot_spec_meta = None
    scene_manager.rigid_object_root_bodies = {"target": "target_body"}
    scene_manager.scene_file_bodies = {}
    world_mount = SensorMountConfig(target_kind="world")

    scene_manager.add_lidar_sites(
        {
            "scan": LidarSensorConfig(
                mount=world_mount,
                body_filter=LidarBodyFilterConfig(target_kind="actor", target="target"),
            ),
            "filter_scan": LidarSensorConfig(
                mount=world_mount,
                body_filter=LidarBodyFilterConfig(target_kind="none"),
            ),
        }
    )
    model = scene_manager.world_spec.compile()

    assert model.site("holosoma_lidar_mount_scan").id >= 0
    assert model.site("holosoma_lidar_filter_scan").id >= 0
    assert model.site("holosoma_lidar_mount_filter_scan").id >= 0


def test_explicit_filter_rejects_target_fused_into_world_body() -> None:
    scene_manager = object.__new__(MujocoSceneManager)
    scene_manager.world_spec = mujoco.MjSpec()
    scene_manager.world_spec.compiler.fusestatic = True
    scene_manager.world_spec.worldbody.add_body(name="static_target")
    scene_manager.robot_spec_meta = None
    scene_manager.rigid_object_root_bodies = {"target": "static_target"}
    scene_manager.scene_file_bodies = {}
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        body_filter=LidarBodyFilterConfig(target_kind="actor", target="target"),
    )

    scene_manager.add_lidar_sites({"scan": lidar})
    model = scene_manager.world_spec.compile()
    sim = SimpleNamespace(root_model=model)

    with pytest.raises(ValueError, match="compiled into MuJoCo's world body"):
        _resolve_body_exclude(sim, "scan", lidar, model.site("holosoma_lidar_mount_scan").id)
