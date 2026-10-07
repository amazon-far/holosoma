"""The depth-shm plugin's render thread contract (pure, no simulator, no GL).

Rendering a camera costs far more than a 500 Hz physics step's 2 ms budget, so this plugin renders on
its own thread instead of on ``FRAME_END``. Three properties keep that from silently regressing:
the camera must be taken off the control-loop schedule (or the cost returns to the physics thread),
``publish`` must stay a no-op (same reason), and the thread must be started before the sim loop and
joined before the shared memory it writes into is released.

A fake simulator stands in for the real one: these tests are about scheduling and lifecycle, not
about pixels — the numeric side is covered by ``test_depth_shm_plugin.py``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import pytest

from holosoma.config_types.plugin import DepthShmPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins import depth_shm_plugin
from holosoma.simulator.plugins.depth_shm_plugin import DepthShmPlugin
from holosoma.simulator.shared.camera_sensor import SensorManager
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.camera_sensor import CameraRuntime

pytestmark = pytest.mark.no_sim

CAM = "front_depth"
SHM_NAME = "depth_img_shm_test_threading"
RENDER_H, RENDER_W = 60, 106


class _FakeSharedMemory:
    blocks: dict[str, bytearray] = {}

    def __init__(self, *, name: str, create: bool = False, size: int = 0) -> None:
        if create:
            if name in self.blocks:
                raise FileExistsError(name)
            self.blocks[name] = bytearray(size)
        elif name not in self.blocks:
            raise FileNotFoundError(name)
        self.name = name
        self.buf = self.blocks[name]
        self.size = len(self.buf)

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        if self.name not in self.blocks:
            raise FileNotFoundError(self.name)
        del self.blocks[self.name]


@pytest.fixture(autouse=True)
def fake_shared_memory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _FakeSharedMemory.blocks.clear()
    monkeypatch.setattr(shared_memory, "SharedMemory", _FakeSharedMemory)
    yield
    _FakeSharedMemory.blocks.clear()


def _camera_config() -> CameraSensorConfig:
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
        width=RENDER_W,
        height=RENDER_H,
        data_types=["depth"],
        update_decimation=1,
    )


class _FakeBackend:
    """Counts render calls and records which thread they came from."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[int] = set()

    def render_cameras(self, cameras: list[CameraRuntime]) -> None:
        self.calls += 1
        self.threads.add(threading.get_ident())


class _FakeTrainingConfig:
    num_envs = 1


class _FakeSimulator:
    """Minimal surface DepthShmPlugin touches: hooks, sensor_manager, backend, camera reads."""

    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.sensor_manager = SensorManager("cpu", control_hz=125.0)
        self.sensor_manager.register_camera(CAM, _camera_config())
        self.backend = _FakeBackend()
        self.sensor_config = {CAM: _camera_config()}
        self.training_config = _FakeTrainingConfig()

    def get_camera_data(
        self,
        name: str,
        data_type: str = "rgb",
        env_ids: Any = None,
        device: Any = None,
    ) -> torch.Tensor:
        # Metric depth, [N, H, W, C], mid-range so it normalizes to something non-degenerate.
        return torch.full((1, RENDER_H, RENDER_W, 1), 1.5, dtype=torch.float32)

    def time(self) -> float:
        return 0.0

    def sensor_config_by_name(self, name: str) -> CameraSensorConfig:
        return self.sensor_config[name]


def _config(**overrides: Any) -> DepthShmPluginConfig:
    kwargs: dict[str, Any] = {
        "camera": CAM,
        "shm_name": SHM_NAME,
        "render_hz": 200.0,  # fast so tests do not sleep long
        "near_clip": 0.3,
        "far_clip": 3.0,
        "crop_top": 2,
        "crop_left": 4,
        "crop_right": 4,
    }
    kwargs.update(overrides)
    return DepthShmPluginConfig(**kwargs)


@pytest.fixture
def accept_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``_FakeBackend`` pass the ClassicBackend install check, which these tests are not about."""
    monkeypatch.setattr(depth_shm_plugin, "_require_classic_backend", lambda simulator, **kwargs: None)  # noqa: ARG005


@pytest.fixture
def plugin(accept_fake_backend: None) -> Iterator[tuple[DepthShmPlugin, _FakeSimulator]]:
    sim = _FakeSimulator()
    p = DepthShmPlugin(_config(), cast("BaseSimulator", sim))
    yield p, sim
    p.stop()  # idempotent; releases shm even if a test already stopped it


def test_camera_is_removed_from_the_control_loop_schedule(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """The whole point: ``collect_due`` must never return a camera this plugin renders itself."""
    _, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    # Many control steps' worth of scheduling; the claimed camera must never come due.
    assert all(sim.sensor_manager.collect_due() == [] for _ in range(50))


def test_publish_is_a_noop_so_frame_end_costs_nothing(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """``FRAME_END`` must not render or write: that would be back on the physics thread."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    p._stop.set()  # freeze the render thread so only the hook can act
    if p._thread is not None:
        p._thread.join(timeout=2.0)

    before = sim.backend.calls
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.backend.calls == before


def test_render_thread_publishes_frames_off_the_main_thread(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """Frames must be produced, and produced by the plugin's thread rather than the caller's."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    deadline = time.monotonic() + 5.0
    while sim.backend.calls < 3 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert sim.backend.calls >= 3, "render thread produced no frames"
    assert threading.get_ident() not in sim.backend.threads, "rendering ran on the calling thread"
    # 1.5 m within [0.3, 3.0] -> (1.5-0.3)/2.7 - 0.5
    assert p._array is not None
    np.testing.assert_allclose(p._array, (1.5 - 0.3) / 2.7 - 0.5, atol=1e-6)


def test_start_blocks_until_the_first_frame_is_published(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """Startup must not race: the GL/first-frame cost belongs before the timed sim loop."""
    p, sim = plugin
    assert sim.backend.calls == 0

    sim.hooks.emit(Phase.EPISODE_START, 0)

    # start() returned, so a frame already exists — no polling needed.
    assert sim.backend.calls >= 1
    assert p._first_frame.is_set()


def test_stop_joins_the_thread_before_releasing_shared_memory(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """Dropping the shm mapping under a live render would fault on the buffer it writes into."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    thread = p._thread
    assert thread is not None and thread.is_alive()

    sim.hooks.emit(Phase.CLOSE)

    assert not thread.is_alive()
    assert p._thread is None
    assert p._array is None


def test_render_thread_survives_a_render_error(plugin: tuple[DepthShmPlugin, _FakeSimulator]) -> None:
    """A transient backend failure must not kill the thread and silently freeze the depth stream."""
    _, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    calls = {"n": 0}
    original = sim.backend.render_cameras

    def flaky(cameras: list[CameraRuntime]) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient GL failure")
        original(cameras)

    sim.backend.render_cameras = flaky  # type: ignore[method-assign]
    before = sim.backend.calls
    deadline = time.monotonic() + 5.0
    while sim.backend.calls <= before and time.monotonic() < deadline:
        time.sleep(0.01)

    assert calls["n"] > 2, "thread stopped after the error instead of retrying"
    assert sim.backend.calls > before, "thread never resumed publishing"


def test_render_hz_must_be_positive() -> None:
    """A zero/negative rate would divide by zero in the loop period."""
    with pytest.raises(ValueError, match="render_hz must be > 0"):
        _config(render_hz=0.0)


def test_producer_rejects_existing_block_with_different_shape(accept_fake_backend: None) -> None:
    name = f"depth_img_shm_test_{uuid4().hex}"
    expected_bytes = 58 * 87 * np.dtype(np.float32).itemsize
    stale = shared_memory.SharedMemory(name=name, create=True, size=expected_bytes + 4)
    plugin = DepthShmPlugin(_config(shm_name=name), cast("BaseSimulator", _FakeSimulator()))

    try:
        with pytest.raises(ValueError, match=f"is {expected_bytes + 4} bytes"):
            plugin.start()
    finally:
        plugin.stop()
        stale.close()
        stale.unlink()


def test_simulator_without_a_backend_is_rejected_at_install() -> None:
    """IsaacSim/IsaacGym have no ``backend``: fail at install, not with an error logged every frame."""
    sim = _FakeSimulator()
    del sim.backend

    with pytest.raises(ValueError, match="only safe on the MuJoCo ClassicBackend.*with no backend"):
        DepthShmPlugin(_config(), cast("BaseSimulator", sim))


def test_non_classic_backend_is_rejected_at_install() -> None:
    """The WarpBackend (any non-ClassicBackend) would race the physics thread's own render."""
    sim = _FakeSimulator()

    with pytest.raises(ValueError, match="only safe on the MuJoCo ClassicBackend.*with _FakeBackend"):
        DepthShmPlugin(_config(), cast("BaseSimulator", sim))


def test_classic_backend_is_accepted() -> None:
    pytest.importorskip("mujoco")
    from holosoma.simulator.mujoco.backends import ClassicBackend

    sim = _FakeSimulator()
    cast("Any", sim).backend = object.__new__(ClassicBackend)  # only the type is checked at install

    DepthShmPlugin(_config(), cast("BaseSimulator", sim)).stop()


@pytest.mark.parametrize("backend_name", ["classic", "warp"])
def test_backend_validation_during_mujoco_construction(backend_name: str) -> None:
    """Plugins install in BaseSimulator.__init__, before MuJoCo.load_assets creates backend."""
    pytest.importorskip("mujoco")
    from holosoma.simulator.mujoco.mujoco import MuJoCo

    sim = object.__new__(MuJoCo)
    sim.__dict__.update(_FakeSimulator().__dict__)
    del sim.backend
    cast("Any", sim).simulator_config = SimpleNamespace(mujoco_backend=backend_name)

    if backend_name == "warp":
        with pytest.raises(ValueError, match="only safe on the MuJoCo ClassicBackend"):
            DepthShmPlugin(_config(), sim)
        return

    plugin = DepthShmPlugin(_config(), sim)
    try:
        # A valid configuration permits installation, but cannot start rendering by itself.
        with pytest.raises(ValueError, match="with no backend"):
            plugin.start()
        assert plugin._shm is None
        assert plugin._thread is None

        # Recheck the actual backend rather than trusting the earlier configuration.
        cast("Any", sim).backend = _FakeBackend()
        with pytest.raises(ValueError, match="with _FakeBackend"):
            plugin.start()
        assert plugin._shm is None
        assert plugin._thread is None
    finally:
        plugin.stop()


#########################################################################################################
## Liveness heartbeat: the producer half of the contract in holosoma_inference.sensors.depth_heartbeat
#########################################################################################################
def _subscribe(endpoint: str) -> Any:
    import zmq

    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(endpoint)
    return socket


def _collect_beats(socket: Any, done: Any, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Receive heartbeats until ``done(beats)`` holds or ``timeout`` expires."""
    beats: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not (beats and done(beats)):
        if socket.poll(5):  # ms; wait for the next heartbeat instead of spinning
            beats.append(json.loads(socket.recv()))
    return beats


@pytest.fixture
def heartbeat_plugin(accept_fake_backend: None) -> Iterator[tuple[DepthShmPlugin, _FakeSimulator, Any]]:
    endpoint = f"ipc:///tmp/hb-test-{uuid4().hex[:12]}"
    subscriber = _subscribe(endpoint)  # subscribe first: PUB drops messages sent before a subscriber joins
    sim = _FakeSimulator()
    p = DepthShmPlugin(_config(heartbeat_endpoint=endpoint), cast("BaseSimulator", sim))
    yield p, sim, subscriber
    p.stop()
    subscriber.close()


def _ok_beats(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in beats if b["state"] == depth_shm_plugin.HEARTBEAT_OK]


def test_heartbeat_seq_advances_with_each_frame_write(
    heartbeat_plugin: tuple[DepthShmPlugin, _FakeSimulator, Any],
) -> None:
    _, sim, subscriber = heartbeat_plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    beats = _ok_beats(_collect_beats(subscriber, lambda b: len(_ok_beats(b)) >= 4))
    assert len(beats) >= 4, "no steady OK heartbeat while frames are being written"
    seqs = [b["seq"] for b in beats]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"seq must increase per frame write: {seqs}"
    assert all(b["v"] == depth_shm_plugin.HEARTBEAT_VERSION for b in beats)


def test_heartbeat_reports_reconnecting_without_advancing_on_render_error(
    heartbeat_plugin: tuple[DepthShmPlugin, _FakeSimulator, Any],
) -> None:
    """A failing camera keeps the last frame and says RECONNECTING, so the consumer sees seq stall."""
    _, sim, subscriber = heartbeat_plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    _collect_beats(subscriber, lambda b: len(_ok_beats(b)) >= 2)

    def broken(cameras: list[CameraRuntime]) -> None:
        raise RuntimeError("camera unplugged")

    sim.backend.render_cameras = broken  # type: ignore[method-assign]
    beats = _collect_beats(
        subscriber, lambda b: sum(x["state"] == depth_shm_plugin.HEARTBEAT_RECONNECTING for x in b) >= 3
    )
    reconnecting = [b for b in beats if b["state"] == depth_shm_plugin.HEARTBEAT_RECONNECTING]
    assert len(reconnecting) >= 3
    assert len({b["seq"] for b in reconnecting}) == 1, "seq must not advance while no frame is written"


def test_heartbeat_reports_failed_when_the_producer_stops(
    heartbeat_plugin: tuple[DepthShmPlugin, _FakeSimulator, Any],
) -> None:
    p, sim, subscriber = heartbeat_plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    _collect_beats(subscriber, lambda b: len(_ok_beats(b)) >= 1)

    p.stop()
    beats = _collect_beats(subscriber, lambda b: b[-1]["state"] == depth_shm_plugin.HEARTBEAT_FAILED)
    assert beats and beats[-1]["state"] == depth_shm_plugin.HEARTBEAT_FAILED


def test_heartbeat_can_be_disabled(accept_fake_backend: None) -> None:
    sim = _FakeSimulator()
    p = DepthShmPlugin(_config(heartbeat=False), cast("BaseSimulator", sim))
    assert p._open_heartbeat() is None
    p.stop()


def test_heartbeat_matches_the_inference_wire_contract() -> None:
    """Both packages must agree on the message and the default endpoint; neither can import the other."""
    heartbeat = pytest.importorskip("holosoma_inference.sensors.depth_heartbeat")

    for state in (
        depth_shm_plugin.HEARTBEAT_OK,
        depth_shm_plugin.HEARTBEAT_RECONNECTING,
        depth_shm_plugin.HEARTBEAT_FAILED,
    ):
        beat = heartbeat.decode_heartbeat(depth_shm_plugin.heartbeat_message(7, state))
        assert (beat["seq"], beat["state"]) == (7, state)
    assert depth_shm_plugin.HEARTBEAT_VERSION == heartbeat.HEARTBEAT_VERSION
    assert depth_shm_plugin.default_heartbeat_endpoint("x") == heartbeat.default_heartbeat_endpoint("x")


def test_inference_monitor_tracks_the_sim_producer(
    heartbeat_plugin: tuple[DepthShmPlugin, _FakeSimulator, Any],
) -> None:
    """End to end across the two packages: live frames read as fresh, a stopped producer as FAILED."""
    heartbeat = pytest.importorskip("holosoma_inference.sensors.depth_heartbeat")
    p, sim, _ = heartbeat_plugin
    monitor = heartbeat.DepthHeartbeatMonitor(p.cfg.heartbeat_endpoint)
    try:
        sim.hooks.emit(Phase.EPISODE_START, 0)

        def poll_until(predicate: Any) -> Any:
            deadline = time.monotonic() + 5.0
            status = monitor.poll()
            while not predicate(status) and time.monotonic() < deadline:
                time.sleep(0.005)
                status = monitor.poll()
            return status

        live = poll_until(lambda s: s.checked and s.state == heartbeat.STATE_OK and (s.seq or 0) >= 2)
        assert live.state == heartbeat.STATE_OK and live.stale_s < 0.5

        p.stop()
        gone = poll_until(lambda s: s.state == heartbeat.STATE_FAILED)
        assert gone.state == heartbeat.STATE_FAILED
    finally:
        monitor.close()
