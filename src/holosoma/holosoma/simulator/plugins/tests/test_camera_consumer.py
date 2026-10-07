"""Unit tests for CameraConsumerPlugin (pure, no simulator, no ROS).

Covers the base-class contract the egress consumers rely on: it registers publish on
FRAME_END and stop on CLOSE; per step it snapshots only the consumer's fresh wanted
streams (one shared cached device->host read per (camera, modality), serving every wanted env);
validates the wanted streams against the configured cameras at construction (fail-loud); isolates
publish failures; and surfaces teardown failures. A ``FakeConsumer`` test double stands in for a
real transport.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.config_types.sensor import (
    CameraDataType,
    CameraSensorConfig,
    IsaacSimCameraConfig,
    IsaacSimFisheyeConfig,
    SensorMountConfig,
)
from holosoma.simulator.base_simulator.hooks import HookCloseError, HookRegistry, Phase
from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin, FramePacket, StreamKey
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


# ----- test double: an in-memory camera consumer with no transport -----


class FakeConsumer(CameraConsumerPlugin):
    """Records every per-step batch it receives. ``wanted_streams`` comes from a passed-in set."""

    def __init__(
        self,
        config: Any,
        simulator: Any,
        *,
        streams: Iterable[StreamKey],
        fail: bool = False,
    ) -> None:
        self._streams = set(streams)
        self._fail = fail
        self.started = False
        self.stopped = False
        self.stop_calls = 0
        self.batches: list[dict[StreamKey, FramePacket]] = []  # each control step's frames dict
        super().__init__(config, simulator)

    @property
    def received(self) -> list[FramePacket]:
        return [pkt for batch in self.batches for pkt in batch.values()]

    def wanted_streams(self) -> set[StreamKey]:
        return self._streams

    def start(self) -> None:
        self.started = True

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        if self._fail:
            raise RuntimeError("boom")
        self.batches.append(frames)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True


# ----- a minimal fake simulator exposing only what the hook base touches -----


class _FakeSensorManager:
    def __init__(self) -> None:
        self.last_due: set[str] = set()


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4  # control_hz = 50


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeTrainingConfig:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs


class _FakeSimulator:
    """Minimal stand-in exposing only what CameraConsumerPlugin touches on the simulator."""

    def __init__(
        self,
        sensors_config: dict[str, CameraSensorConfig],
        frames: dict[tuple[str, str], npt.NDArray[np.uint8]],
        num_envs: int = 1,
    ) -> None:
        self.hooks = HookRegistry()
        self.sensor_config = sensors_config
        self.sensor_manager = _FakeSensorManager()
        self.training_config = _FakeTrainingConfig(num_envs)
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self._frames = frames  # (camera, modality) -> [N, H, W, C] numpy
        self._t = 0.0
        self.reads: list[tuple[str, str]] = []
        self.simulator_type = SimulatorType.MUJOCO

    def time(self) -> float:
        return self._t

    def get_simulator_type(self) -> SimulatorType:
        return self.simulator_type

    def get_camera_data(
        self,
        name: str,
        data_type: str = "rgb",
        env_ids: Any = None,
        device: Any = None,
    ) -> torch.Tensor:
        # The hook base reads the full [N, ...] host buffer once (device="cpu") and indexes envs
        # itself; frames here are already host numpy, so device is accepted and ignored.
        self.reads.append((name, data_type))
        return torch.from_numpy(self._frames[(name, data_type)])


def _cam(data_types: Sequence[CameraDataType] = ("rgb",)) -> CameraSensorConfig:
    return CameraSensorConfig(mount=_MOUNT, data_types=list(data_types))


def _sensors(*names: str, data_types: Sequence[CameraDataType] = ("rgb",)) -> dict[str, CameraSensorConfig]:
    return {n: _cam(data_types) for n in names}


def _rgb(h: int = 2, w: int = 2, n: int = 1) -> npt.NDArray[np.uint8]:
    return np.zeros((n, h, w, 3), dtype=np.uint8)


def _step(sim: _FakeSimulator) -> None:
    """Emit one FRAME_END — the phase every consumer registers its publish on."""
    sim.hooks.emit(Phase.FRAME_END)


# ----- tests -----


def test_registers_publish_and_close_callbacks() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    # One FRAME_END hook (publish) and one CLOSE hook (stop) were registered.
    assert len(sim.hooks._snapshots[Phase.FRAME_END]) == 1
    assert len(sim.hooks._snapshots[Phase.CLOSE]) == 1


def test_publishes_only_fresh_wanted_streams() -> None:
    sim = _FakeSimulator(_sensors("head", "wrist"), {("head", "rgb"): _rgb(), ("wrist", "rgb"): _rgb()})
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0), ("wrist", "rgb", 0)])
    # Only 'head' rendered this step -> only head is published, wrist is not read.
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    assert c.started  # lazily started on first publish
    assert [(p.camera, p.modality) for p in c.received] == [("head", "rgb")]
    assert ("wrist", "rgb") not in sim.reads


def test_frame_intrinsics_include_active_backend_config() -> None:
    camera = CameraSensorConfig(
        mount=_MOUNT,
        isaacsim=IsaacSimCameraConfig(
            projection_type="fisheyePolynomial",
            fisheye=IsaacSimFisheyeConfig(max_fov=200.0),
        ),
    )
    sim = _FakeSimulator({"head": camera}, {("head", "rgb"): _rgb()})
    consumer = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])

    sim.sensor_manager.last_due = {"head"}
    expected = (
        (SimulatorType.MUJOCO, camera.mujoco),
        (SimulatorType.ISAACGYM, camera.isaacgym),
        (SimulatorType.ISAACSIM, camera.isaacsim),
    )
    for simulator_type, backend_config in expected:
        sim.simulator_type = simulator_type
        _step(sim)
        intrinsics = consumer.received[-1].intrinsics
        assert intrinsics.simulator_type is simulator_type
        assert intrinsics.backend_config is backend_config


def test_snapshot_gives_each_consumer_its_frame() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    a = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    b = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    # Both consumers got the frame. Each self-serves get_camera_data(device="cpu"); the FULL-buffer
    # cache dedups the device->host copy across them at the runtime layer (covered in the runtime
    # cache tests) — here we assert both received exactly one packet.
    assert len(a.received) == 1
    assert len(b.received) == 1


def test_one_read_serves_multiple_envs() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb(n=3)}, num_envs=3)
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0), ("head", "rgb", 2)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    assert sim.reads == [("head", "rgb")]  # ONE read for both envs
    assert sorted(p.env_id for p in c.received) == [0, 2]


def test_packet_carries_intrinsics_sim_time_and_env() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb(h=4, w=6)})
    sim._t = 1.25
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    pkt = c.received[0]
    assert pkt.sim_time == 1.25
    assert pkt.env_id == 0
    assert (pkt.intrinsics.width, pkt.intrinsics.height) == (128, 128)  # config defaults
    assert tuple(pkt.array.shape) == (4, 6, 3) and pkt.array.dtype == np.uint8


def test_failing_consumer_is_isolated() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    FakeConsumer(None, sim, streams=[("head", "rgb", 0)], fail=True)
    good = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)  # must NOT raise despite the failing consumer
    assert len(good.received) == 1  # the good consumer (registered second) still got its frame


def test_stop_runs_on_close() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.hooks.emit(Phase.CLOSE)
    c._on_close()
    assert c.stopped
    assert c.stop_calls == 1


def test_stop_failure_propagates_to_close_registry() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])

    def fail_stop() -> None:
        raise RuntimeError("stop failed")

    c.stop = fail_stop  # type: ignore[method-assign]
    with pytest.raises(HookCloseError, match="stop failed"):
        sim.hooks.emit(Phase.CLOSE)


def test_validation_rejects_unknown_camera() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    with pytest.raises(ValueError, match="not among the configured cameras"):
        FakeConsumer(None, sim, streams=[("nonexistent", "rgb", 0)])


def test_validation_rejects_unrendered_modality() -> None:
    sim = _FakeSimulator(_sensors("head", data_types=("rgb",)), {("head", "rgb"): _rgb()})
    with pytest.raises(ValueError, match="renders only"):
        FakeConsumer(None, sim, streams=[("head", "depth", 0)])


def test_validation_rejects_out_of_range_env() -> None:
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()}, num_envs=1)
    with pytest.raises(ValueError, match="env 5"):
        FakeConsumer(None, sim, streams=[("head", "rgb", 5)])


def test_config_layer_imports_without_rclpy() -> None:
    # The optional-dependency guarantee: importing the config + config_values + plugins package must
    # not import rclpy (the deferred get_cls import is the only path that would). Guards against a
    # regression where someone top-level-imports a transport dep in the ROS-free layer.
    import holosoma.config_types.plugin
    import holosoma.config_values.plugin
    import holosoma.simulator.plugins  # noqa: F401

    assert "rclpy" not in sys.modules, "config/plugins layer must stay ROS-free; rclpy leaked in."
