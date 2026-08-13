"""Unit tests for CameraConsumerPlugin (pure, no simulator, no ROS).

Covers the base-class contract the egress consumers rely on: it registers publish on
FRAME_END and stop on CLOSE; per step it snapshots only the consumer's fresh wanted
streams (one shared cached device->host read per (camera, modality), serving every wanted env);
validates the wanted streams against the configured cameras at construction (fail-loud); and isolates
a failing consumer so it neither breaks the sim loop nor its siblings. A ``FakeConsumer`` test double
stands in for a real transport.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin, FramePacket
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


# ----- test double: an in-memory camera consumer with no transport -----


class FakeConsumer(CameraConsumerPlugin):
    """Records every per-step batch it receives. ``wanted_streams`` comes from a passed-in set."""

    def __init__(self, config, simulator, *, streams, fail=False):
        self._streams = set(streams)
        self._fail = fail
        self.started = False
        self.stopped = False
        self.batches: list[dict] = []  # each control step's frames dict
        super().__init__(config, simulator)

    @property
    def received(self) -> list[FramePacket]:
        return [pkt for batch in self.batches for pkt in batch.values()]

    def wanted_streams(self):
        return self._streams

    def start(self):
        self.started = True

    def publish(self, frames):
        if self._fail:
            raise RuntimeError("boom")
        self.batches.append(frames)

    def stop(self):
        self.stopped = True


# ----- a minimal fake simulator exposing only what the hook base touches -----


class _FakeSensorManager:
    def __init__(self):
        self.last_due: set[str] = set()


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4  # control_hz = 50


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeTrainingConfig:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class _FakeSimulator:
    """Minimal stand-in exposing only what CameraConsumerPlugin touches on the simulator."""

    def __init__(self, sensors_config: dict[str, CameraSensorConfig], frames, num_envs: int = 1):
        self.hooks = HookRegistry()
        self.sensor_config = sensors_config
        self.sensor_manager = _FakeSensorManager()
        self.training_config = _FakeTrainingConfig(num_envs)
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self._frames = frames  # (camera, modality) -> [N, H, W, C] numpy
        self._t = 0.0
        self.reads: list[tuple[str, str]] = []

    def time(self) -> float:
        return self._t

    def sensor_config_by_name(self, name):
        return self.sensor_config[name]

    def get_camera_data(self, name, data_type="rgb", env_ids=None, device=None):
        # The hook base reads the full [N, ...] host buffer once (device="cpu") and indexes envs
        # itself; frames here are already host numpy, so device is accepted and ignored.
        self.reads.append((name, data_type))
        return torch.from_numpy(self._frames[(name, data_type)])


def _cam(data_types=("rgb",)) -> CameraSensorConfig:
    return CameraSensorConfig(mount=_MOUNT, data_types=list(data_types))


def _sensors(*names, data_types=("rgb",)) -> dict[str, CameraSensorConfig]:
    return {n: _cam(data_types) for n in names}


def _rgb(h=2, w=2, n=1):
    return np.zeros((n, h, w, 3), dtype=np.uint8)


def _step(sim: _FakeSimulator) -> None:
    """Emit one FRAME_END — the phase every consumer registers its publish on."""
    sim.hooks.emit(Phase.FRAME_END)


# ----- tests -----


def test_registers_publish_and_close_callbacks():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    # One FRAME_END hook (publish) and one CLOSE hook (stop) were registered.
    assert len(sim.hooks._snapshots[Phase.FRAME_END]) == 1
    assert len(sim.hooks._snapshots[Phase.CLOSE]) == 1


def test_publishes_only_fresh_wanted_streams():
    sim = _FakeSimulator(_sensors("head", "wrist"), {("head", "rgb"): _rgb(), ("wrist", "rgb"): _rgb()})
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0), ("wrist", "rgb", 0)])
    # Only 'head' rendered this step -> only head is published, wrist is not read.
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    assert c.started  # lazily started on first publish
    assert [(p.camera, p.modality) for p in c.received] == [("head", "rgb")]
    assert ("wrist", "rgb") not in sim.reads


def test_snapshot_gives_each_consumer_its_frame():
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


def test_one_read_serves_multiple_envs():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb(n=3)}, num_envs=3)
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0), ("head", "rgb", 2)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    assert sim.reads == [("head", "rgb")]  # ONE read for both envs
    assert sorted(p.env_id for p in c.received) == [0, 2]


def test_packet_carries_intrinsics_sim_time_and_env():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb(h=4, w=6)})
    sim._t = 1.25
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)
    pkt = c.received[0]
    assert pkt.sim_time == 1.25
    assert pkt.env_id == 0
    assert (pkt.intrinsics.width, pkt.intrinsics.height) == (128, 128)  # config defaults
    assert pkt.array.shape == (4, 6, 3) and pkt.array.dtype == np.uint8


def test_failing_consumer_is_isolated():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    FakeConsumer(None, sim, streams=[("head", "rgb", 0)], fail=True)
    good = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.sensor_manager.last_due = {"head"}
    _step(sim)  # must NOT raise despite the failing consumer
    assert len(good.received) == 1  # the good consumer (registered second) still got its frame


def test_stop_runs_on_close():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    c = FakeConsumer(None, sim, streams=[("head", "rgb", 0)])
    sim.hooks.emit(Phase.CLOSE)
    assert c.stopped


def test_validation_rejects_unknown_camera():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()})
    with pytest.raises(ValueError, match="not among the configured cameras"):
        FakeConsumer(None, sim, streams=[("nonexistent", "rgb", 0)])


def test_validation_rejects_unrendered_modality():
    sim = _FakeSimulator(_sensors("head", data_types=("rgb",)), {("head", "rgb"): _rgb()})
    with pytest.raises(ValueError, match="renders only"):
        FakeConsumer(None, sim, streams=[("head", "depth", 0)])


def test_validation_rejects_out_of_range_env():
    sim = _FakeSimulator(_sensors("head"), {("head", "rgb"): _rgb()}, num_envs=1)
    with pytest.raises(ValueError, match="env 5"):
        FakeConsumer(None, sim, streams=[("head", "rgb", 5)])


def test_config_layer_imports_without_rclpy():
    # The optional-dependency guarantee: importing the config + config_values + plugins package must
    # not import rclpy (the deferred get_cls import is the only path that would). Guards against a
    # regression where someone top-level-imports a transport dep in the ROS-free layer.
    import holosoma.config_types.plugin
    import holosoma.config_values.plugin
    import holosoma.simulator.plugins  # noqa: F401

    assert "rclpy" not in sys.modules, "config/plugins layer must stay ROS-free; rclpy leaked in."
