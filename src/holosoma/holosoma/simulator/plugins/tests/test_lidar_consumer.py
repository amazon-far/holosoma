"""Unit tests for the backend-neutral LiDAR consumer lifecycle."""

from __future__ import annotations

from typing import Any, Iterable, cast

import pytest

from holosoma.config_types.sensor import GridLidarRayPatternConfig, LidarSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.lidar_consumer import LidarConsumerPlugin, LidarStreamKey, PointCloudPacket
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _TrainingConfig:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs


class _FakeSimulator:
    def __init__(self, *, num_envs: int = 2, update_decimation: int = 1) -> None:
        lidar = LidarSensorConfig(
            mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
            pattern=GridLidarRayPatternConfig(horizontal_angles=[0.0, 90.0]),
            update_decimation=update_decimation,
        )
        self.hooks = HookRegistry()
        self.sensor_config = {"lidar": lidar}
        self.training_config = _TrainingConfig(num_envs)
        self.sensor_manager = SensorManager("cpu", control_hz=50.0)
        self.sensor_manager.register_lidar("lidar", lidar)
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


class _Consumer(LidarConsumerPlugin):
    def __init__(
        self,
        simulator: _FakeSimulator,
        streams: Iterable[LidarStreamKey],
        *,
        fail_start: bool = False,
    ) -> None:
        self._streams = set(streams)
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.stop_count = 0
        self.batches: list[dict[LidarStreamKey, PointCloudPacket]] = []
        super().__init__(cast("Any", None), cast("Any", simulator))

    def wanted_streams(self) -> set[LidarStreamKey]:
        return self._streams

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("start failed")
        self.started = True

    def publish(self, clouds: dict[LidarStreamKey, PointCloudPacket]) -> None:
        self.batches.append(clouds)

    def stop(self) -> None:
        self.stopped = True
        self.stop_count += 1


def test_consumer_publishes_only_fresh_lidar_scans() -> None:
    sim = _FakeSimulator(update_decimation=2)
    consumer = _Consumer(sim, {("lidar", 1)})
    record = sim.sensor_manager.get_lidar("lidar")
    record.set_buffer(
        "points",
        torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        ),
    )

    sim.hooks.emit(Phase.FRAME_END)
    assert not consumer.batches

    sim.sensor_manager.collect_lidars_due()
    sim.hooks.emit(Phase.FRAME_END)
    packet = consumer.batches[0][("lidar", 1)]
    assert consumer.started
    assert packet.sim_time == 1.25
    assert (packet.height, packet.width) == (1, 2)
    assert packet.points.tolist() == [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

    assert not sim.sensor_manager.collect_lidars_due()
    sim._time = 1.27
    sim.hooks.emit(Phase.FRAME_END)
    assert len(consumer.batches) == 1


def test_consumer_validates_lidar_and_environment() -> None:
    sim = _FakeSimulator(num_envs=1)
    with pytest.raises(ValueError, match="not configured"):
        _Consumer(sim, {("missing", 0)})
    with pytest.raises(ValueError, match="wants env 2"):
        _Consumer(sim, {("lidar", 2)})


def test_consumer_stops_on_close_even_before_first_scan() -> None:
    sim = _FakeSimulator()
    consumer = _Consumer(sim, {("lidar", 0)})
    sim.hooks.emit(Phase.CLOSE)
    assert consumer.stopped


def test_failed_start_rolls_back_before_retry() -> None:
    sim = _FakeSimulator()
    consumer = _Consumer(sim, {("lidar", 0)}, fail_start=True)
    sim.sensor_manager.get_lidar("lidar").set_buffer("points", torch.ones((2, 2, 3)))
    sim.sensor_manager.collect_lidars_due()

    sim.hooks.emit(Phase.FRAME_END)
    assert consumer.stop_count == 1
    assert not bool(consumer.started)
    assert not consumer.batches

    consumer.fail_start = False
    sim.sensor_manager.collect_lidars_due()
    sim.hooks.emit(Phase.FRAME_END)
    assert bool(consumer.started)
    assert len(consumer.batches) == 1
