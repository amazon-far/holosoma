"""Backend-neutral host snapshot path for LiDAR consumer plugins."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.sensor import LidarSensorConfig
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import PluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

LidarStreamKey = Tuple[str, int]


@dataclass
class PointCloudPacket:
    lidar: str
    env_id: int
    points: npt.NDArray[np.float32]
    sim_time: float
    height: int
    width: int

    @property
    def key(self) -> LidarStreamKey:
        return self.lidar, self.env_id


class LidarConsumerPlugin:
    """Base plugin for consumers of fresh sensor-local LiDAR point clouds."""

    def __init__(self, config: PluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = config
        self.config = config
        self.simulator = simulator
        self._wanted = self.wanted_streams()
        self._validate_streams()
        self._started = False
        self._closed = False
        simulator.hooks.add(Phase.FRAME_END, self._on_frame_end, name=f"{type(self).__name__}.publish")
        simulator.hooks.add(Phase.CLOSE, self._on_close, name=f"{type(self).__name__}.stop")

    @abstractmethod
    def wanted_streams(self) -> set[LidarStreamKey]:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def publish(self, clouds: dict[LidarStreamKey, PointCloudPacket]) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def _validate_streams(self) -> None:
        lidars = {
            name: config
            for name, config in self.simulator.sensor_config.items()
            if isinstance(config, LidarSensorConfig)
        }
        num_envs = self.simulator.training_config.num_envs
        for name, env_id in sorted(self._wanted):
            if name not in lidars:
                raise ValueError(
                    f"{type(self).__name__} references LiDAR '{name}', which is not configured "
                    f"(LiDARs: {sorted(lidars)})."
                )
            if not 0 <= env_id < num_envs:
                raise ValueError(
                    f"{type(self).__name__} wants env {env_id} of LiDAR '{name}', but the sim has "
                    f"{num_envs} env(s) [0, {num_envs})."
                )

    def _snapshot(self) -> dict[LidarStreamKey, PointCloudPacket]:
        manager = self.simulator.sensor_manager
        if manager is None:
            return {}
        fresh = manager.last_lidar_due
        sim_time = self.simulator.time()
        packets = {}
        for name, env_id in self._wanted:
            if name not in fresh:
                continue
            points = self.simulator.get_lidar_data(name, "points", device="cpu").detach().numpy()[env_id]
            height, width = manager.get_lidar(name).pattern_shape
            packet = PointCloudPacket(
                lidar=name,
                env_id=env_id,
                points=points,
                sim_time=sim_time,
                height=height,
                width=width,
            )
            packets[packet.key] = packet
        return packets

    def _on_frame_end(self) -> None:
        try:
            batch = self._snapshot()
            if not batch:
                return
            if not self._started:
                try:
                    self.start()
                except Exception:
                    # A transport may fail after allocating only part of its resources.
                    # Roll that partial start back before the next fresh scan retries it.
                    try:
                        self.stop()
                    except Exception as cleanup_exc:
                        logger.error(
                            f"LiDAR consumer {type(self).__name__} cleanup after start failure failed: {cleanup_exc}"
                        )
                    raise
                self._started = True
            self.publish(batch)
        except Exception as exc:
            logger.error(f"LiDAR consumer {type(self).__name__} publish failed: {exc}")

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
