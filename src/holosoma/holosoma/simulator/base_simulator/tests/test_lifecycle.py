from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


class _Recorder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def capture_frame(self) -> None:
        self.events.append("video.capture")

    def cleanup(self) -> None:
        self.events.append("video.cleanup")


class _Bridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("bridge.close")


class _InitializedFakeSim(BaseSimulator):
    """Minimal subclass that deliberately uses the real BaseSimulator initialization path."""

    def __init__(self) -> None:
        self.events: list[str] = []
        sim_cfg = SimpleNamespace(
            debug_viz=False,
            sim=SimpleNamespace(fps=100, control_decimation_steps=2, kinematic_playback=False),
        )
        config = SimpleNamespace(
            training=TrainingConfig(num_envs=1),
            simulator=sim_cfg,
            scene=SimpleNamespace(),
            sensors={},
            robot=SimpleNamespace(),
            logger=DisabledLoggerConfig(),
            experiment_dir=None,
        )
        super().__init__(cast("Any", config), cast("Any", None), "cpu")
        self.video_recorder = cast("Any", _Recorder(self.events))
        self.bridge = cast("Any", _Bridge(self.events))

    def render_sensors(self) -> None:
        self.events.append("render")

    def _close_resources(self) -> None:
        self.events.append("backend.close")


def test_end_frame_renders_sensors_before_capturing_video() -> None:
    sim = _InitializedFakeSim()
    sim.end_frame()
    assert sim.events == ["render", "video.capture"]


def test_close_releases_video_then_bridge_then_backend_once() -> None:
    sim = _InitializedFakeSim()
    sim.close()
    sim.close()
    assert sim.events == ["video.cleanup", "bridge.close", "backend.close"]
