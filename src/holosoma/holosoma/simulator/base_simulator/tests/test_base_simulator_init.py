from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.plugin import PluginConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.hooks import HookRegistry, Phase
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


@dataclass(frozen=True)
class _RenderedSensorConsumerConfig(PluginConfig):
    def get_cls(self) -> Callable[..., Any]:
        return _RenderedSensorConsumer


class _RenderedSensorConsumer:
    def __init__(self, cfg: _RenderedSensorConsumerConfig, task: BaseTask) -> None:
        self.sim = cast("_InitializedFakeSim", task.simulator)
        task.hooks.add(Phase.FRAME_END, self.publish, name="consumer.publish")

    def publish(self) -> None:
        self.sim.events.append(f"publish:{self.sim.latest_sensor_value}")


class _InitializedFakeSim(BaseSimulator):
    """Minimal subclass that deliberately uses the real BaseSimulator initialization path."""

    def __init__(self, hooks: HookRegistry | None = None) -> None:
        self.events: list[str] = []
        self.latest_sensor_value = "stale"
        sim_cfg = SimpleNamespace(
            debug_viz=False,
            sim=SimpleNamespace(
                fps=100,
                control_decimation_steps=2,
                kinematic_playback=False,
            ),
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
        super().__init__(cast("Any", config), cast("Any", None), "cpu", hooks or HookRegistry())

    def render_sensors(self) -> None:
        self.latest_sensor_value = "fresh"
        self.events.append("render")

    def _advance_substep(self) -> None:
        self.events.append("kinematics" if self.kinematic_playback else "dynamics")


@pytest.mark.parametrize("kinematic_playback", [False, True])
def test_physics_step_is_bracketed_by_pre_and_post_step(kinematic_playback: bool) -> None:
    hooks = HookRegistry()
    sim = _InitializedFakeSim(hooks)
    sim.kinematic_playback = kinematic_playback
    hooks.add(Phase.PRE_STEP, lambda: sim.events.append("pre_step"), name="probe.pre_step")
    hooks.add(Phase.POST_STEP, lambda: sim.events.append("post_step"), name="probe.post_step")

    sim.simulate_at_each_physics_step()

    assert sim.events == ["pre_step", "kinematics" if kinematic_playback else "dynamics", "post_step"]


def test_base_simulator_renders_sensors_before_task_plugins_consume_them() -> None:
    hooks = HookRegistry()
    sim = _InitializedFakeSim(hooks)
    task = BaseTask.__new__(BaseTask)
    task.config = cast("Any", SimpleNamespace(plugin={"consumer": _RenderedSensorConsumerConfig()}))
    task.simulator = sim
    task.hooks = hooks
    task._install_plugins()
    hooks.emit(Phase.FRAME_END)
    assert sim.events == ["render", "publish:fresh"]


def test_empty_sensor_registry_is_available_from_real_base_initialization() -> None:
    sim = _InitializedFakeSim()
    assert sim.get_sensor_names() == []
    assert sim.sensor_manager.num_envs == sim.training_config.num_envs
    with pytest.raises(KeyError, match="No sensor named 'missing'"):
        sim.sensor_manager.get_mount_pose("missing")
    with pytest.raises(NotImplementedError, match="no camera 'missing'"):
        sim.get_camera_data("missing")
    with pytest.raises(NotImplementedError, match="no LiDAR 'missing'"):
        sim.get_lidar_data("missing")
