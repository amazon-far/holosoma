from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.hooks import HookRegistry, Phase
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


class _StepRecordingSim(BaseSimulator):
    """Minimal subclass that deliberately uses the real BaseSimulator initialization path."""

    def __init__(self, hooks: HookRegistry, *, kinematic_playback: bool) -> None:
        self.events: list[str] = []
        sim_cfg = SimpleNamespace(
            debug_viz=False,
            sim=SimpleNamespace(fps=100, control_decimation_steps=2, kinematic_playback=kinematic_playback),
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
        super().__init__(cast("Any", config), cast("Any", None), "cpu", hooks)

    def _step_dynamics(self) -> None:
        self.events.append("dynamics")

    def forward_kinematics(self) -> None:
        self.events.append("kinematics")


@pytest.mark.parametrize(("kinematic_playback", "step"), [(False, "dynamics"), (True, "kinematics")])
def test_physics_step_is_bracketed_by_pre_and_post_step(kinematic_playback: bool, step: str) -> None:
    hooks = HookRegistry()
    sim = _StepRecordingSim(hooks, kinematic_playback=kinematic_playback)
    hooks.add(Phase.PRE_STEP, lambda: sim.events.append("pre_step"))
    hooks.add(Phase.POST_STEP, lambda: sim.events.append("post_step"))

    sim.simulate_at_each_physics_step()

    assert sim.events == ["pre_step", step, "post_step"]
