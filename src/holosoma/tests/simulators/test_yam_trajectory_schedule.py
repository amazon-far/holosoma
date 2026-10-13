"""The YAM harness must close each frame after all physics substeps, like ``DirectSimulation.run``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.simulator.base_simulator.hooks import Phase
from tests.simulators.yam_trajectory_assert import _run_until

pytestmark = pytest.mark.no_sim


class _Hooks:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    def emit(self, phase: Phase) -> None:
        self.events.append(phase)


class _Sim:
    sim_dt = 0.1

    def __init__(self) -> None:
        self.now = 0.0
        self.events: list[Any] = []
        self.hooks = _Hooks(self.events)
        self.simulator_config = SimpleNamespace(sim=SimpleNamespace(control_decimation_steps=3))

    def time(self) -> float:
        return self.now

    def refresh_sim_tensors(self) -> None:
        self.events.append("refresh")

    def simulate_at_each_physics_step(self) -> None:
        self.events.append("simulate")
        self.now += self.sim_dt


def test_frame_end_follows_the_last_substep_and_a_refresh() -> None:
    sim = _Sim()

    _run_until(sim, 0.6)

    phases = [event for event in sim.events if isinstance(event, Phase)]
    one_frame = [
        Phase.FRAME_BEGIN,
        Phase.PRE_STEP,
        Phase.POST_STEP,
        Phase.PRE_STEP,
        Phase.POST_STEP,
        Phase.PRE_STEP,
        Phase.POST_STEP,
        Phase.FRAME_END,
    ]
    assert phases == one_frame * 2
    for index, event in enumerate(sim.events):
        if event is Phase.FRAME_END:
            assert sim.events[index - 1] == "refresh"
