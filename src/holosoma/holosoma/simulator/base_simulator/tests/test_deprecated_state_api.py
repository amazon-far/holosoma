from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.state import SimState

pytestmark = pytest.mark.no_sim


class _FakeSim(BaseSimulator):
    """Writable joint and root state served by a registry, with FK counted."""

    def __init__(self) -> None:
        self.state = SimState()
        self.num_envs, self.num_dof, self.sim_device = 2, 3, "cpu"
        self.backing = {"q": torch.zeros(2, 3), "v": torch.zeros(2, 3), "root": torch.zeros(2, 13)}
        for name, key in (("dof_pos", "q"), ("dof_vel", "v"), ("robot_root_states", "root")):
            self.state.register(name, self._reader(self.backing[key]), write=self._writer(self.backing[key]))
        self.fk_calls = 0

    @staticmethod
    def _reader(backing: torch.Tensor) -> Callable[[], torch.Tensor]:
        return lambda: backing

    @staticmethod
    def _writer(backing: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], None]:
        def write(rows: torch.Tensor, out: torch.Tensor) -> None:
            backing[rows] = out[rows]

        return write

    @property
    def all_root_states(self) -> torch.Tensor:
        return torch.empty(0)

    def _forward_kinematics(self) -> None:
        self.fk_calls += 1


@pytest.mark.parametrize("flat", [False, True])
def test_deprecated_setters_write_through_and_warn(flat: bool) -> None:
    sim = _FakeSim()
    states = torch.stack([torch.full((1, 3), 1.0), torch.full((1, 3), 2.0)], dim=-1)  # env 1 only, pre-sliced
    with pytest.warns(FutureWarning, match="set_dof_state_tensor_robots"):
        sim.set_dof_state_tensor_robots(torch.tensor([1]), states.reshape(-1, 2) if flat else states)
    assert torch.all(sim.backing["q"][1] == 1.0) and torch.all(sim.backing["v"][1] == 2.0)
    assert torch.all(sim.backing["q"][0] == 0.0) and sim.fk_calls == 1

    root = torch.ones(2, 13)
    with pytest.warns(FutureWarning, match="set_actor_root_state_tensor_robots"):
        sim.set_actor_root_state_tensor(torch.tensor([0]), root)  # full-width input, legacy alias
    assert torch.all(sim.backing["root"][0] == 1.0) and torch.all(sim.backing["root"][1] == 0.0)

    with pytest.warns(FutureWarning, match="refresh_sim_tensors"):
        sim.refresh_sim_tensors()
