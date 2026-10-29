"""``BaseSimulator.set_robot_state`` writes row ``i`` of its inputs to ``env_ids[i]``."""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _FakeSim(BaseSimulator):
    """Two envs with one DOF each; the backend writers are no-ops."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.num_dof = 1
        self.robot_root_states = torch.zeros(2, 13)
        self.dof_pos = torch.zeros(2, 1)
        self.dof_vel = torch.zeros(2, 1)

    def set_dof_state_tensor_robots(self, env_ids: object = None, dof_states: object = None) -> None:
        pass

    def set_actor_root_state_tensor_robots(self, env_ids: object = None, root_states: object = None) -> None:
        pass


def test_states_follow_a_reordered_full_env_id_list() -> None:
    sim = _FakeSim()
    env_ids = torch.tensor([1, 0])
    root_states = torch.zeros(2, 13)
    root_states[:, 0] = torch.tensor([10.0, 20.0])  # x for env 1, then env 0
    dof_states = torch.tensor([[[1.0, 0.1]], [[2.0, 0.2]]])

    sim.set_robot_state(env_ids, root_states=root_states, dof_states=dof_states)

    assert sim.robot_root_states[:, 0].tolist() == [20.0, 10.0]
    assert sim.dof_pos[:, 0].tolist() == [2.0, 1.0]
