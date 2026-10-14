"""``robot_rows``: the row-selection rule shared by every backend's robot root writer.

Runs it on a stand-in carrying only the three attributes it reads (``num_envs``, ``robot_root_states``,
``all_root_states``). Pure torch, no simulator.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.root_states import robot_rows

pytestmark = pytest.mark.no_sim

NUM_ENVS = 4
ENV_IDS = torch.tensor([0, 2])


def _sim() -> BaseSimulator:
    sim: Any = SimpleNamespace(
        num_envs=NUM_ENVS,
        robot_root_states=torch.arange(NUM_ENVS * 13, dtype=torch.float).view(NUM_ENVS, 13),
        all_root_states=object(),  # the duck-typed unified proxy (not a Tensor); only identity matters
    )
    return cast("BaseSimulator", sim)


@pytest.mark.parametrize("form", ["none", "all_root_states", "robot_root_states"])
def test_current_state_forms_select_robot_rows_for_env_ids(form: str) -> None:
    sim = _sim()
    root_states = {"none": None, "all_root_states": sim.all_root_states, "robot_root_states": sim.robot_root_states}
    rows = robot_rows(sim, ENV_IDS, root_states[form])
    assert torch.equal(rows, sim.robot_root_states[ENV_IDS])
    assert rows.shape == (len(ENV_IDS), 13)


def test_pre_sliced_tensor_is_returned_as_is() -> None:
    sim = _sim()
    pre_sliced = torch.full((len(ENV_IDS), 13), 7.0)
    assert robot_rows(sim, ENV_IDS, pre_sliced) is pre_sliced


def test_global_tensor_is_indexed_by_env_ids() -> None:
    sim = _sim()
    global_states = -torch.arange(NUM_ENVS * 13, dtype=torch.float).view(NUM_ENVS, 13)
    rows = robot_rows(sim, ENV_IDS, global_states)
    assert torch.equal(rows, global_states[ENV_IDS])
    assert not torch.equal(rows, sim.robot_root_states[ENV_IDS])  # the caller's tensor, not the mirror


def test_full_length_tensor_is_global_even_when_env_ids_are_permuted() -> None:
    """Global takes precedence over pre-sliced (as in set_dof_state_tensor_robots): a full-length tensor is
    indexed by env_ids, so a permuted full env_ids reorders its rows."""
    sim = _sim()
    all_ids = torch.tensor([3, 2, 1, 0])
    global_states = torch.arange(NUM_ENVS * 13, dtype=torch.float).view(NUM_ENVS, 13) + 100.0
    rows = robot_rows(sim, all_ids, global_states)
    assert torch.equal(rows, global_states[all_ids])
    assert torch.equal(rows[:, 0], torch.tensor([139.0, 126.0, 113.0, 100.0]))


def test_other_lengths_raise_value_error() -> None:
    sim = _sim()
    with pytest.raises(
        ValueError, match=r"dimension mismatch: got 5, expected either 4 \(global\) or 2 \(pre-sliced\)"
    ):
        robot_rows(sim, ENV_IDS, torch.zeros(NUM_ENVS + 1, 13))
    with pytest.raises(ValueError, match="dimension mismatch: got 1"):
        robot_rows(sim, ENV_IDS, torch.zeros(1, 13))


def test_non_tensor_raises_type_error() -> None:
    sim = _sim()
    with pytest.raises(
        TypeError, match="root_states must be a torch.Tensor, all_root_states or robot_root_states, got list"
    ):
        robot_rows(sim, ENV_IDS, [[0.0] * 13, [0.0] * 13])
    with pytest.raises(TypeError, match="got ndarray"):
        robot_rows(sim, ENV_IDS, torch.zeros(len(ENV_IDS), 13).numpy())
