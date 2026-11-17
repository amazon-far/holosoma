"""Batched object roots retain native getter conventions and stable cache storage."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("mujoco")

from holosoma.simulator.base_simulator.state import SimState
from holosoma.simulator.mujoco.object_state_reader import ObjectReadPlan

pytestmark = pytest.mark.no_sim


def mixed_plan() -> ObjectReadPlan:
    """Two free objects plus a static body, deliberately ordered unlike qpos."""
    half = math.sqrt(0.5)
    qpos = torch.tensor(
        [
            [1, 2, 3, half, 0, 0, half, 4, 5, 6, 1, 0, 0, 0],
            [7, 8, 9, 1, 0, 0, 0, 10, 11, 12, half, 0, 0, half],
        ],
        dtype=torch.float32,
    )
    qvel = torch.tensor(
        [
            [1, 2, 3, 1, 0, 0, 4, 5, 6, 0, 0, 2],
            [7, 8, 9, 0, 0, 3, 10, 11, 12, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    xpos = torch.zeros(2, 3, 3)
    xquat = torch.zeros(2, 3, 4)
    xpos[:, 1] = torch.tensor([[20, 21, 22], [23, 24, 25]])
    xquat[:, 1, 0] = 1
    return ObjectReadPlan.build(
        ("static", "second", "first"),
        {"first": {"qpos_addr": 0, "qvel_addr": 0}, "second": {"qpos_addr": 7, "qvel_addr": 6}},
        {"static": 1},
        qpos,
        qvel,
        xpos,
        xquat,
    )


def test_mixed_roots_match_world_state_in_named_order() -> None:
    plan = mixed_plan()
    half = math.sqrt(0.5)
    expected = torch.tensor(
        [
            [
                [20, 21, 22, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 1, 4, 5, 6, 0, 0, 2],
                [1, 2, 3, 0, 0, half, half, 1, 2, 3, 0, 1, 0],
            ],
            [
                [23, 24, 25, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [10, 11, 12, 0, 0, half, half, 10, 11, 12, 0, 1, 0],
                [7, 8, 9, 0, 0, 0, 1, 7, 8, 9, 0, 0, 3],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(plan.read(), expected, atol=1e-6, rtol=1e-6)


def test_refresh_changes_values_without_replacing_simstate_storage() -> None:
    plan = mixed_plan()
    state = SimState()
    allocations: list[torch.Tensor] = []

    def allocate(_name: str, value: torch.Tensor, _device: torch.device | str | None) -> torch.Tensor:
        allocation = torch.empty_like(value)
        allocations.append(allocation)
        return allocation

    state.set_buffer_allocator(allocate)
    state.register("object_root_states", plan.read)
    held = state.get("object_root_states")
    plan.qpos[:, :3].add_(100)
    plan.xpos[:, 1].add_(200)
    state.mark_stale()
    assert state.get("object_root_states") is held and allocations == [held]
    torch.testing.assert_close(held[:, 2, :3], plan.qpos[:, :3])
    torch.testing.assert_close(held[:, 0, :3], plan.xpos[:, 1])
    assert torch.count_nonzero(held[:, 0, 7:]) == 0


def test_empty_scene_has_an_empty_object_axis() -> None:
    plan = ObjectReadPlan.build(
        (),
        {},
        {},
        torch.empty(2, 0),
        torch.empty(2, 0),
        torch.zeros(2, 1, 3),
        torch.zeros(2, 1, 4),
    )
    assert plan.read().shape == (2, 0, 13)


@pytest.mark.parametrize("names", [("robot",), ("first", "first")])
def test_invalid_object_names_are_rejected(names: tuple[str, ...]) -> None:
    plan = mixed_plan()
    with pytest.raises(ValueError, match="unique non-robot"):
        ObjectReadPlan.build(names, {}, {}, plan.qpos, plan.qvel, plan.xpos, plan.xquat)


@pytest.mark.parametrize("kind", ["free", "static"])
def test_invalid_native_indices_are_rejected(kind: str) -> None:
    plan = mixed_plan()
    addresses = {"first": {"qpos_addr": 100, "qvel_addr": 0}}
    bodies = {"first": 100} if kind == "static" else {}
    with pytest.raises(ValueError, match="Invalid"):
        ObjectReadPlan.build(("first",), addresses, bodies, plan.qpos, plan.qvel, plan.xpos, plan.xquat)
