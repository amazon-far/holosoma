"""``IsaacSimStateAdapter.write_robot_root_states``: the single IsaacSim robot root writer (no sim).

Pure torch over a recording stand-in for the IsaacLab ``Articulation``. Pins the xyzw -> wxyz conversion of
the written pose, the fixed-base rule (velocity columns ignored; a pose equal to the current root pose --
``q`` and ``-q`` alike -- writes nothing; a pose change writes the pose once, compared in wxyz against
``robot.data.root_state_w``) and the floating-base path (pose and velocity both written). Both
``set_actor_root_state_tensor_robots`` and ``set_actor_states(["robot"], ...)`` end here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter

pytestmark = pytest.mark.no_sim

NUM_ENVS = 4
ENV_IDS = torch.tensor([1, 3])
# A 90 deg yaw: the xyzw and wxyz spellings differ, so a compare done in the wrong order cannot look equal.
YAW90_XYZW = [0.0, 0.0, 0.70710678, 0.70710678]
YAW90_WXYZ = [0.70710678, 0.0, 0.0, 0.70710678]
VELOCITY = [0.3, 0.0, 0.0, 0.0, 0.0, 0.5]


class _Articulation:
    """Records the two IsaacLab root writers; ``data.root_state_w`` is the wxyz state the sim reports."""

    def __init__(self, root_state_w: torch.Tensor) -> None:
        self.data = SimpleNamespace(root_state_w=root_state_w)
        self.pose_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.velocity_writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_writes.append((pose.clone(), env_ids.clone()))

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_writes.append((velocity.clone(), env_ids.clone()))


def _current_wxyz() -> torch.Tensor:
    """``[NUM_ENVS, 13]`` wxyz root states: env i at x = i, the 90 deg yaw, zero velocity."""
    state = torch.zeros(NUM_ENVS, 13)
    state[:, 0] = torch.arange(NUM_ENVS, dtype=torch.float)
    state[:, 3:7] = torch.tensor(YAW90_WXYZ)
    return state


def _current_xyzw(env_ids: torch.Tensor) -> torch.Tensor:
    """The same rows as :func:`_current_wxyz` for ``env_ids``, in holosoma's xyzw convention."""
    state = _current_wxyz()[env_ids]
    state[:, 3:7] = torch.tensor(YAW90_XYZW)
    return state


def _adapter(*, fixed_base: bool) -> tuple[IsaacSimStateAdapter, _Articulation]:
    robot = _Articulation(_current_wxyz())
    registry = SimpleNamespace(get_object_type=lambda _name: "robot")
    adapter = IsaacSimStateAdapter(
        torch.device("cpu"),
        cast("Any", registry),
        scene=None,
        robot=robot,
        robot_states=cast("Any", None),
        robot_fixed_base=fixed_base,
    )
    return adapter, robot


def test_fixed_base_same_pose_with_velocity_writes_nothing() -> None:
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[:, 7:] = torch.tensor(VELOCITY)

    adapter.write_robot_root_states(states, ENV_IDS)

    assert robot.pose_writes == []
    assert robot.velocity_writes == []
    assert adapter.is_dirty() is False


def test_fixed_base_negated_quaternion_is_the_same_pose() -> None:
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[:, 3:7] = -states[:, 3:7]

    adapter.write_robot_root_states(states, ENV_IDS)

    assert robot.pose_writes == []
    assert robot.velocity_writes == []


def test_fixed_base_pose_change_writes_the_pose_once_in_wxyz_and_no_velocity() -> None:
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[:, 2] += 0.5
    states[:, 7:] = torch.tensor(VELOCITY)

    adapter.write_robot_root_states(states, ENV_IDS)

    assert len(robot.pose_writes) == 1
    pose, env_ids = robot.pose_writes[0]
    assert torch.equal(env_ids, ENV_IDS)
    expected = _current_wxyz()[ENV_IDS, :7]
    expected[:, 2] += 0.5
    assert torch.allclose(pose, expected)
    assert robot.velocity_writes == []


def test_fixed_base_compare_is_done_in_wxyz() -> None:
    # An xyzw row that spells the current WXYZ numbers is a different rotation (a 90 deg roll); a compare
    # that skipped the conversion would call it the same pose and drop the write.
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[:, 3:7] = torch.tensor(YAW90_WXYZ)

    adapter.write_robot_root_states(states, ENV_IDS)

    assert len(robot.pose_writes) == 1
    pose, _ = robot.pose_writes[0]
    # xyzw [w, 0, 0, w] -> wxyz [w, w, 0, 0]
    assert torch.allclose(pose[:, 3:7], torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0]] * len(ENV_IDS)))


def test_fixed_base_one_changed_row_writes_the_moved_env_its_new_pose() -> None:
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[1, 0] += 0.25  # only env 3 moves

    adapter.write_robot_root_states(states, ENV_IDS)

    assert robot.velocity_writes == []
    # Per env id, however the writer batches its rows: env 3 gets its new pose in wxyz, an unchanged env written
    # along with it gets its current pose, and no env outside the request is written.
    written = {int(env_id): row for pose, env_ids in robot.pose_writes for row, env_id in zip(pose, env_ids)}
    assert 3 in written and set(written) <= set(ENV_IDS.tolist())
    moved = _current_wxyz()[3, :7].clone()
    moved[0] += 0.25
    assert torch.allclose(written[3], moved)
    if 1 in written:
        assert torch.allclose(written[1], _current_wxyz()[1, :7])


def test_floating_base_writes_pose_in_wxyz_and_velocity() -> None:
    adapter, robot = _adapter(fixed_base=False)
    states = _current_xyzw(ENV_IDS)
    states[:, 7:] = torch.tensor(VELOCITY)

    adapter.write_robot_root_states(states, ENV_IDS)

    assert len(robot.pose_writes) == 1 and len(robot.velocity_writes) == 1
    pose, pose_env_ids = robot.pose_writes[0]
    velocity, velocity_env_ids = robot.velocity_writes[0]
    assert torch.equal(pose_env_ids, ENV_IDS) and torch.equal(velocity_env_ids, ENV_IDS)
    assert torch.allclose(pose, _current_wxyz()[ENV_IDS, :7])
    assert torch.allclose(velocity, torch.tensor([VELOCITY] * len(ENV_IDS)))
    assert adapter.is_dirty() is False  # the direct path does not mark the adapter dirty


def test_write_object_states_robot_routes_to_the_root_writer_and_marks_dirty() -> None:
    adapter, robot = _adapter(fixed_base=True)
    states = _current_xyzw(ENV_IDS)
    states[:, 2] += 0.5

    adapter.write_object_states("robot", states, ENV_IDS)

    assert len(robot.pose_writes) == 1
    assert robot.velocity_writes == []
    assert adapter.is_dirty() is True
