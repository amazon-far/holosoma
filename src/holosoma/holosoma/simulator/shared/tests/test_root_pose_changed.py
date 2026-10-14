"""``root_pose_changed``: the one same-pose / pose-change rule every backend's fixed-base root write keys on.

MuJoCo and IsaacSim treat a same-pose write as a no-op and a change as a kinematic teleport; IsaacGym
refuses a change with ``NotImplementedError``. Pure torch, no simulator.
"""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.shared.root_states import ROOT_POSE_ATOL, root_pose_changed

pytestmark = pytest.mark.no_sim


def _poses() -> torch.Tensor:
    """Three ``[pos, quat_xyzw]`` rows with non-identity, non-symmetric quaternions."""
    quats = torch.tensor([[0.0, 0.6, 0.0, 0.8], [0.36, 0.48, 0.0, 0.8], [0.0, 0.0, 0.6, -0.8]])
    pos = torch.tensor([[0.0, 0.0, 0.5], [1.0, -2.0, 0.3], [5.0, 5.0, 5.0]])
    return torch.cat([pos, quats], dim=1)


def test_root_pose_atol_is_the_documented_value() -> None:
    assert ROOT_POSE_ATOL == 1e-5


def test_empty_input_is_unchanged() -> None:
    assert root_pose_changed(torch.zeros(0, 7), torch.zeros(0, 7)) is False


def test_identical_poses_are_unchanged() -> None:
    current = _poses()
    assert root_pose_changed(current.clone(), current) is False


def test_negated_quaternion_is_the_same_rotation() -> None:
    current = _poses()
    requested = current.clone()
    requested[:, 3:7] = -current[:, 3:7]
    assert root_pose_changed(requested, current) is False
    # Mixed signs within a row (rows 0 and 2 negated, row 1 not) are still all the same rotation.
    requested[1, 3:7] = current[1, 3:7]
    assert root_pose_changed(requested, current) is False


def test_flipping_one_quaternion_component_is_a_change() -> None:
    """Only a whole-row negation is equivalent; a single flipped component is a different rotation."""
    current = _poses()
    requested = current.clone()
    requested[0, 6] = -requested[0, 6]  # w only, xyz kept
    assert root_pose_changed(requested, current) is True


@pytest.mark.parametrize("column", [0, 1, 2], ids=["x", "y", "z"])
def test_position_tolerance_boundary(column: int) -> None:
    current = _poses()
    inside = current.clone()
    inside[1, column] += 0.5 * ROOT_POSE_ATOL
    assert root_pose_changed(inside, current) is False
    outside = current.clone()
    outside[1, column] += 2.0 * ROOT_POSE_ATOL
    assert root_pose_changed(outside, current) is True


@pytest.mark.parametrize("column", [3, 4, 5, 6], ids=["qx", "qy", "qz", "qw"])
def test_quaternion_tolerance_boundary(column: int) -> None:
    current = _poses()
    inside = current.clone()
    inside[2, column] += 0.5 * ROOT_POSE_ATOL
    assert root_pose_changed(inside, current) is False
    outside = current.clone()
    outside[2, column] += 2.0 * ROOT_POSE_ATOL
    assert root_pose_changed(outside, current) is True


def test_one_changed_row_in_a_batch_is_a_change() -> None:
    current = _poses()
    requested = current.clone()
    requested[2, 2] += 0.5  # teleport env 2 only
    assert root_pose_changed(requested, current) is True
    assert root_pose_changed(requested[:2], current[:2]) is False


def test_custom_atol_is_honoured() -> None:
    current = _poses()
    requested = current.clone()
    requested[0, 0] += 1e-3
    assert root_pose_changed(requested, current) is True
    assert root_pose_changed(requested, current, atol=1e-2) is False


@pytest.mark.parametrize("column", range(7), ids=["x", "y", "z", "qx", "qy", "qz", "qw"])
def test_difference_of_exactly_atol_is_unchanged(column: int) -> None:
    """The contract is 'more than atol' (``<=``): a difference equal to atol is the same pose, the next float is a
    change. The probed column holds 0 so ``0 + atol`` is exact in float32 (``1.0 + 1e-5 - 1.0`` is not)."""
    current = torch.zeros(1, 7)
    current[0, 3 if column == 6 else 6] = 1.0  # a unit quaternion with 0 in the probed column
    edge = current.clone()
    edge[0, column] = ROOT_POSE_ATOL
    assert root_pose_changed(edge, current) is False
    edge[0, column] = torch.nextafter(torch.tensor(ROOT_POSE_ATOL), torch.tensor(1.0)).item()
    assert root_pose_changed(edge, current) is True
