"""Tests for scattering robot-ordered tensors into motion body/joint order (no sim).

Two index directions share ``_body_indexes_in_motion``. Gathering with it is fine: several robot
bodies reading one motion slot is exactly what ``FAKE_BODY_NAME_ALIASES`` is for. Scattering with it
is last-write-wins, and G1/T1 list each fake foot contact point directly after the ankle link it
aliases onto -- so the contact point overwrote the ankle's own pose with one taken from a few
centimetres lower in the URDF, on a body that is in ``body_names_to_track``. Pins:

- an aliased body never clobbers the real body it aliases onto;
- motion slots no robot body covers take the clip's value, not zeros (a zero quaternion is
  degenerate for slerp);
- a genuine slot collision raises instead of silently keeping one writer;
- leading (time) dimensions pass through, since the transition scatters a whole segment at once.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionCommand
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

# Ankle link followed by its aliased fake contact point, exactly as G1 and T1 order them.
ROBOT_BODIES = [
    "pelvis",
    "left_ankle_roll_link",
    "left_foot_contact_point",
    "right_ankle_roll_link",
    "right_foot_contact_point",
]
# The motion file has no contact points, and carries an extra body the robot lacks.
MOTION_BODIES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
MOTION_JOINTS = ["hip", "knee", "ankle", "wrist"]
ROBOT_JOINTS = ["hip", "knee", "ankle"]


def _command(robot_bodies: list[str] | None = None) -> Any:
    """A MotionCommand carrying only the state the scatter maps need."""
    robot_bodies = ROBOT_BODIES if robot_bodies is None else robot_bodies
    aliases = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_bodies]

    num_motion_bodies = len(MOTION_BODIES)
    command = types.SimpleNamespace(
        device=torch.device("cpu"),
        motion=types.SimpleNamespace(
            _body_pos_w=torch.zeros(1, num_motion_bodies, 3),
            _joint_pos=torch.zeros(1, len(MOTION_JOINTS)),
        ),
        _body_indexes_in_motion=torch.tensor([MOTION_BODIES.index(name) for name in aliases], dtype=torch.long),
        _joint_indexes_in_motion=torch.tensor([MOTION_JOINTS.index(name) for name in ROBOT_JOINTS], dtype=torch.long),
    )
    MotionCommand._build_motion_scatter_map(command, robot_bodies, aliases)  # type: ignore[arg-type]
    return command


def _body_values(num_bodies: int, offset: float = 0.0) -> torch.Tensor:
    """Distinct per-body rows, so a clobber is visible in the values."""
    return (torch.arange(num_bodies, dtype=torch.float32) + 1.0 + offset).reshape(-1, 1).expand(num_bodies, 3).clone()


def test_aliased_contact_point_does_not_clobber_its_ankle():
    command = _command()
    robot_values = _body_values(len(ROBOT_BODIES))

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, robot_values)

    ankle_row = ROBOT_BODIES.index("left_ankle_roll_link")
    contact_row = ROBOT_BODIES.index("left_foot_contact_point")
    ankle_slot = MOTION_BODIES.index("left_ankle_roll_link")
    torch.testing.assert_close(motion_values[ankle_slot], robot_values[ankle_row])
    assert not torch.allclose(motion_values[ankle_slot], robot_values[contact_row])


def test_both_ankles_keep_their_own_pose():
    command = _command()
    robot_values = _body_values(len(ROBOT_BODIES))

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, robot_values)

    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        torch.testing.assert_close(motion_values[MOTION_BODIES.index(name)], robot_values[ROBOT_BODIES.index(name)])


def test_uncovered_motion_slot_takes_the_fill_value():
    command = _command()
    fill = _body_values(len(MOTION_BODIES), offset=100.0)

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, _body_values(len(ROBOT_BODIES)), fill=fill)

    torso_slot = MOTION_BODIES.index("torso_link")
    torch.testing.assert_close(motion_values[torso_slot], fill[torso_slot])


def test_uncovered_motion_slot_is_zero_without_a_fill():
    command = _command()
    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, _body_values(len(ROBOT_BODIES)))
    torch.testing.assert_close(motion_values[MOTION_BODIES.index("torso_link")], torch.zeros(3))


def test_fill_does_not_alias_the_caller_s_tensor():
    command = _command()
    fill = _body_values(len(MOTION_BODIES), offset=100.0)
    original = fill.clone()

    MotionCommand._map_robot_bodies_to_motion_order(command, _body_values(len(ROBOT_BODIES)), fill=fill)

    torch.testing.assert_close(fill, original)


def test_scatter_passes_leading_time_dimensions_through():
    command = _command()
    num_frames = 7
    robot_values = _body_values(len(ROBOT_BODIES)).unsqueeze(0).repeat(num_frames, 1, 1)
    robot_values *= torch.arange(1, num_frames + 1, dtype=torch.float32).reshape(-1, 1, 1)

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, robot_values)

    assert motion_values.shape == (num_frames, len(MOTION_BODIES), 3)
    ankle_slot = MOTION_BODIES.index("left_ankle_roll_link")
    ankle_row = ROBOT_BODIES.index("left_ankle_roll_link")
    torch.testing.assert_close(motion_values[:, ankle_slot], robot_values[:, ankle_row])


def test_scatter_broadcasts_a_single_frame_fill_over_time():
    command = _command()
    num_frames = 4
    robot_values = _body_values(len(ROBOT_BODIES)).unsqueeze(0).expand(num_frames, -1, -1)
    fill = _body_values(len(MOTION_BODIES), offset=100.0)

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, robot_values, fill=fill)

    torso_slot = MOTION_BODIES.index("torso_link")
    expected = fill[torso_slot].expand(num_frames, 3)
    torch.testing.assert_close(motion_values[:, torso_slot], expected.contiguous())


def test_two_real_bodies_on_one_slot_raise_rather_than_silently_dropping_one():
    """Only aliases are safe to drop; a real collision means the index map is wrong."""
    robot_bodies = ["pelvis", "left_ankle_roll_link", "left_ankle_roll_link"]
    with pytest.raises(RuntimeError, match="same motion body slot"):
        _command(robot_bodies)


def test_alias_is_kept_when_its_real_body_is_absent():
    """With no real ankle in the robot list the alias is the only writer, so it must survive."""
    robot_bodies = ["pelvis", "left_foot_contact_point", "right_ankle_roll_link"]
    command = _command(robot_bodies)
    robot_values = _body_values(len(robot_bodies))

    motion_values = MotionCommand._map_robot_bodies_to_motion_order(command, robot_values)

    torch.testing.assert_close(motion_values[MOTION_BODIES.index("left_ankle_roll_link")], robot_values[1])


def test_joint_scatter_fills_uncovered_columns_from_the_clip():
    command = _command()
    robot_values = torch.tensor([1.0, 2.0, 3.0])
    fill = torch.tensor([10.0, 20.0, 30.0, 40.0])

    motion_values = MotionCommand._map_robot_joints_to_motion_order(command, robot_values, fill=fill)

    # 'wrist' has no robot counterpart, so it holds the clip value.
    torch.testing.assert_close(motion_values, torch.tensor([1.0, 2.0, 3.0, 40.0]))


def test_joint_scatter_zero_fills_without_a_fill():
    command = _command()
    motion_values = MotionCommand._map_robot_joints_to_motion_order(command, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(motion_values, torch.tensor([1.0, 2.0, 3.0, 0.0]))
