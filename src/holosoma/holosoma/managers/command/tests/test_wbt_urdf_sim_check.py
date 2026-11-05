"""Tests for the startup check that the transition URDF matches the simulated robot (no sim)."""

from __future__ import annotations

import types

import pytest

from holosoma.config_values.robot import g1_29dof
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.rotations import quat_from_angle_axis, quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.urdf_kinematics import UrdfForwardKinematics

pytest.importorskip("pinocchio")
pytestmark = pytest.mark.no_sim

URDF_PATH = resolve_asset_path(g1_29dof.asset.urdf_file, g1_29dof.asset.asset_root)
BODY_NAMES = list(g1_29dof.body_names)


def _command_and_fk() -> tuple[MotionCommand, UrdfForwardKinematics]:
    """A command whose fake simulator reports the URDF's own FK at a non-default pose."""
    urdf_fk = UrdfForwardKinematics(URDF_PATH, list(g1_29dof.dof_names), BODY_NAMES)
    dof_pos = torch.linspace(-0.3, 0.3, len(g1_29dof.dof_names)).unsqueeze(0)
    # Env 0 sits away from the world origin, as it does on a terrain grid; quats xyzw.
    root_states = torch.zeros(1, 13)
    root_states[0, :3] = torch.tensor([12.0, -3.0, 0.8])
    root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.3826834, 0.9238795])
    body_pos, body_quat = urdf_fk(dof_pos, root_states[:, :3], root_states[:, 3:7])

    command = MotionCommand.__new__(MotionCommand)
    command._env = types.SimpleNamespace(  # type: ignore[assignment]
        simulator=types.SimpleNamespace(
            dof_pos=dof_pos,
            robot_root_states=root_states,
            rigid_body_pos_w=body_pos.float(),
            rigid_body_quat_w=body_quat.float(),
            _body_list=BODY_NAMES,
        )
    )
    return command, urdf_fk


def test_matching_urdf_passes() -> None:
    command, urdf_fk = _command_and_fk()
    command._check_urdf_fk_matches_sim(urdf_fk, URDF_PATH)


def test_a_millimeter_body_offset_raises_and_names_the_body() -> None:
    command, urdf_fk = _command_and_fk()
    worst = BODY_NAMES.index("left_wrist_yaw_link")
    command._env.simulator.rigid_body_pos_w[0, worst, 2] += 1e-3

    with pytest.raises(RuntimeError, match="worst body left_wrist_yaw_link"):
        command._check_urdf_fk_matches_sim(urdf_fk, URDF_PATH)


def test_a_rotated_body_frame_raises() -> None:
    # Same origin, frame turned 0.05 rad about z: what a USD with a re-oriented link frame looks like.
    command, urdf_fk = _command_and_fk()
    turn = quat_from_angle_axis(torch.tensor(0.05), torch.tensor([0.0, 0.0, 1.0]), w_last=True)
    quat = command._env.simulator.rigid_body_quat_w
    quat[0, 0] = quat_mul(quat[0, 0], turn, w_last=True)

    with pytest.raises(RuntimeError, match=f"worst body {BODY_NAMES[0]}"):
        command._check_urdf_fk_matches_sim(urdf_fk, URDF_PATH)
