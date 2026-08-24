from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
from holosoma.managers.termination.terms.wbt import BadTracking, BadTrackingZOnly, _threshold_failure, motion_ends

pytestmark = pytest.mark.no_sim


def test_motion_ends_uses_each_environment_selected_clip() -> None:
    motion_command = SimpleNamespace(
        time_steps=torch.tensor([8, 9, 18, 19]),
        motion_ids=torch.tensor([0, 0, 1, 1]),
        motion=SimpleNamespace(
            motion_end_idx=torch.tensor([10, 20]),
            motion_ends=torch.tensor([False] * 9 + [True] + [False] * 9 + [True]),
            time_step_total=20,
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_state=lambda name: motion_command if name == "motion_command" else None,
        )
    )

    assert motion_ends(env).tolist() == [False, True, False, True]


def test_motion_ends_honors_boundaries_inside_a_combined_file() -> None:
    source_clip_ends = torch.zeros(10, dtype=torch.bool)
    source_clip_ends[[4, 9]] = True
    motion_command = SimpleNamespace(
        time_steps=torch.tensor([3, 4, 8, 9]),
        motion_ids=torch.zeros(4, dtype=torch.long),
        motion=SimpleNamespace(
            motion_end_idx=torch.tensor([10]),
            motion_ends=source_clip_ends,
            time_step_total=10,
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_state=lambda name: motion_command if name == "motion_command" else None,
        )
    )

    assert motion_ends(env).tolist() == [False, True, False, True]


def test_motion_ends_rejects_a_missing_command() -> None:
    env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _name: None))

    with pytest.raises(RuntimeError, match="no stateful term"):
        motion_ends(env)


def test_motion_ends_rejects_invalid_boundary_metadata() -> None:
    motion_command = SimpleNamespace(
        time_steps=torch.tensor([0]),
        motion_ids=torch.tensor([0]),
        motion=SimpleNamespace(
            motion_end_idx=torch.tensor([2]),
            motion_ends=torch.tensor([False]),
            time_step_total=2,
        ),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _name: motion_command))

    with pytest.raises(RuntimeError, match="metadata length"):
        motion_ends(env)


def test_default_wbt_termination_leaves_motion_end_optional() -> None:
    assert "motion_end" not in g1_29dof_wbt_termination.terms


def test_tracking_thresholds_fail_closed_on_non_finite_values() -> None:
    error = torch.tensor([0.1, float("nan"), float("inf"), 0.1])
    threshold = torch.tensor([0.2, 0.2, 0.2, float("nan")])

    assert _threshold_failure(error, threshold).tolist() == [False, True, True, True]


def test_position_tracking_checks_fail_closed() -> None:
    command: Any = SimpleNamespace(
        ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
        robot_ref_pos_w=torch.zeros(2, 3),
        body_pos_relative_w=torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, float("inf"), 0.0]]]),
        robot_body_pos_w=torch.zeros(2, 1, 3),
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, float("nan")]]),
        simulator_object_pos_w=torch.zeros(2, 3),
    )
    term = BadTracking.__new__(BadTracking)
    term.bad_ref_pos_threshold = 0.5
    term.bad_motion_body_pos_threshold = 0.5
    term.bad_motion_body_pos_body_indexes = torch.tensor([0])
    term.bad_object_pos_threshold = 0.5

    assert term.bad_ref_pos(command).tolist() == [False, True]
    assert term.bad_motion_body_pos(command).tolist() == [False, True]
    assert term.bad_object_pos(command).tolist() == [False, True]

    z_term = BadTrackingZOnly.__new__(BadTrackingZOnly)
    z_term.bad_ref_pos_threshold = 0.5
    z_term.bad_motion_body_pos_threshold = 0.5
    z_term.bad_motion_body_pos_body_indexes = torch.tensor([0])
    command.ref_pos_w[1, 2] = float("nan")
    command.body_pos_relative_w[1, 0, 2] = float("nan")
    assert z_term.bad_ref_pos(command).tolist() == [False, True]
    assert z_term.bad_motion_body_pos(command).tolist() == [False, True]
