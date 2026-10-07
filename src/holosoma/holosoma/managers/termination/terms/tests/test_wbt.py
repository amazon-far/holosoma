from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from holosoma.managers.termination.terms.wbt import BadTracking, BadTrackingBodyPositionZOnly
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.managers.command.terms.wbt import MotionCommand

pytestmark = pytest.mark.no_sim


def test_body_position_z_only_preserves_root_check() -> None:
    term = object.__new__(BadTrackingBodyPositionZOnly)
    term.bad_motion_body_pos_body_indexes = torch.tensor([0])
    term.bad_motion_body_pos_threshold = 0.5
    term.bad_ref_pos_threshold = 0.5

    command = cast(
        "MotionCommand",
        SimpleNamespace(
            body_pos_relative_w=torch.tensor([[[5.0, 0.0, 0.1]], [[0.0, 0.0, 0.6]]]),
            robot_body_pos_w=torch.zeros(2, 1, 3),
            ref_pos_w=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            robot_ref_pos_w=torch.zeros(2, 3),
        ),
    )

    assert BadTracking.bad_motion_body_pos(term, command).tolist() == [True, True]
    assert term.bad_motion_body_pos(command).tolist() == [False, True]
    assert term.bad_ref_pos(command).tolist() == [True, False]
