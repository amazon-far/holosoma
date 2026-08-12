"""Tests for retargeting utility functions."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity


def _toe_motion(times_s: np.ndarray, speed_mps: float) -> np.ndarray:
    joints = np.zeros((times_s.size, 2, 3), dtype=float)
    joints[:, :, 0] = speed_mps * times_s[:, None]
    return joints


@pytest.mark.parametrize(
    ("times_s", "speed_mps", "expected_sticking"),
    [
        (np.arange(4) / 30.0, 0.2, True),
        (np.arange(7) / 60.0, 0.2, True),
        (np.array([0.0, 0.017, 0.050, 0.100]), 0.2, True),
        (np.arange(4) / 30.0, 0.4, False),
    ],
)
def test_foot_sticking_uses_speed_independently_of_frame_timing(
    times_s: np.ndarray,
    speed_mps: float,
    expected_sticking: bool,
) -> None:
    contacts = extract_foot_sticking_sequence_velocity(
        _toe_motion(times_s, speed_mps),
        ["L_Toe", "R_Toe"],
        ["L_Toe", "R_Toe"],
        frame_times_s=times_s,
    )

    assert contacts[0] == {"L_Toe": False, "R_Toe": False}
    assert all(contact == {"L_Toe": expected_sticking, "R_Toe": expected_sticking} for contact in contacts[1:])


def test_foot_sticking_rejects_non_increasing_timestamps() -> None:
    times_s = np.array([0.0, 0.1, 0.1])

    with pytest.raises(ValueError, match="strictly increasing"):
        extract_foot_sticking_sequence_velocity(
            _toe_motion(times_s, 0.2),
            ["L_Toe", "R_Toe"],
            ["L_Toe", "R_Toe"],
            frame_times_s=times_s,
        )


def test_foot_sticking_default_preserves_legacy_30_fps_cutoff() -> None:
    joints = np.zeros((2, 2, 3), dtype=float)
    joints[1, :, 0] = 0.01

    contacts = extract_foot_sticking_sequence_velocity(
        joints,
        ["L_Toe", "R_Toe"],
        ["L_Toe", "R_Toe"],
    )

    assert contacts[1] == {"L_Toe": True, "R_Toe": True}
