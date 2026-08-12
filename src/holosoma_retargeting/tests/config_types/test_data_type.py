"""Tests for motion-data configuration validation."""

from __future__ import annotations

import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig


@pytest.mark.parametrize("target_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_motion_data_config_rejects_invalid_target_fps(target_fps: float) -> None:
    with pytest.raises(ValueError, match="target_fps must be finite and positive"):
        MotionDataConfig(data_format="xsens", target_fps=target_fps)


@pytest.mark.parametrize(
    "window",
    [
        {"frame_start": 1},
        {"max_frames": 10},
        {"frame_start": 1, "max_frames": 10},
    ],
)
def test_motion_data_config_rejects_sparse_indices_with_frame_window(window: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="frame_indices is mutually exclusive"):
        MotionDataConfig(data_format="xsens", frame_indices=(0, 2), **window)


def test_motion_data_config_accepts_sparse_indices_without_frame_window() -> None:
    config = MotionDataConfig(data_format="xsens", target_fps=None, frame_indices=(0, 2))

    assert config.frame_indices == (0, 2)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("frame_start", -1, "frame_start must be non-negative"),
        ("max_frames", 0, "max_frames must be positive"),
        ("max_frames", -1, "max_frames must be positive"),
    ],
)
def test_motion_data_config_rejects_invalid_frame_window(
    setting: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MotionDataConfig(data_format="xsens", **{setting: value})
