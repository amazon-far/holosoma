"""Tests for motion-conversion viewer compatibility."""

from __future__ import annotations

import pytest
import tyro
from holosoma_retargeting.config_types.data_conversion import DataConversionConfig


@pytest.mark.parametrize(
    ("headless", "visualize", "show_viewer", "exit_after_one_pass"),
    [
        (False, True, True, False),
        (True, True, False, True),
        (False, False, False, True),
        (True, False, False, True),
    ],
)
def test_headless_and_visualize_share_one_execution_mode(
    headless: bool,
    visualize: bool,
    show_viewer: bool,
    exit_after_one_pass: bool,
) -> None:
    config = DataConversionConfig(
        input_file="motion.npz",
        headless=headless,
        visualize=visualize,
    )

    assert config.show_viewer is show_viewer
    assert config.exit_after_one_pass is exit_after_one_pass


def test_once_exits_after_one_visible_pass() -> None:
    config = DataConversionConfig(input_file="motion.npz", once=True)

    assert config.show_viewer is True
    assert config.exit_after_one_pass is True


@pytest.mark.parametrize("flag", ["--headless", "--no-visualize"])
def test_cli_accepts_both_no_viewer_flags(flag: str) -> None:
    config = tyro.cli(DataConversionConfig, args=["--input-file", "motion.npz", flag])

    assert config.show_viewer is False
    assert config.exit_after_one_pass is True
