from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
pytest.importorskip("isaaclab")

from tests.simulators._run_harness import run_harness  # noqa: E402


def test_canonical_names_and_order_reach_physics():
    run_harness(
        Path(__file__).resolve().parents[1] / "robot_name_mapping_assert.py",
        "--simulator",
        "isaacsim",
        label="isaacsim/canonical robot names",
        timeout=900,
    )
