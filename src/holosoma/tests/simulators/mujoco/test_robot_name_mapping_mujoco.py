from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.mujoco_classic
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402


def test_canonical_names_and_order_reach_physics():
    run_harness(
        Path(__file__).resolve().parents[1] / "robot_name_mapping_assert.py",
        "--simulator",
        "mujoco",
        label="mujoco/canonical robot names",
        timeout=900,
    )
