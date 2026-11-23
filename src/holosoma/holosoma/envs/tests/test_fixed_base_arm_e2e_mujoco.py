"""Run the shared fixed-base environment contract on MuJoCo Classic."""

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness

pytestmark = pytest.mark.mujoco_classic


def test_fixed_base_arm_steps_and_learns() -> None:
    run_harness(
        Path(__file__).resolve().parents[3] / "tests/simulators/fixed_base_env_assert.py",
        "--simulator",
        "mujoco",
        label="mujoco/fixed-base environment",
        timeout=900,
    )
