"""Live MuJoCo ClassicBackend (CPU) run of dof_order_assert.py: DOF state and torques follow
``robot_config.dof_names`` when it differs from the MJCF joint order. One sim per subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "dof_order_assert.py"


def test_dof_order():
    run_harness(_HARNESS, "--simulator", "mujoco", label="mujoco/dof-order", timeout=600)
