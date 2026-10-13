"""Live CPU (ClassicBackend) run of the fixed-base robot harness.

Routes the MuJoCo ClassicBackend through fixed_base_robot_assert.py: the ``test-arm-fixed`` robot
(``asset.fix_base_link=True``) spawns with a constant ``[num_envs, 13]`` root state, its base stays
put under zero torque while the joints fall, and root-state / actor-state writes teleport the base
by pose and ignore velocity -- the static-scene-object contract applied to a robot. One sim per
subprocess; the verdict is the ``--result-file`` sentinel.

Runs in the MuJoCo (hsmujoco) CPU env -- no CUDA. The Warp GPU analogue is test_fixed_base_warp.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_robot_assert.py"


def test_fixed_base_robot_classic(tmp_path: Path) -> None:
    result_file = tmp_path / "fixed_base_mujoco.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--result-file",
        str(result_file),
        label="mujoco/fixed-base",
        timeout=300,
        result_file=result_file,
    )
