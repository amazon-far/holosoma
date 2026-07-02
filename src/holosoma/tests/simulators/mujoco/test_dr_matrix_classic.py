"""Full cross-backend DR matrix on the MuJoCo ClassicBackend (CPU, single-env).

Builds ONE real, fully-managed locomotion env (real action_manager + randomization_manager, no
shims) and runs every robot + object DR term against it via the shared ``_dr_matrix`` checks,
reading mutated fields back through the ``dr_matrix_assert`` MuJoCo reader. This is the
classic-CPU column of the matrix; the GPU columns (mjwarp / isaacgym / isaacsim) run the SAME
``dr_matrix_assert`` harness in a subprocess under their own launcher.

The MuJoCo ClassicBackend is safe to build in-process, so this runs the harness pieces directly
(no subprocess) and needs no CUDA — it executes in the MuJoCo (hsmujoco) CPU env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.config_values import simulator as sim_values  # noqa: E402
from tests.simulators import _dr_matrix as dr  # noqa: E402
from tests.simulators.dr_matrix_assert import _MujocoReader  # noqa: E402


def test_dr_matrix_classic_cpu():
    with dr.build_full_env(sim_values.mujoco, num_envs=1, device="cpu") as env:
        reader = _MujocoReader(env.simulator)
        dr.run_robot_dr(env, reader)
        dr.run_push_dr(env)
        dr.run_object_dr(env, reader)
        dr.run_distribution_dr(env, reader)
