"""Joint force limits survive the CPU-model to Warp-model transfer."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco_warp
mjw = pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

from holosoma.simulator.mujoco.robot_addressing import (  # noqa: E402
    apply_dof_properties_from_config,
    read_dof_limits,
)
from holosoma.simulator.mujoco.tests.test_robot_dof_props_mujoco import (  # noqa: E402
    _DOFS,
    _JOINT_CLAMPED_MJCF,
    _compile_robot,
    _dof_addrs,
    _robot,
)


@pytest.mark.parametrize("effort", [None, [10.0, 20.0, 30.0]])
def test_joint_force_limits_reach_warp_physics(effort):
    model, meta = _compile_robot(_JOINT_CLAMPED_MJCF)
    robot = _robot(_DOFS, effort=effort)
    apply_dof_properties_from_config(model, meta, robot)
    _, _, reported = read_dof_limits(model, meta, robot)
    assert reported == (effort if effort is not None else [2.0, 1.0, 4.0])

    with wp.ScopedDevice("cuda:0"):
        warp_model = mjw.put_model(model)
        data = mjw.make_data(model, nworld=2, nconmax=32, njmax=64)
        # Positive and negative commands in separate worlds, above every clamp.
        data.ctrl.assign(np.array([[100.0] * model.nu, [-100.0] * model.nu], dtype=np.float32))
        mjw.forward(warp_model, data)
        forces = data.qfrc_actuator.numpy()[:, _dof_addrs(model, meta)]

    expected = [[3.0, 1.0, 4.0], [-2.0, -1.0, -4.0]] if effort is None else [[10, 20, 30], [-10, -20, -30]]
    np.testing.assert_allclose(forces, expected, atol=1e-6)
