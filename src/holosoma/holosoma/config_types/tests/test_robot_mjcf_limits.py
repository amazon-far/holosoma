"""Compiled shipped robots enforce named config overrides, including differences from authored MJCF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mujoco

    from holosoma.config_types.robot import RobotConfig
else:
    # no_sim lane runs in an env without the mujoco package (isaacsim conda); skip when unavailable.
    mujoco = pytest.importorskip("mujoco")

from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.simulator.mujoco.robot_addressing import apply_dof_properties_from_config
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.robot_properties import resolve_dof_properties
from tests.simulators._robot_presets import test_arm_fixed

pytestmark = pytest.mark.no_sim

ATOL = 1e-5  # the IsaacGym load-time assertion tolerance, shared with the URDF test
ONE_DOF_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}

# yam's MJCF authors no motors and no ctrlrange (both come from the config at load), so it is not compared here.
_ROBOTS: dict[str, RobotConfig] = {
    name: ROBOT_REGISTRY[name] for name in sorted(ROBOT_REGISTRY) if ROBOT_REGISTRY[name].asset.robot_type != "yam"
}
_ROBOTS["test_arm_fixed"] = test_arm_fixed


def _compile(robot: RobotConfig) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(resolve_asset_path(robot.asset.xml_file, robot.asset.asset_root))
    # The robot files' <contact><pair> entries name the scene floor geom the loader's world supplies; drop them.
    for pair in list(spec.pairs):
        spec.delete(pair)
    manager = MujocoSceneManager.__new__(MujocoSceneManager)
    meta = manager._capture_spec_meta(spec, "", str(spec.worldbody.bodies[0].name))
    model = spec.compile()
    apply_dof_properties_from_config(model, meta, robot)
    return model


@pytest.mark.parametrize("name", sorted(_ROBOTS))
def test_joint_ranges_enforce_configured_limits(name: str) -> None:
    robot = _ROBOTS[name]
    model = _compile(robot)
    overrides = resolve_dof_properties(robot)
    for dof in robot.dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof)
        assert jid != -1, f"{name}: DOF '{dof}' has no MJCF joint"
        assert model.jnt_type[jid] in ONE_DOF_JOINT_TYPES, f"{name}: DOF '{dof}' is not a hinge/slide joint"
        assert model.jnt_limited[jid], f"{name}: DOF '{dof}' has no joint range"
        expected = overrides.position[dof]
        np.testing.assert_allclose(model.jnt_range[jid], expected, atol=ATOL, err_msg=f"{name}: {dof}")


@pytest.mark.parametrize("name", sorted(_ROBOTS))
def test_motor_ctrlranges_enforce_configured_effort(name: str) -> None:
    robot = _ROBOTS[name]
    model = _compile(robot)
    for dof, effort in resolve_dof_properties(robot).effort.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, dof)
        assert aid != -1, f"{name}: DOF '{dof}' has no motor named like it"
        assert model.actuator_ctrllimited[aid], f"{name}: motor '{dof}' has no ctrlrange"
        np.testing.assert_allclose(
            model.actuator_ctrlrange[aid], [-effort, effort], atol=ATOL, err_msg=f"{name}: {dof}"
        )
