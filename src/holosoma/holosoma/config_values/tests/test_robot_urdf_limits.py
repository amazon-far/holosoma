"""Configured robot limits match the named URDF joints for shipped presets and the fixed-base test arm."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.robot_properties import resolve_dof_properties
from tests.simulators._robot_presets import test_arm_fixed

pytestmark = pytest.mark.no_sim

ATOL = 1e-5
ONE_DOF_JOINT_TYPES = {"revolute", "prismatic"}

# yam's vendored URDF carries placeholder effort/velocity limits (1.0); its config lists are the contract on every
# backend (config wins), so only its position limits are compared below.
_ROBOTS: dict[str, RobotConfig] = {name: ROBOT_REGISTRY[name] for name in sorted(ROBOT_REGISTRY)}
_ROBOTS["test_arm_fixed"] = test_arm_fixed


def _urdf_joints(robot: RobotConfig) -> dict[str, tuple[str, dict[str, float]]]:
    """``{joint name: (type, {lower, upper, effort, velocity})}`` for every URDF joint carrying a ``<limit>``."""
    path = resolve_asset_path(robot.asset.urdf_file, robot.asset.asset_root)
    root = ET.parse(path).getroot()  # noqa: S314  # shipped asset files
    joints: dict[str, tuple[str, dict[str, float]]] = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        name, joint_type = joint.get("name"), joint.get("type")
        if limit is None or name is None or joint_type is None:
            continue
        joints[name] = (
            joint_type,
            {key: float(limit.get(key, "nan")) for key in ("lower", "upper", "effort", "velocity")},
        )
    return joints


@pytest.mark.parametrize(
    ("name", "dof"),
    [(name, dof) for name, robot in sorted(_ROBOTS.items()) for dof in robot.dof_names],
    ids=[f"{name}/{dof}" for name, robot in sorted(_ROBOTS.items()) for dof in robot.dof_names],
)
def test_configured_limits_match_the_named_urdf_joint(name: str, dof: str) -> None:
    robot = _ROBOTS[name]
    joints = _urdf_joints(robot)
    assert dof in joints, f"{name}: DOF '{dof}' has no limited URDF joint"
    joint_type, limit = joints[dof]
    assert joint_type in ONE_DOF_JOINT_TYPES
    overrides = resolve_dof_properties(robot)
    assert limit["lower"] == pytest.approx(overrides.position[dof][0], abs=ATOL)
    assert limit["upper"] == pytest.approx(overrides.position[dof][1], abs=ATOL)
    if robot.asset.robot_type == "yam":
        assert limit["effort"] == limit["velocity"] == 1.0  # placeholders in the source URDF
    else:
        assert limit["effort"] == pytest.approx(overrides.effort[dof], abs=ATOL)
        assert limit["velocity"] == pytest.approx(overrides.velocity[dof], abs=ATOL)
