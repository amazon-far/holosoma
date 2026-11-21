"""``RobotConfig`` per-DOF limit lists equal the URDF ``<limit>`` values (pure XML parse, no simulator).

IsaacGym reads the limits from the URDF and asserts them equal to the config within 1e-5 at load
(``simulator/isaacgym/isaacgym.py``); MuJoCo and IsaacSim build their limit tensors from the config lists
and never compare. This pins the same contract on the CPU for every shipped preset and the fixed-base test
arm, so a preset/URDF drift fails here instead of only on the GPU lane. Every DOF must also be a named
1-DOF URDF joint (revolute or prismatic).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.robot_properties import resolve_dof_properties
from tests.simulators._robot_presets import test_arm_fixed

pytestmark = pytest.mark.no_sim

ATOL = 1e-5  # the IsaacGym load-time assertion tolerance
ONE_DOF_JOINT_TYPES = {"revolute", "prismatic"}

# yam's vendored URDF carries placeholder effort/velocity limits (1.0); its config lists are the contract on every
# backend (config wins), so only its position limits are compared below.
_ROBOTS: dict[str, RobotConfig] = {
    name: ROBOT_REGISTRY[name] for name in sorted(ROBOT_REGISTRY) if ROBOT_REGISTRY[name].asset.robot_type != "yam"
}
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


@pytest.mark.parametrize("name", sorted(_ROBOTS))
def test_dof_limit_lists_equal_the_urdf_limits(name: str) -> None:
    robot = _ROBOTS[name]
    joints = _urdf_joints(robot)

    missing = [dof for dof in robot.dof_names if dof not in joints]
    assert missing == [], f"{name}: dof_names without a limited URDF joint: {missing}"
    overrides = resolve_dof_properties(robot)
    for dof in robot.dof_names:
        joint_type, limit = joints[dof]
        assert joint_type in ONE_DOF_JOINT_TYPES, f"{name}: DOF '{dof}' is a {joint_type} joint"
        assert limit["lower"] == pytest.approx(overrides.position[dof][0], abs=ATOL), dof
        assert limit["upper"] == pytest.approx(overrides.position[dof][1], abs=ATOL), dof
        assert limit["effort"] == pytest.approx(overrides.effort[dof], abs=ATOL), dof
        assert limit["velocity"] == pytest.approx(overrides.velocity[dof], abs=ATOL), dof


def test_yam_position_limits_equal_the_urdf_and_the_effort_velocity_come_from_the_config() -> None:
    robot = ROBOT_REGISTRY["yam"]
    joints = _urdf_joints(robot)
    assert list(joints) == list(robot.dof_names)
    overrides = resolve_dof_properties(robot)
    for dof in robot.dof_names:
        joint_type, limit = joints[dof]
        assert joint_type in ONE_DOF_JOINT_TYPES, dof
        assert limit["lower"] == pytest.approx(overrides.position[dof][0], abs=ATOL), dof
        assert limit["upper"] == pytest.approx(overrides.position[dof][1], abs=ATOL), dof
        assert limit["effort"] == limit["velocity"] == 1.0, dof  # placeholders in the source URDF
