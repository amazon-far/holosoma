"""Live CPU (ClassicBackend) check that the shipped robots satisfy the MuJoCo articulation contract.

Builds the real single-env sim for g1_29dof and t1_29dof_waist_wrist through ``load_assets`` (scene
manager -> compile -> ``_set_robot_properties`` -> ``_set_robot_joint_addressing``), so every
``robot_addressing.validate_*`` check runs against the shipped MJCF + ``RobotConfig`` pair, and asserts
the config armature/friction lists landed on the compiled model. Also checks reordered canonical joint and body lists.

Runs in the MuJoCo (hsmujoco) CPU env — no CUDA.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

import holosoma.config_values.robot as robot_values  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.utils.robot_properties import resolve_dof_properties  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402


@pytest.mark.parametrize(
    "robot",
    [robot_values.g1_29dof, robot_values.t1_29dof_waist_wrist, robot_values.yam],
    ids=["g1_29dof", "t1_29dof_waist_wrist", "yam"],
)
def test_shipped_robot_builds_and_matches_config(robot: RobotConfig) -> None:
    sim = build_classic_sim(robot=robot)
    assert sim.dof_names == list(robot.dof_names)
    assert sim.body_names == list(robot.body_names)
    assert sim.root_model.nu == len(robot.dof_names)
    assert sim.dof_qpos_addrs == list(range(sim.dof_qpos_addrs[0], sim.dof_qpos_addrs[0] + len(robot.dof_names)))
    # The config's per-DOF lists override the MJCF values on the compiled model (g1 authors
    # frictionloss=0.1, t1 armature=0.01 in the MJCF; the config lists win on every backend), and the
    # limit tensors read back from the model equal the config.
    np.testing.assert_allclose(
        sim.root_model.dof_armature[sim.dof_qvel_addrs],
        [resolve_dof_properties(robot).armature[n] for n in robot.dof_names],
    )
    np.testing.assert_allclose(
        sim.root_model.dof_frictionloss[sim.dof_qvel_addrs],
        [resolve_dof_properties(robot).friction[n] for n in robot.dof_names],
    )
    pos_limits, vel_limits, torque_limits = sim.get_dof_limits_properties()
    np.testing.assert_allclose(
        pos_limits[:, 0].cpu(), [resolve_dof_properties(robot).position[n][0] for n in robot.dof_names] or []
    )
    np.testing.assert_allclose(
        pos_limits[:, 1].cpu(), [resolve_dof_properties(robot).position[n][1] for n in robot.dof_names] or []
    )
    np.testing.assert_allclose(
        vel_limits.cpu(), [resolve_dof_properties(robot).velocity[n] for n in robot.dof_names] or []
    )
    np.testing.assert_allclose(
        torque_limits.cpu(), [resolve_dof_properties(robot).effort[n] for n in robot.dof_names] or []
    )


@pytest.mark.parametrize("field_name", ["dof_names", "body_names"])
def test_reordered_names_preserved(field_name: str) -> None:
    g1 = robot_values.g1_29dof
    swapped = list(getattr(g1, field_name))
    swapped[0], swapped[1] = swapped[1], swapped[0]
    # Values are deliberately left in their config slots: backends must respect that order.
    overrides: dict[str, Any] = {field_name: swapped}
    robot = dataclasses.replace(g1, **overrides)
    sim = build_classic_sim(robot=robot)
    assert getattr(sim, field_name) == swapped


@pytest.mark.parametrize("zero_dof", [False, True])
def test_limit_properties_are_cached_with_consistent_empty_shapes(zero_dof: bool) -> None:
    from tests.simulators._robot_presets import onelink_box, test_arm_fixed

    sim = build_classic_sim(robot=onelink_box if zero_dof else test_arm_fixed)
    try:
        first = sim.get_dof_limits_properties()
        second = sim.get_dof_limits_properties()
        assert first is second
        assert first[0].shape == (len(sim.dof_names), 2)
        assert first[1].shape == first[2].shape == (len(sim.dof_names),)
        sim.simulate_at_each_physics_step()
    finally:
        sim.close()
