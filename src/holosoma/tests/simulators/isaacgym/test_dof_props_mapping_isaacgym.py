"""Unit test for the IsaacGym per-DOF property mapping (CPU, no GPU, no IsaacGym SDK).

``apply_dof_properties_from_config`` (``simulator/isaacgym/physics.py``) writes the config's per-DOF
maps (limits, armature, friction) onto the asset DOF properties and preserves uncovered asset values.
The mapping is pure indexed assignment, so it is driven with a stubbed ``gymapi`` and a
structured numpy array shaped like ``gymapi`` DOF properties, without the SDK.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.no_sim

from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv  # noqa: E402
from holosoma.config_values.robot import g1_29dof  # noqa: E402
from holosoma.envs.base_task.base_task import BaseTask  # noqa: E402
from holosoma.utils.robot_properties import resolve_dof_properties  # noqa: E402

_DOF_PROPS_DTYPE = np.dtype(
    [
        ("hasLimits", np.bool_),
        ("lower", np.float32),
        ("upper", np.float32),
        ("velocity", np.float32),
        ("effort", np.float32),
        ("armature", np.float32),
        ("friction", np.float32),
    ]
)


class _FakeAssetOptions:
    """Stand-in for ``gymapi.AssetOptions`` with the pybind object's attribute set.

    ``__slots__`` lists the real ``AssetOptions`` attributes (IsaacGym Preview 4), so setting a knob the SDK
    does not have raises ``AttributeError`` here as it does on the pybind object.
    """

    __slots__ = (
        "angular_damping",
        "armature",
        "collapse_fixed_joints",
        "convex_decomposition_from_submeshes",
        "default_dof_drive_mode",
        "density",
        "disable_gravity",
        "enable_gyroscopic_forces",
        "fix_base_link",
        "flip_visual_attachments",
        "linear_damping",
        "max_angular_velocity",
        "max_linear_velocity",
        "mesh_normal_mode",
        "min_particle_mass",
        "override_com",
        "override_inertia",
        "replace_cylinder_with_capsule",
        "slices_per_cylinder",
        "tendon_limit_stiffness",
        "thickness",
        "use_mesh_materials",
        "use_physx_armature",
        "vhacd_enabled",
        "vhacd_params",
    )

    def __init__(self) -> None:
        self.default_dof_drive_mode = 0
        self.collapse_fixed_joints = False
        self.replace_cylinder_with_capsule = False
        self.flip_visual_attachments = True
        self.fix_base_link = False
        self.thickness = 0.02
        self.disable_gravity = False
        self.density = -1.0
        self.linear_damping = -1.0
        self.angular_damping = -1.0
        self.max_linear_velocity = -1.0
        self.max_angular_velocity = -1.0


@pytest.fixture
def isaacgym_physics():
    """Import ``simulator/isaacgym/physics.py`` with a stubbed ``isaacgym`` if the SDK is absent."""
    stubbed = "isaacgym" not in sys.modules
    if stubbed:
        fake_isaacgym = types.ModuleType("isaacgym")
        fake_gymapi = types.ModuleType("isaacgym.gymapi")
        fake_gymapi.Gym = object  # type: ignore[attr-defined]
        fake_gymapi.AssetOptions = _FakeAssetOptions  # type: ignore[attr-defined]
        fake_isaacgym.gymapi = fake_gymapi  # type: ignore[attr-defined]
        sys.modules["isaacgym"] = fake_isaacgym
        sys.modules["isaacgym.gymapi"] = fake_gymapi
    try:
        yield importlib.import_module("holosoma.simulator.isaacgym.physics")
    finally:
        if stubbed:
            for name in ("holosoma.simulator.isaacgym.physics", "isaacgym.gymapi", "isaacgym"):
                sys.modules.pop(name, None)


def test_writes_every_given_list_per_dof(isaacgym_physics) -> None:
    props = np.zeros(len(g1_29dof.dof_names), dtype=_DOF_PROPS_DTYPE)
    for name in ("lower", "upper", "velocity", "effort", "armature", "friction"):
        props[name][:] = 0.5  # asset-side values must be overwritten

    out = isaacgym_physics.apply_dof_properties_from_config(props, g1_29dof)

    assert out is props
    for field, values in (
        ("lower", [resolve_dof_properties(g1_29dof).position[n][0] for n in g1_29dof.dof_names]),
        ("upper", [resolve_dof_properties(g1_29dof).position[n][1] for n in g1_29dof.dof_names]),
        ("velocity", [resolve_dof_properties(g1_29dof).velocity[n] for n in g1_29dof.dof_names]),
        ("effort", [resolve_dof_properties(g1_29dof).effort[n] for n in g1_29dof.dof_names]),
        ("armature", [resolve_dof_properties(g1_29dof).armature[n] for n in g1_29dof.dof_names]),
        ("friction", [resolve_dof_properties(g1_29dof).friction[n] for n in g1_29dof.dof_names]),
    ):
        np.testing.assert_allclose(props[field], np.asarray(values, dtype=np.float32))
    assert props["hasLimits"].all()


def test_none_lists_leave_the_asset_values(isaacgym_physics) -> None:
    robot = dataclasses.replace(g1_29dof, dof_pos_limits=None, dof_effort_limits=None)
    props = np.zeros(len(robot.dof_names), dtype=_DOF_PROPS_DTYPE)
    for name in ("lower", "upper", "effort"):
        props[name][:] = 0.5

    isaacgym_physics.apply_dof_properties_from_config(props, robot)

    for name in ("lower", "upper", "effort"):
        np.testing.assert_array_equal(props[name], 0.5)
    assert not props["hasLimits"].any()  # position limits stay as the asset loaded them
    np.testing.assert_allclose(
        props["velocity"],
        np.asarray([resolve_dof_properties(robot).velocity[n] for n in robot.dof_names], dtype=np.float32),
    )


def test_reported_limits_respect_has_limits_and_canonical_order(isaacgym_physics) -> None:
    """Inactive asset endpoints cannot become FastSAC action bounds."""
    source = Path(isaacgym_physics.__file__).with_name("isaacgym.py")
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacGym")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_process_dof_props")
    # Execute the production reader without loading the unavailable native SDK or a simulator.
    namespace: dict[str, Any] = {"torch": torch, "Any": Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    sim = types.SimpleNamespace(num_dof=2, dof_ids=[1, 0], device="cpu")
    props = np.zeros(2, dtype=_DOF_PROPS_DTYPE)
    props["hasLimits"] = [False, True]
    props["lower"] = [0.0, -2.0]
    props["upper"] = [0.0, 3.0]
    props["velocity"] = [7.0, 11.0]
    props["effort"] = [13.0, 17.0]
    original = props.copy()

    assert namespace["_process_dof_props"](sim, props, 0) is props

    torch.testing.assert_close(sim.dof_pos_limits, torch.tensor([[-2.0, 3.0], [-float("inf"), float("inf")]]))
    torch.testing.assert_close(sim.dof_vel_limits, torch.tensor([11.0, 7.0]))
    torch.testing.assert_close(sim.torque_limits, torch.tensor([17.0, 13.0]))
    np.testing.assert_array_equal(props, original)
    env = cast(
        "BaseTask",
        types.SimpleNamespace(
            device="cpu",
            dof_names=["bounded", "continuous"],
            dof_pos_limits=sim.dof_pos_limits,
            robot_config=types.SimpleNamespace(
                dof_names=["bounded", "continuous"],
                control=types.SimpleNamespace(action_scale=0.5),
                default_dof_positions=lambda: {"bounded": 0.0, "continuous": 0.0},
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"unbounded for \['continuous'\]"):
        FastSACEnv(env, (), ())


def test_dof_count_mismatch_raises(isaacgym_physics) -> None:
    props = np.zeros(len(g1_29dof.dof_names) - 1, dtype=_DOF_PROPS_DTYPE)
    with pytest.raises(ValueError, match="Asset has 28 DOFs but robot_config.dof_names has 29"):
        isaacgym_physics.apply_dof_properties_from_config(props, g1_29dof)


def test_zero_dof_robot_is_a_noop(isaacgym_physics) -> None:
    robot = dataclasses.replace(
        g1_29dof,
        dof_names=[],
        dof_pos_limits={},
        dof_vel_limits={},
        dof_effort_limits={},
        dof_armature={},
        dof_joint_friction={},
        init_state=dataclasses.replace(g1_29dof.init_state, default_joint_positions={}),
        joint_tags={},
        # This zero-DOF robot has no SDK motors.
        bridge=dataclasses.replace(g1_29dof.bridge, controlled_dof_names=(), excluded_dof_names=()),
    )
    props = np.zeros(0, dtype=_DOF_PROPS_DTYPE)
    assert isaacgym_physics.apply_dof_properties_from_config(props, robot) is props


def test_build_robot_asset_options_reads_only_existing_asset_fields(isaacgym_physics) -> None:
    """Every knob copied onto AssetOptions must exist on both RobotAssetConfig (getattr) and AssetOptions (setattr)."""
    # The fake refuses unknown knobs like the pybind object, so the setattr direction is constrained too.
    unknown_knob = "link_physics"
    with pytest.raises(AttributeError):
        setattr(_FakeAssetOptions(), unknown_knob, None)
    options = isaacgym_physics.build_robot_asset_options(g1_29dof.asset)
    assert options.fix_base_link is False
    assert options.collapse_fixed_joints is True
    assert options.thickness == pytest.approx(g1_29dof.asset.thickness)  # gym stores float32
    assert options.default_dof_drive_mode == 3
    # disable_gravity is None on g1 -> the gym default is kept.
    assert options.disable_gravity is False
    fixed = dataclasses.replace(g1_29dof.asset, fix_base_link=True)
    assert isaacgym_physics.build_robot_asset_options(fixed).fix_base_link is True
