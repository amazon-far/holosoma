"""Config-layer unit tests for ``RobotAssetConfig.link_physics`` (pure, no simulator).

``link_physics`` applies a shared ``PhysicsConfig`` to robot links cross-backend. These tests
cover the config contract:

- ``link_physics`` rejects ``mass``;
- the 5 physx/density fields live only on ``link_physics``, not as flat ``RobotAssetConfig`` fields;
- the shipped presets pin damping to ``0.0`` (``PhysXPhysicsConfig`` defaults it to ``0.1``);
- the static MuJoCo freejoint-damping machinery is absent;
- the nested ``PhysicsConfig`` survives the tyro CLI round-trip for every robot preset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

from holosoma.config_types.robot import RobotAssetConfig
from holosoma.config_types.scene import (
    IsaacSimPhysicsConfig,
    MujocoPhysicsConfig,
    PhysicsConfig,
    PhysXPhysicsConfig,
)

pytestmark = pytest.mark.no_sim

# The 13 required RobotAssetConfig fields, so each test only varies link_physics. Values are
# arbitrary since no asset is loaded here.
_BASE = {
    "asset_root": "x",
    "collapse_fixed_joints": True,
    "replace_cylinder_with_capsule": True,
    "flip_visual_attachments": False,
    "armature": 0.001,
    "thickness": 0.01,
    "urdf_file": "u.urdf",
    "usd_file": None,
    "xml_file": "x.xml",
    "robot_type": "r",
    "enable_self_collisions": False,
    "default_dof_drive_mode": 3,
    "fix_base_link": False,
}

# The five physics fields that live only on link_physics, not as flat RobotAssetConfig fields.
_MIGRATED_FIELDS = (
    "density",
    "linear_damping",
    "angular_damping",
    "max_linear_velocity",
    "max_angular_velocity",
)


# --------------------------------------------------------------------------------------------
# Validator: mass on link_physics is rejected; density is allowed.
# --------------------------------------------------------------------------------------------


def test_link_physics_mass_rejected():
    # ValidationError subclasses ValueError; match the validator's message.
    with pytest.raises(pydantic.ValidationError, match="applies to all links"):
        RobotAssetConfig(**_BASE, link_physics=PhysicsConfig(mass=2.0))


def test_link_physics_density_allowed():
    cfg = RobotAssetConfig(**_BASE, link_physics=PhysicsConfig(density=500.0))
    assert cfg.link_physics is not None
    assert cfg.link_physics.density == 500.0
    assert cfg.link_physics.mass is None


def test_link_physics_none_is_allowed():
    # Default: no link_physics keeps the asset's authored physics.
    cfg = RobotAssetConfig(**_BASE)
    assert cfg.link_physics is None


# --------------------------------------------------------------------------------------------
# Shipped presets pin the physics values, and pin them away from the PhysXPhysicsConfig default.
# --------------------------------------------------------------------------------------------


def test_presets_pin_exact_physx():
    # Import inside the test so a config_values import error surfaces here, not at collection.
    from holosoma.config_values.robot import g1_29dof, t1_29dof_waist_wrist

    default_damping = PhysXPhysicsConfig().linear_damping  # 0.1
    for preset in (g1_29dof, t1_29dof_waist_wrist):
        physx = preset.asset.link_physics.physx
        assert physx is not None
        assert physx.linear_damping == 0.0
        assert physx.angular_damping == 0.0
        assert physx.linear_damping != default_damping
        assert physx.max_linear_velocity == 1000.0
        assert physx.max_angular_velocity == 1000.0
    # t1 sets a density fallback; g1 sets none.
    assert t1_29dof_waist_wrist.asset.link_physics.density == 0.001
    assert g1_29dof.asset.link_physics.density is None


def test_w_object_inherits_link_physics():
    # g1_29dof_w_object keeps the same link_physics object as g1_29dof.
    from holosoma.config_values.robot import g1_29dof, g1_29dof_w_object

    assert g1_29dof_w_object.asset.link_physics is g1_29dof.asset.link_physics


# --------------------------------------------------------------------------------------------
# The physics fields are not flat fields on RobotAssetConfig.
# --------------------------------------------------------------------------------------------


def test_legacy_flat_fields_removed():
    # RobotAssetConfig is not extra='forbid', so an unknown kwarg is silently dropped rather than
    # raising; assert field absence rather than using pytest.raises.
    cfg = RobotAssetConfig(**_BASE)
    for name in _MIGRATED_FIELDS:
        assert name not in RobotAssetConfig.__dataclass_fields__, f"{name} resurrected as a flat field"
        assert not hasattr(cfg, name), f"{name} still present on the instance"


# --------------------------------------------------------------------------------------------
# Sub-config schemas: extra='forbid' catches typos/misplacement.
# --------------------------------------------------------------------------------------------


def test_mujoco_physics_rejects_damping():
    # MujocoPhysicsConfig has no linear_damping field. The error message names the field, so match
    # the field name rather than "extra".
    with pytest.raises(pydantic.ValidationError, match="linear_damping"):
        MujocoPhysicsConfig(linear_damping=0.0)


@pytest.mark.parametrize(
    ("ctor", "kwargs"),
    [
        (PhysXPhysicsConfig, {"linaer_damping": 0.0}),  # typo'd field
        (PhysicsConfig, {"linear_damping": 0.0}),  # field that belongs on .physx, not the top level
        (IsaacSimPhysicsConfig, {"frictio": 1.0}),  # typo'd field
    ],
)
def test_sub_configs_forbid_extra(ctor, kwargs):
    # Every physics sub-config carries _FORBID_EXTRA, so a misplaced or typo'd field fails at
    # construction.
    with pytest.raises(pydantic.ValidationError):
        ctor(**kwargs)


# --------------------------------------------------------------------------------------------
# Guard: the static freejoint-damping path is absent; the runtime damping DR term is present.
# --------------------------------------------------------------------------------------------


def test_removed_freejoint_damping_machinery_absent():
    import holosoma.config_types.scene as scene_mod

    # No object-only static type or field: MuJoCo freejoint damping is runtime-DR only.
    assert not hasattr(scene_mod, "MujocoFreeBodyConfig")
    from holosoma.config_types.scene import MujocoPhysicsConfig, RigidObjectConfig

    assert "free_body_physics" not in RigidObjectConfig.__dataclass_fields__
    for f in ("linear_damping", "angular_damping"):
        assert f not in MujocoPhysicsConfig.__dataclass_fields__


# --------------------------------------------------------------------------------------------
# No circular import: robot.py imports scene.py for PhysicsConfig; scene.py must not import robot.
# --------------------------------------------------------------------------------------------


def test_no_circular_import():
    import holosoma.config_types.scene as scene_mod

    # scene.py source contains no robot import.
    scene_text = Path(scene_mod.__file__).read_text()
    assert "config_types.robot" not in scene_text and "import robot" not in scene_text

    # Both import orders succeed in a fresh interpreter.
    for first, second in (
        ("holosoma.config_types.robot", "holosoma.config_types.scene"),
        ("holosoma.config_types.scene", "holosoma.config_types.robot"),
    ):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {first}; import {second}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"import {first} then {second} failed:\n{proc.stderr}"


# --------------------------------------------------------------------------------------------
# tyro CLI round-trip: the nested PhysicsConfig survives parsing for every robot preset.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("robot_key", ["g1-29dof", "t1-29dof-waist-wrist", "g1-29dof-w-object"])
def test_tyro_cli_round_trips_link_physics(robot_key):
    import tyro

    from holosoma.config_types.run_sim import RunSimConfig

    cfg = tyro.cli(
        RunSimConfig,
        args=["simulator:mujoco", f"robot:{robot_key}", "terrain:terrain-locomotion-plane", "scene:empty"],
    )
    # The nested PhysicsConfig survives the CLI surface.
    assert isinstance(cfg.robot.asset.link_physics, PhysicsConfig)
    assert cfg.robot.asset.link_physics.physx.linear_damping == 0.0
