"""Unit tests for the IsaacGym PhysicsConfig -> shape/asset mapping (CPU, no GPU, no IsaacGym SDK).

``apply_rigid_shape_properties`` and ``apply_physx_asset_options`` (``simulator/isaacgym/physics.py``)
translate a ``PhysicsConfig`` into IsaacGym physics, used identically by the robot and object paths.
The mapping is pure attribute assignment, so it is driven with a stubbed ``gymapi`` plus fake
shape-props / options objects, without the IsaacGym SDK or a GPU.

Covers the friction-family fields (``rolling_friction``/``torsion_friction``/``compliance``) and the
load-time physx/density options that behavioral tests cannot discriminate.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.no_sim

from holosoma.config_types.scene import (  # noqa: E402
    IsaacGymPhysicsConfig,
    PhysicsConfig,
    PhysXPhysicsConfig,
)


@pytest.fixture
def isaacgym_physics():
    """Import ``simulator/isaacgym/physics.py`` with a stubbed ``isaacgym`` if the SDK is absent.

    ``physics.py`` does ``from isaacgym import gymapi`` only for a type hint, so a bare stub module
    satisfies it on a no-GPU machine. The stub is installed only when the SDK is missing and is
    removed in teardown so it cannot leak into sibling tests' ``importorskip``. With the SDK present
    this is a plain import.
    """
    stubbed = "isaacgym" not in sys.modules
    if stubbed:
        fake_isaacgym = types.ModuleType("isaacgym")
        fake_gymapi = types.ModuleType("isaacgym.gymapi")
        fake_gymapi.Gym = object
        fake_isaacgym.gymapi = fake_gymapi
        sys.modules["isaacgym"] = fake_isaacgym
        sys.modules["isaacgym.gymapi"] = fake_gymapi
    try:
        physics = importlib.import_module("holosoma.simulator.isaacgym.physics")
        yield physics
    finally:
        if stubbed:
            # Drop the stub and the module imported against it.
            for name in ("holosoma.simulator.isaacgym.physics", "isaacgym.gymapi", "isaacgym"):
                sys.modules.pop(name, None)


class _Shape:
    """Mutable stand-in for ``gymapi.RigidShapeProperties`` (the 5 friction-family fields)."""

    def __init__(self):
        self.friction = None
        self.rolling_friction = None
        self.torsion_friction = None
        self.restitution = None
        self.compliance = None


class _FakeGym:
    """Duck-typed ``gymapi.Gym`` exposing only the two shape-prop calls the seam uses."""

    def __init__(self, n_shapes: int = 2):
        self.props = [_Shape() for _ in range(n_shapes)]
        self.set_called = False

    def get_actor_rigid_shape_properties(self, env_ptr, actor_handle):
        return self.props

    def set_actor_rigid_shape_properties(self, env_ptr, actor_handle, props):
        assert props is self.props, "must write back the same (mutated) shape-props list"
        self.set_called = True


class _Options:
    """Stand-in for ``gymapi.AssetOptions`` or ``urdf_scene_loader.AssetConfig`` (both name the
    physx/density fields identically)."""

    def __init__(self):
        self.density = -1.0
        self.linear_damping = -1.0
        self.angular_damping = -1.0
        self.max_linear_velocity = -1.0
        self.max_angular_velocity = -1.0


# --------------------------------------------------------------------------------------------
# apply_rigid_shape_properties: the full friction family, including rolling, torsion, and compliance.
# --------------------------------------------------------------------------------------------


def test_rigid_shape_properties_maps_full_friction_family(isaacgym_physics):
    gym = _FakeGym(n_shapes=3)
    cfg = PhysicsConfig(
        isaacgym=IsaacGymPhysicsConfig(
            friction=0.7, rolling_friction=0.11, torsion_friction=0.22, restitution=0.33, compliance=0.44
        )
    )
    isaacgym_physics.apply_rigid_shape_properties(gym, 0, 0, cfg, "robot")
    assert gym.set_called, "modified shape-props were never written back"
    # Every shape gets all five fields.
    for s in gym.props:
        assert s.friction == 0.7
        assert s.rolling_friction == 0.11  # otherwise untested
        assert s.torsion_friction == 0.22  # otherwise untested
        assert s.restitution == 0.33
        assert s.compliance == 0.44  # otherwise untested


def test_rigid_shape_properties_noop_without_isaacgym_subconfig(isaacgym_physics):
    # No isaacgym sub-config: the asset's authored shape props are kept (no write back).
    gym = _FakeGym()
    isaacgym_physics.apply_rigid_shape_properties(gym, 0, 0, PhysicsConfig(physx=PhysXPhysicsConfig()), "robot")
    assert not gym.set_called, "must NOT write shape props when isaacgym sub-config is absent"
    for s in gym.props:
        assert s.friction is None  # untouched


def test_rigid_shape_properties_raises_when_no_shapes(isaacgym_physics):
    gym = _FakeGym(n_shapes=0)
    with pytest.raises(RuntimeError, match="No rigid shape properties"):
        isaacgym_physics.apply_rigid_shape_properties(
            gym, 0, 0, PhysicsConfig(isaacgym=IsaacGymPhysicsConfig()), "robot"
        )


# --------------------------------------------------------------------------------------------
# apply_physx_asset_options: the shared load-time density and physx mapping (robot and object).
# --------------------------------------------------------------------------------------------


def test_physx_asset_options_maps_density_and_physx(isaacgym_physics):
    opts = _Options()
    isaacgym_physics.apply_physx_asset_options(
        opts,
        PhysicsConfig(
            density=250.0,
            physx=PhysXPhysicsConfig(
                linear_damping=0.4, angular_damping=0.5, max_linear_velocity=12.0, max_angular_velocity=9.0
            ),
        ),
    )
    assert opts.density == 250.0
    assert opts.linear_damping == 0.4
    assert opts.angular_damping == 0.5
    assert opts.max_linear_velocity == 12.0
    assert opts.max_angular_velocity == 9.0


def test_physx_asset_options_partial_keeps_untouched(isaacgym_physics):
    # density set, physx unset: only density is written; the four physx knobs keep their prior value.
    opts = _Options()
    isaacgym_physics.apply_physx_asset_options(opts, PhysicsConfig(density=99.0))
    assert opts.density == 99.0
    assert opts.linear_damping == -1.0  # untouched (no physx sub-config)
    assert opts.max_linear_velocity == -1.0


def test_physx_asset_options_none_is_noop(isaacgym_physics):
    opts = _Options()
    isaacgym_physics.apply_physx_asset_options(opts, None)
    assert opts.density == -1.0 and opts.linear_damping == -1.0  # nothing written
    # Empty PhysicsConfig (no density, no physx) is also a no-op.
    isaacgym_physics.apply_physx_asset_options(opts, PhysicsConfig())
    assert opts.density == -1.0 and opts.linear_damping == -1.0
