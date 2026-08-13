"""MuJoCo apply-seam tests for robot ``link_physics`` (CPU, no CUDA, no full sim).

Exercise the spawn-time seam ``MujocoSceneManager._apply_link_physics_to_robot`` /
``_apply_physics_to_body`` against a synthetic multi-link ``MjSpec``, then read the result back from
the compiled ``MjModel``. Checks that configured values land on robot link geoms, that the
whole-robot path does not write mass, and that foreign-backend sub-configs are ignored on MuJoCo.

The seam methods only touch ``self._apply_physics_to_body``, ``robot_config.asset``, and the spec,
so they are called via a tiny ``_Shim`` ``self`` plus a ``SimpleNamespace`` robot-config stub, with
no full ``MujocoSceneManager`` construction.

The seam applies the whole-robot ``link_physics`` uniformly with ``apply_mass=False`` (no per-link
override).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from types import SimpleNamespace  # noqa: E402

from holosoma.config_types.scene import (  # noqa: E402
    IsaacSimPhysicsConfig,
    MujocoPhysicsConfig,
    PhysicsConfig,
    PhysXPhysicsConfig,
)
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager  # noqa: E402

# Authored baseline geom values, distinct from anything a test applies, so "kept unchanged" and
# "overwritten" are distinguishable.
_BASE_FRICTION = [0.3, 0.005, 0.0001]
_BASE_CONDIM = 3


class _Shim:
    """A stand-in ``self`` exposing the unbound ``_apply_physics_to_body``.

    ``_apply_link_physics_to_robot`` calls ``self._apply_physics_to_body(...)``; binding it here
    drives the seam without constructing a full ``MujocoSceneManager``.
    """

    _apply_physics_to_body = MujocoSceneManager._apply_physics_to_body


def _make_robot_spec(link_masses: list[float] | None = None) -> mujoco.MjSpec:  # type: ignore[name-defined]
    """A synthetic 3-link robot: ``pelvis`` (freejoint) -> ``link_a`` (hinge) -> ``link_b`` (hinge).

    Each body carries one box geom with the authored baseline friction/condim and an explicit
    ``mass``/``inertia`` (a moving body needs nonzero mass and inertia or the compiler rejects it).
    ``link_masses`` (3 values) sets distinct per-link masses for the "mass not clobbered"
    assertions; otherwise every body gets a default mass.
    """
    masses = link_masses if link_masses is not None else [1.0, 1.0, 1.0]
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_freejoint()
    pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], friction=_BASE_FRICTION, condim=_BASE_CONDIM)

    a = pelvis.add_body(name="link_a")
    a.add_joint(name="ja", type=mujoco.mjtJoint.mjJNT_HINGE)
    a.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], friction=_BASE_FRICTION, condim=_BASE_CONDIM)

    b = a.add_body(name="link_b")
    b.add_joint(name="jb", type=mujoco.mjtJoint.mjJNT_HINGE)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], friction=_BASE_FRICTION, condim=_BASE_CONDIM)

    # A moving body (freejoint or hinge) needs mass and inertia > mjMINVAL or compile() raises;
    # body.mass with explicitinertial also requires a nonzero inertia tensor.
    for body, m in zip((pelvis, a, b), masses):
        body.mass = m
        body.inertia = [1e-2, 1e-2, 1e-2]
        body.explicitinertial = True
    return spec


def _make_density_spec() -> mujoco.MjSpec:  # type: ignore[name-defined]
    """A 3-link robot whose bodies have no explicit mass; mass is geom-density-derived.

    Used by the density test, where an explicit ``body.mass`` would override density.
    """
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_freejoint()
    pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], friction=_BASE_FRICTION, condim=_BASE_CONDIM)
    a = pelvis.add_body(name="link_a")
    a.add_joint(name="ja", type=mujoco.mjtJoint.mjJNT_HINGE)
    a.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], friction=_BASE_FRICTION, condim=_BASE_CONDIM)
    b = a.add_body(name="link_b")
    b.add_joint(name="jb", type=mujoco.mjtJoint.mjJNT_HINGE)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], friction=_BASE_FRICTION, condim=_BASE_CONDIM)
    return spec


def _robot_stub(link_physics: PhysicsConfig | None):
    return SimpleNamespace(asset=SimpleNamespace(link_physics=link_physics, robot_type="testrobot"))


def _apply_whole_robot(spec, link_physics):
    """Run the whole-robot seam (``apply_mass=False``)."""
    MujocoSceneManager._apply_link_physics_to_robot(_Shim(), spec, _robot_stub(link_physics))


def _compile(spec):
    return spec.compile()


def _all_geom_friction(model):
    return [[round(float(model.geom_friction[g, k]), 6) for k in range(3)] for g in range(model.ngeom)]


# --------------------------------------------------------------------------------------------
# Smoke fixture: the synthetic spec compiles and carries the authored baseline.
# --------------------------------------------------------------------------------------------


def test_smoke_synthetic_spec():
    model = _compile(_make_robot_spec())
    assert model.ngeom == 3
    assert model.nbody == 4  # world + 3 links
    for g in range(model.ngeom):
        assert float(model.geom_friction[g, 0]) == pytest.approx(_BASE_FRICTION[0], abs=1e-6)
        assert int(model.geom_condim[g]) == _BASE_CONDIM


# --------------------------------------------------------------------------------------------
# Whole-robot geom props reach every geom; density changes total mass.
# --------------------------------------------------------------------------------------------


def test_whole_robot_geom_props_reach_every_geom():
    lp = PhysicsConfig(mujoco=MujocoPhysicsConfig(friction=[0.9, 0.01, 0.002], condim=4))
    spec = _make_robot_spec()
    _apply_whole_robot(spec, lp)
    model = _compile(spec)
    # Every geom on every body is written.
    for g in range(model.ngeom):
        assert model.geom_friction[g, 0] == pytest.approx(0.9, abs=1e-6)
        assert model.geom_friction[g, 1] == pytest.approx(0.01, abs=1e-6)
        assert model.geom_friction[g, 2] == pytest.approx(0.002, abs=1e-6)
        assert int(model.geom_condim[g]) == 4


def test_density_routed_to_geoms_changes_mass():
    # density is a per-geom write; on a geom-mass-derived spec, total mass rises versus a no-density
    # build.
    baseline_mass = float(_compile(_make_density_spec()).body_mass.sum())

    spec = _make_density_spec()
    _apply_whole_robot(spec, PhysicsConfig(density=5000.0))
    dense_mass = float(_compile(spec).body_mass.sum())
    assert dense_mass > baseline_mass * 1.5


def test_solref_solimp_routed():
    lp = PhysicsConfig(mujoco=MujocoPhysicsConfig(solref=[0.01, 1.5], solimp=[0.8, 0.95, 0.001, 0.5, 2.0]))
    spec = _make_robot_spec()
    _apply_whole_robot(spec, lp)
    model = _compile(spec)
    for g in range(model.ngeom):
        assert model.geom_solref[g, 0] == pytest.approx(0.01, abs=1e-6)
        assert model.geom_solref[g, 1] == pytest.approx(1.5, abs=1e-6)
        assert model.geom_solimp[g, 0] == pytest.approx(0.8, abs=1e-6)


# --------------------------------------------------------------------------------------------
# The whole-robot path does not write mass.
# --------------------------------------------------------------------------------------------


def test_apply_mass_false_never_writes_mass():
    # apply_mass=False with a mass-carrying config writes the geom props but leaves body mass alone.
    spec = _make_robot_spec()
    body = spec.body("link_a")
    MujocoSceneManager._apply_physics_to_body(
        _Shim(),
        body,
        PhysicsConfig(mass=99.0, mujoco=MujocoPhysicsConfig(friction=[2.0, 0.005, 0.0001])),
        "link_a",
        apply_mass=False,
    )
    model = _compile(spec)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_a")
    assert float(model.body_mass[bid]) != pytest.approx(99.0, abs=1e-3)  # mass not written
    a_geom = model.body_geomadr[bid]
    assert model.geom_friction[a_geom, 0] == pytest.approx(2.0, abs=1e-6)  # geom prop was written


def test_whole_robot_apply_does_not_collapse_distinct_link_masses():
    # Distinct per-link masses stay distinct through the whole-robot seam (geom material only,
    # apply_mass=False).
    masses = [3.0, 2.0, 1.0]
    spec = _make_robot_spec(link_masses=masses)
    _apply_whole_robot(spec, PhysicsConfig(mujoco=MujocoPhysicsConfig(friction=[0.7, 0.005, 0.0001])))
    model = _compile(spec)
    got = [
        float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)])
        for n in ("pelvis", "link_a", "link_b")
    ]
    assert got == pytest.approx(masses, abs=1e-6)  # masses preserved, not smeared to one value


def test_apply_mass_true_writes_mass():
    # apply_mass=True writes the explicit mass and sets explicitinertial.
    spec = _make_robot_spec()
    body = spec.body("link_a")
    MujocoSceneManager._apply_physics_to_body(_Shim(), body, PhysicsConfig(mass=7.5), "link_a", apply_mass=True)
    model = _compile(spec)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_a")
    assert float(model.body_mass[bid]) == pytest.approx(7.5, abs=1e-3)


# --------------------------------------------------------------------------------------------
# Foreign sub-configs are ignored on MuJoCo; None and physx-only are no-ops.
# --------------------------------------------------------------------------------------------


def test_foreign_sub_configs_ignored():
    # physx / isaacgym / isaacsim carry a 9.0 sentinel and mujoco=None; MuJoCo reads none of them,
    # so the compiled geoms match the un-applied baseline.
    baseline = _all_geom_friction(_compile(_make_robot_spec()))
    lp = PhysicsConfig(
        physx=PhysXPhysicsConfig(linear_damping=9.0),
        isaacsim=IsaacSimPhysicsConfig(static_friction=9.0, dynamic_friction=9.0),
        mujoco=None,
    )
    spec = _make_robot_spec()
    _apply_whole_robot(spec, lp)
    assert _all_geom_friction(_compile(spec)) == baseline


def test_link_physics_none_is_noop():
    # No link_physics: the seam touches nothing.
    baseline = _all_geom_friction(_compile(_make_robot_spec()))
    spec = _make_robot_spec()
    _apply_whole_robot(spec, None)
    assert _all_geom_friction(_compile(spec)) == baseline


def test_physx_only_whole_robot_is_geom_noop():
    # link_physics is set (the loop runs) but carries only physx (mujoco=None), so the MuJoCo geom
    # write is a no-op.
    baseline = _all_geom_friction(_compile(_make_robot_spec()))
    spec = _make_robot_spec()
    _apply_whole_robot(
        spec,
        PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
    )
    assert _all_geom_friction(_compile(spec)) == baseline
