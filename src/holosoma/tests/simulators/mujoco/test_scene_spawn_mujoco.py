"""Live MuJoCo (ClassicBackend, CPU) scene-spawn tests.

Builds the real simulator and asserts the cross-backend spawn contract on MuJoCo:
- an arbitrary number of FREE bodies spawn, are addressable, and fall under gravity;
- STATIC bodies spawn welded (no joint), hold their configured pose, and are read-only;
- free and static bodies coexist in one scene.

Runs in the MuJoCo (hsmujoco) env. GPU/Warp + the Isaac backends are exercised by
separate suites / the tests/simulators/scene_spawn_assert.py harness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.scene import (  # noqa: E402
    MujocoPhysicsConfig,
    ObjectPatternConfig,
    PhysicsConfig,
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)
from holosoma.simulator.shared.object_registry import ObjectType  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
SMALL_BOX_XML = "holosoma/data/scene_objects/boxes/small_box.xml"
MULTIBODY_XML = "holosoma/data/scene_objects/multibody/multibody.xml"
STEPS = 40


def _build(rigid_objects):
    """Build a sim with the given standalone rigid objects."""
    return build_classic_sim(SceneConfig(rigid_objects=rigid_objects))


def _z(sim, name):
    return float(sim.get_actor_states([name], torch.arange(sim.num_envs, device=sim.sim_device))[0, 2])


@pytest.mark.parametrize("n_free", [1, 3, 5])
def test_arbitrary_free_bodies_spawn_and_fall(n_free):
    objs = {f"free{i}": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.1 * i, 0.6]) for i in range(n_free)}
    sim = _build(objs)

    names = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    assert names == [f"free{i}" for i in range(n_free)]

    z0 = {n: _z(sim, n) for n in names}
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    for n in names:
        assert _z(sim, n) < z0[n] - 1e-3, f"{n} did not fall"


def test_static_body_holds_pose():
    sim = _build({"pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.6, 0.0, 0.3], fixed=True)})

    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["pillar"]
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == []
    assert "pillar" not in sim.object_addrs  # static: no freejoint qpos slice

    z0 = _z(sim, "pillar")
    assert abs(z0 - 0.3) < 1e-3  # spawned at its configured height
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert abs(_z(sim, "pillar") - z0) < 1e-3  # stayed put


def test_xml_format_object_spawns_and_falls():
    """MuJoCo loads MJCF (xml_file) for rigid objects, not just urdf_file."""
    sim = _build({"xbox": RigidObjectConfig(xml_file=SMALL_BOX_XML, position=[0.4, 0.0, 0.6])})

    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["xbox"]
    z0 = _z(sim, "xbox")
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert _z(sim, "xbox") < z0 - 1e-3, "xml-format object did not fall"


def test_xml_static_body_holds_pose():
    """A static (fixed=True) object loaded from MJCF (xml_file) holds its pose."""
    sim = _build({"xpillar": RigidObjectConfig(xml_file=SMALL_BOX_XML, position=[0.6, 0.0, 0.3], fixed=True)})

    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["xpillar"]
    assert "xpillar" not in sim.object_addrs  # static: no freejoint qpos slice
    z0 = _z(sim, "xpillar")
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert abs(_z(sim, "xpillar") - z0) < 1e-3  # stayed put


def test_usd_only_object_errors_clearly():
    """MuJoCo cannot load USD — a usd_file-only object must fail loud, not silently drop."""
    with pytest.raises(ValueError, match="(?i)usd"):
        _build({"usd_obj": RigidObjectConfig(usd_file="some/object.usd", position=[0.4, 0.0, 0.5])})


def test_free_and_static_mix():
    sim = _build(
        {
            "free0": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
            "free1": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
            "pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.7, 0.0, 0.3], fixed=True),
        }
    )

    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["free0", "free1"]
    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["pillar"]

    z0 = {n: _z(sim, n) for n in ("free0", "free1", "pillar")}
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert _z(sim, "free0") < z0["free0"] - 1e-3
    assert _z(sim, "free1") < z0["free1"] - 1e-3
    assert abs(_z(sim, "pillar") - z0["pillar"]) < 1e-3


def _x(sim, name):
    return float(sim.get_actor_states([name], torch.arange(sim.num_envs, device=sim.sim_device))[0, 0])


def _state(sim, name):
    """Full 13-vector actor state [pos(3), quat xyzw(4), lin_vel(3), ang_vel(3)] in env 0."""
    return sim.get_actor_states([name], torch.arange(sim.num_envs, device=sim.sim_device))[0]


def test_initial_velocity_takes_effect():
    """A free object configured with non-zero lin_vel + ang_vel reads its velocity back
    immediately AND integrates it: it translates along the gravity-free x/y axes with the
    right sign and its orientation actually changes (the spin is applied). A pure read-back
    could pass even if the sim ignored the value, so the post-step motion check is included.
    Identity initial orientation => world frame == body frame, so the config velocity matches
    MuJoCo's body-local qvel read-back directly."""
    lin, ang = [1.0, 0.5, 0.0], [0.0, 0.0, 3.0]
    sim = _build(
        {
            "vbox": RigidObjectConfig(
                urdf_file=SMALL_BOX, position=[0.0, 0.0, 0.6], linear_velocity=lin, angular_velocity=ang
            )
        }
    )

    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["vbox"]

    # Immediate read-back: the configured velocity is live in the freejoint qvel slice.
    st0 = _state(sim, "vbox")
    assert torch.allclose(st0[7:10], torch.tensor(lin), atol=1e-4), f"lin_vel read-back {st0[7:10].tolist()} != {lin}"
    assert torch.allclose(st0[10:13], torch.tensor(ang), atol=1e-4), f"ang_vel read-back {st0[10:13].tolist()} != {ang}"

    p0, q0 = st0[:3].clone(), st0[3:7].clone()
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()

    st1 = _state(sim, "vbox")
    # Linear: moved +x and +y (gravity does not touch these axes).
    assert st1[0] - p0[0] > 1e-3, f"vbox did not move +x: dx={float(st1[0] - p0[0])}"
    assert st1[1] - p0[1] > 1e-3, f"vbox did not move +y: dy={float(st1[1] - p0[1])}"
    # Angular: a non-zero spin about z must change the orientation quaternion.
    assert float((st1[3:7] - q0).abs().max()) > 1e-2, "vbox orientation did not change under ang_vel"


def _object_total_mass(sim, name):
    """Sum of the root body and its descendant subtree mass, from the live MuJoCo model."""
    import mujoco

    model = sim.backend.model
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sim.scene_manager.rigid_object_root_bodies[name])
    total = 0.0
    for bid in range(model.nbody):
        anc = bid
        while anc not in (0, root_id):
            anc = int(model.body_parentid[anc])
        if anc == root_id:
            total += float(model.body_mass[bid])
    return total


def _object_sliding_friction(sim, name):
    """Sliding friction (geom_friction axis 0) of the object's first geom, from the live model."""
    import mujoco

    model = sim.backend.model
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sim.scene_manager.rigid_object_root_bodies[name])
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == root_id:
            return float(model.geom_friction[g][0])
    return None


def test_physics_override_reaches_spawned_body():
    """A per-object physics override (configured mass + MuJoCo sliding friction) reaches the
    spawned body in the live model — the read-back path scene_spawn_assert.py checks on the GPU
    backends, run here on the ClassicBackend (the only backend without that coverage)."""
    sim = _build(
        {
            "pbox": RigidObjectConfig(
                urdf_file=SMALL_BOX,
                position=[0.0, 0.0, 0.6],
                physics=PhysicsConfig(mass=3.0, mujoco=MujocoPhysicsConfig(friction=[0.4, 0.01, 0.001])),
            )
        }
    )
    assert abs(_object_total_mass(sim, "pbox") - 3.0) < 1e-4, "configured mass did not reach the body"
    assert abs(_object_sliding_friction(sim, "pbox") - 0.4) < 1e-4, "configured friction did not reach the geom"


def test_initial_velocity_on_fixed_object_rejected():
    """Setting an initial velocity on a static (fixed=True) object is a config error — it can
    never take effect, so RigidObjectConfig rejects it loudly rather than silently dropping it."""
    with pytest.raises(ValueError, match="(?i)fixed"):
        RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.0, 0.0, 0.3], fixed=True, linear_velocity=[1.0, 0.0, 0.0])


def test_multibody_scene_file_spawns_with_types_and_offset():
    """A single scene FILE expands 1->N: free_box (free) + static_post (static), at the
    file's authored +0.5m-x relative offset, classified by the file's joint structure."""
    scene = SceneConfig(scene_files={"scene": SceneFileConfig(xml_path=MULTIBODY_XML, position=[0.4, 0.0, 0.6])})
    sim = build_classic_sim(scene)

    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["scene_free_box"]
    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["scene_static_post"]
    # Relative offset preserved (static_post authored +0.5m x from free_box).
    assert abs((_x(sim, "scene_static_post") - _x(sim, "scene_free_box")) - 0.5) < 1e-3

    z_free, z_static = _z(sim, "scene_free_box"), _z(sim, "scene_static_post")
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert _z(sim, "scene_free_box") < z_free - 1e-3  # free body fell
    assert abs(_z(sim, "scene_static_post") - z_static) < 1e-3  # static body held


def test_multibody_object_config_overrides_static_dynamic():
    """object_configs flips each body's type vs. the file: free_box forced static (drops its
    freejoint), static_post forced free (gains one). Registered types and motion follow."""
    scene = SceneConfig(
        scene_files={
            "scene": SceneFileConfig(
                xml_path=MULTIBODY_XML,
                position=[0.4, 0.0, 0.6],
                object_configs={
                    "free_box": ObjectPatternConfig(fixed=True),
                    "static_post": ObjectPatternConfig(fixed=False),
                },
            )
        }
    )
    sim = build_classic_sim(scene)

    # Types are flipped relative to the file's structure.
    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["scene_free_box"]
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["scene_static_post"]
    assert "scene_free_box" not in sim.object_addrs  # forced static: no freejoint qpos slice

    z_free, z_static = _z(sim, "scene_free_box"), _z(sim, "scene_static_post")
    for _ in range(STEPS):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
    assert abs(_z(sim, "scene_free_box") - z_free) < 1e-3  # forced static: held
    assert _z(sim, "scene_static_post") < z_static - 1e-3  # forced free: fell
