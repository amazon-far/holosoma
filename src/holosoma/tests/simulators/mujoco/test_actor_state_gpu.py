"""GPU (WarpBackend, multi-env) actor-state tests.

Cover behaviors the CPU/MjSpec unit tests cannot reach because they require live
per-environment GPU state. Skipped automatically when no CUDA device or the
MuJoCo-Warp stack is available.

Erroneous behaviors guarded against (all silent — wrong values, not exceptions):
- set_actor_states writing only a CPU snapshot is a no-op on the WarpBackend, whose
  live state is the GPU qpos/qvel: the write must reach the GPU state.
- get_actor_states reading one CPU value and broadcasting it returns the same value
  for every environment: it must return each environment's own GPU state.
- A get/set angular-velocity frame mismatch fails to round-trip at non-identity
  orientation: set and get must use the same (body-local) frame.
- An object initialized only on the CPU and tiled to the GPU spawns at the same world
  point in every environment: each env's object must sit at its own env_origin.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend multi-env tests require a CUDA device", allow_module_level=True)

import numpy as np  # noqa: E402

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig, SceneFileConfig  # noqa: E402
from holosoma.simulator.shared.object_registry import ObjectType  # noqa: E402
from tests.simulators.mujoco._build import build_warp_sim  # noqa: E402

BOX_URDF = "holosoma/data/scene_objects/boxes/large_box.urdf"
SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
MULTIBODY_XML = "holosoma/data/scene_objects/multibody/multibody.xml"
NUM_ENVS = 4
BOX_POS = (0.4, 0.0, 0.5)


def _default_scene():
    return SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=BOX_URDF, position=list(BOX_POS))})


@pytest.fixture(scope="module")
def sim():
    return build_warp_sim(_default_scene(), seed=42)


def test_set_get_actor_states_per_env(sim):
    """A per-env set reaches GPU state and reads back distinct per environment."""
    env_ids = torch.arange(NUM_ENVS, device=sim.sim_device)
    target = torch.zeros(NUM_ENVS, 13, device=sim.sim_device)
    for e in range(NUM_ENVS):
        target[e, :3] = torch.tensor([10.0 + e, 20.0 + e, 1.0 + e], device=sim.sim_device)
    target[:, 6] = 1.0  # identity quat (qw) in xyzw

    sim.set_actor_states(["box"], env_ids, target)

    # The GPU qpos must actually change (a CPU-only write would leave it at BOX_POS).
    box_qpos = sim.object_addrs["box"]["qpos_addr"]
    gpu_pos = sim.backend.qpos_t[:, box_qpos : box_qpos + 3]
    assert not torch.allclose(gpu_pos.cpu(), torch.tensor(BOX_POS).expand(NUM_ENVS, 3), atol=1e-3)

    sim.backend.step()
    got = sim.get_actor_states(["box"], env_ids)
    # Per-env distinct, matching the per-env targets (xy exact; z drifts under gravity).
    # A broadcast-one-value read would make all envs identical.
    assert torch.allclose(got[:, :2], target[:, :2], atol=1e-2)
    assert len({round(float(got[e, 0]), 2) for e in range(NUM_ENVS)}) == NUM_ENVS


def test_angular_velocity_roundtrip_non_identity(sim):
    """set->get round-trips angular velocity at a non-identity orientation (frame match)."""
    env_ids = torch.arange(NUM_ENVS, device=sim.sim_device)
    s = float(np.sin(np.pi / 4))
    state = torch.zeros(NUM_ENVS, 13, device=sim.sim_device)
    state[:, :3] = torch.tensor([0.0, 0.0, 1.0], device=sim.sim_device)
    state[:, 3:7] = torch.tensor([0.0, 0.0, s, s], device=sim.sim_device)  # 90deg about Z, xyzw
    state[:, 10:13] = torch.tensor([0.3, -0.2, 0.5], device=sim.sim_device)

    sim.set_actor_states(["box"], env_ids, state)
    back = sim.get_actor_states(["box"], env_ids)
    assert torch.allclose(back[:, 10:13], state[:, 10:13], atol=1e-4)


def test_initial_object_pose_spread_by_env_origins():
    """Each env's object spawns at its own env_origin world position (not all the same)."""
    # Distinct per-env origins so the spread is observable.
    origins = torch.zeros(NUM_ENVS, 3)
    origins[:, 0] = torch.arange(NUM_ENVS) * 5.0
    origins = origins.to("cuda:0")
    s = build_warp_sim(_default_scene(), env_origins=origins, seed=42)

    env_ids = torch.arange(NUM_ENVS, device=s.sim_device)
    pos = s.get_actor_states(["box"], env_ids)[:, :3]
    expected_x = origins[:, 0] + BOX_POS[0]
    assert torch.allclose(pos[:, 0], expected_x, atol=1e-3), f"box x {pos[:, 0].cpu()} != {expected_x.cpu()}"


def _z_all(s, name):
    """z of ``name`` in every env, shape [NUM_ENVS]."""
    return s.get_actor_states([name], torch.arange(NUM_ENVS, device=s.sim_device))[:, 2]


def test_static_body_holds_pose_multi_env():
    """A static (fixed=True) 1->1 object holds its pose across all envs on the Warp backend
    (static bodies have no qpos slice — read from xpos — a path the free-body tests miss)."""
    scene = SceneConfig(
        rigid_objects={"pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.6, 0.0, 0.3], fixed=True)}
    )
    s = build_warp_sim(scene, seed=42)

    assert s.object_registry.get_names_by_type(ObjectType.SCENE) == ["pillar"]
    assert "pillar" not in s.object_addrs  # static: no freejoint qpos slice
    z0 = _z_all(s, "pillar").clone()
    assert torch.allclose(z0, torch.full((NUM_ENVS,), 0.3, device=s.sim_device), atol=1e-3)
    for _ in range(40):
        s.backend.step()
    assert torch.allclose(_z_all(s, "pillar"), z0, atol=1e-3)  # held in every env


def test_static_body_collides_per_env():
    """A static (fixed=True) body must BLOCK dynamics in EVERY env, not just env 0.

    Regression guard: a body welded to the worldbody is "world-static" in mujoco_warp, whose
    per-step kinematics SKIP recomputing such a geom's collision pose (geom_xpos) — it's baked
    once at make_data from the compiled (env-0) pose. So placing the static per-env via body_pos
    alone left its COLLISION geom frozen at env 0 in every other env: the obstacle blocked the
    box only in env 0 while its reported pose looked correct everywhere (a silent, env-dependent
    wrong-physics trap). The fix syncs per-env geom_xpos/geom_xmat; this test drops a free box
    onto a static pillar and asserts it rests ON the pillar (not on the floor) in all envs.

    test_static_body_holds_pose_multi_env only checks the static HOLDS its pose (no dynamics
    touch it), so it passed throughout the bug — this is the missing collision assertion.
    """
    box_half, pillar_z = 0.05, 0.30
    rest_z = pillar_z + 2 * box_half  # box center resting on the pillar top
    scene = SceneConfig(
        rigid_objects={
            "freebox": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.6, 0.0, rest_z + 0.05]),
            "pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.6, 0.0, pillar_z], fixed=True),
        }
    )
    # Spread envs so the per-env static-geom sync is actually exercised (env 0 would pass even
    # with the bug; envs 1..N are where a world-frozen geom shows up).
    origins = torch.zeros(NUM_ENVS, 3, device="cuda:0")
    origins[:, 0] = torch.arange(NUM_ENVS, device="cuda:0", dtype=torch.float32) * 8.0
    s = build_warp_sim(scene, env_origins=origins, seed=42)

    for _ in range(200):
        s.backend.step()
    z = _z_all(s, "freebox")
    assert torch.allclose(z, torch.full((NUM_ENVS,), rest_z, device=s.sim_device), atol=3e-2), (
        f"box did not rest on the static pillar in every env (z={z.tolist()}, expected ~{rest_z}); "
        f"a per-env value near {box_half} means the box fell through the pillar (collision frozen "
        f"at env 0)."
    )


def test_initial_velocity_takes_effect_multi_env():
    """A free object configured with non-zero lin_vel + ang_vel starts with that velocity in
    EVERY env on the Warp GPU backend, and integrates it: it translates +x/+y (gravity-free)
    and its orientation changes. Read-back alone could pass on a silent ignore, so the
    post-step motion check is included. Identity initial orientation => config (world) velocity
    matches the body-local qvel read-back directly."""
    lin, ang = [1.0, 0.5, 0.0], [0.0, 0.0, 3.0]
    scene = SceneConfig(
        rigid_objects={
            "vbox": RigidObjectConfig(
                urdf_file=BOX_URDF, position=[0.0, 0.0, 0.6], linear_velocity=lin, angular_velocity=ang
            )
        }
    )
    s = build_warp_sim(scene, seed=42)
    env_ids = torch.arange(NUM_ENVS, device=s.sim_device)

    # Immediate read-back: the configured velocity is live on the GPU in every env.
    st0 = s.get_actor_states(["vbox"], env_ids)  # [NUM_ENVS, 13]
    assert torch.allclose(st0[:, 7:10], torch.tensor(lin, device=s.sim_device).expand(NUM_ENVS, 3), atol=1e-4)
    assert torch.allclose(st0[:, 10:13], torch.tensor(ang, device=s.sim_device).expand(NUM_ENVS, 3), atol=1e-4)

    p0, q0 = st0[:, :3].clone(), st0[:, 3:7].clone()
    for _ in range(40):
        s.backend.step()

    st1 = s.get_actor_states(["vbox"], env_ids)
    # Linear: every env moved +x and +y.
    assert torch.all(st1[:, 0] - p0[:, 0] > 1e-3), f"vbox dx {(st1[:, 0] - p0[:, 0]).cpu()}"
    assert torch.all(st1[:, 1] - p0[:, 1] > 1e-3), f"vbox dy {(st1[:, 1] - p0[:, 1]).cpu()}"
    # Angular: orientation changed in every env.
    assert torch.all((st1[:, 3:7] - q0).abs().amax(dim=1) > 1e-2), "vbox orientation unchanged under ang_vel"


def test_multibody_scene_file_multi_env():
    """A 1->N scene FILE expands to free + static bodies on the Warp backend (multi-env):
    free_box falls in every env, static_post holds, with the authored +0.5m-x offset."""
    scene = SceneConfig(scene_files={"scene": SceneFileConfig(xml_path=MULTIBODY_XML, position=[0.4, 0.0, 0.6])})
    s = build_warp_sim(scene, seed=42)

    assert s.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["scene_free_box"]
    assert s.object_registry.get_names_by_type(ObjectType.SCENE) == ["scene_static_post"]
    env_ids = torch.arange(NUM_ENVS, device=s.sim_device)
    free_x = s.get_actor_states(["scene_free_box"], env_ids)[:, 0]
    static_x = s.get_actor_states(["scene_static_post"], env_ids)[:, 0]
    assert torch.allclose(static_x - free_x, torch.full((NUM_ENVS,), 0.5, device=s.sim_device), atol=1e-2)

    z_free0, z_static0 = _z_all(s, "scene_free_box").clone(), _z_all(s, "scene_static_post").clone()
    for _ in range(40):
        s.backend.step()
    assert torch.all(_z_all(s, "scene_free_box") < z_free0 - 1e-3)  # fell in every env
    assert torch.allclose(_z_all(s, "scene_static_post"), z_static0, atol=1e-3)  # held in every env
