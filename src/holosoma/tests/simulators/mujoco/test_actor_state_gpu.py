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


from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
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


def _z_all(s, name):
    """z of ``name`` in every env, shape [NUM_ENVS]."""
    return s.get_actor_states([name], torch.arange(NUM_ENVS, device=s.sim_device))[:, 2]


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
