"""Live MuJoCo (ClassicBackend, CPU) integration test for object observation terms.

Drives the task-independent object-state observation terms through a REAL simulator and the
REAL ObservationManager, proving:
- the terms read live object state via the cached accessor and return base-frame state;
- the ObservationManager measures their variable (k*N) dimension at init with NO declared
  dim and NO validator (handoff Part 3) and concatenates them into a group;
- a robot-only scene yields empty object terms (dim 0) without error.

Runs in the MuJoCo (hsmujoco) env; mirrors the builder in test_scene_spawn_mujoco.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg  # noqa: E402
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.managers.observation.manager import ObservationManager  # noqa: E402
from holosoma.simulator.shared.object_registry import ObjectType  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_TERMS = "holosoma.managers.observation.terms.objects"


class _TaskShell:
    """Minimal env exposing what the object obs terms and ObservationManager need."""

    def __init__(self, sim):
        self.simulator = sim
        self.device = "cpu"
        self.num_envs = sim.num_envs


def _obs_cfg():
    return ObservationManagerCfg(
        groups={
            "object_obs": ObsGroupCfg(
                concatenate=True,
                enable_noise=False,
                history_length=1,
                terms={
                    "object_pos_b": ObsTermCfg(func=f"{_TERMS}:object_pos_b"),
                    "object_quat_b": ObsTermCfg(func=f"{_TERMS}:object_quat_b"),
                    "object_lin_vel_b": ObsTermCfg(func=f"{_TERMS}:object_lin_vel_b"),
                    "object_ang_vel_b": ObsTermCfg(func=f"{_TERMS}:object_ang_vel_b"),
                },
            )
        }
    )


def test_object_obs_dim_measured_for_two_objects():
    """Two free bodies -> manager measures dim = 3+4+3+3 per object = 13*2 = 26, no validator."""
    scene = SceneConfig(
        rigid_objects={
            "free0": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
            "free1": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
        }
    )
    sim = build_classic_sim(scene)
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["free0", "free1"]

    mgr = ObservationManager(_obs_cfg(), _TaskShell(sim), "cpu")
    dims = mgr.get_obs_dims()
    # pos(3) + quat(4) + lin(3) + ang(3) = 13 per object, 2 objects.
    assert dims["object_obs"] == 13 * 2

    obs = mgr.compute()["object_obs"]
    assert obs.shape == (1, 26)
    assert torch.isfinite(obs).all()


def test_object_obs_empty_for_robot_only_scene():
    """Robot-only scene -> each object term is dim 0, group concatenates to 0, no error."""
    sim = build_classic_sim(SceneConfig())
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == []

    mgr = ObservationManager(_obs_cfg(), _TaskShell(sim), "cpu")
    assert mgr.get_obs_dims()["object_obs"] == 0
    assert mgr.compute()["object_obs"].shape == (1, 0)


# ----- base-frame transform correctness -----
#
# These set a KNOWN robot pose/yaw and KNOWN object state on the real sim, then call the
# observation terms directly and assert the base-frame result against the hand-computed
# world->base transform. The set/get actor-state path holds the written value without a physics
# step (same one-shot round-trip the GPU set/get tests rely on).

_ENV0 = None  # filled per test


def _set_robot_pose(sim, pos, quat_xyzw):
    """Write the robot root pose (xyzw quat) on the live sim and flush to the backend."""
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    sim.robot_root_states[:, :3] = torch.tensor(pos, device=sim.sim_device, dtype=torch.float32)
    sim.robot_root_states[:, 3:7] = torch.tensor(quat_xyzw, device=sim.sim_device, dtype=torch.float32)
    sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states[env_ids])


def _set_object_state(sim, name, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0), lin=(0, 0, 0), ang=(0, 0, 0)):
    """Write a free object's full 13-state (xyzw) on the live sim."""
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    st = torch.zeros(sim.num_envs, 13, device=sim.sim_device)
    st[:, :3] = torch.tensor(pos, device=sim.sim_device, dtype=torch.float32)
    st[:, 3:7] = torch.tensor(quat_xyzw, device=sim.sim_device, dtype=torch.float32)
    st[:, 7:10] = torch.tensor(lin, device=sim.sim_device, dtype=torch.float32)
    st[:, 10:13] = torch.tensor(ang, device=sim.sim_device, dtype=torch.float32)
    sim.set_actor_states([name], env_ids, st)


_YAW90_Z = (0.0, 0.0, 0.70710678, 0.70710678)  # +90 deg about Z, xyzw


def _single_box_sim(pos=(1.0, 2.0, 0.5)):
    return build_classic_sim(
        SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=SMALL_BOX, position=list(pos))})
    )


def test_object_pos_b_identity_robot_is_world():
    """Identity robot at origin -> base frame == world frame."""
    from holosoma.managers.observation.terms.objects import object_pos_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    _set_object_state(sim, "box", (1.0, 2.0, 0.5))
    pos_b = object_pos_b(_TaskShell(sim))
    assert torch.allclose(pos_b[0], torch.tensor([1.0, 2.0, 0.5]), atol=1e-4)


def test_object_pos_b_subtracts_robot_root():
    """Translated robot -> object position is relative to the robot root (relative_to_root)."""
    from holosoma.managers.observation.terms.objects import object_pos_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    _set_object_state(sim, "box", (2.0, 0.0, 0.0))
    pos_b = object_pos_b(_TaskShell(sim))
    assert torch.allclose(pos_b[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


def test_object_pos_b_yawed_robot_rotates_world_to_base():
    """Robot yawed +90 about Z, object at world +x -> base-frame -y (world->base rotation)."""
    from holosoma.managers.observation.terms.objects import object_pos_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), _YAW90_Z)
    _set_object_state(sim, "box", (1.0, 0.0, 0.0))
    pos_b = object_pos_b(_TaskShell(sim))
    assert torch.allclose(pos_b[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-4)


def test_object_quat_b_relative_orientation():
    """Robot and object both yawed +90 -> relative orientation is identity (up to sign)."""
    from holosoma.managers.observation.terms.objects import object_quat_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), _YAW90_Z)
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), quat_xyzw=_YAW90_Z)
    quat_b = object_quat_b(_TaskShell(sim))
    assert torch.allclose(quat_b[0].abs(), torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-4)


def test_object_lin_vel_b_rotated_not_root_relative():
    """Robot yawed +90, object lin-vel +x -> base-frame -y (rotated, NOT root-relative)."""
    from holosoma.managers.observation.terms.objects import object_lin_vel_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), _YAW90_Z)
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), lin=(1.0, 0.0, 0.0))
    lv_b = object_lin_vel_b(_TaskShell(sim))
    assert torch.allclose(lv_b[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-3)


def test_object_ang_vel_b_identity_passthrough():
    """Identity robot, object ang-vel (0,0,2) -> unchanged in base frame."""
    from holosoma.managers.observation.terms.objects import object_ang_vel_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), ang=(0.0, 0.0, 2.0))
    av_b = object_ang_vel_b(_TaskShell(sim))
    assert torch.allclose(av_b[0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-3)


def test_object_ang_vel_b_yawed_robot_rotates_world_to_base():
    """Robot yawed +90 about Z, object ang-vel world +x -> base-frame -y.

    This exercises the robot world->base rotation in object_ang_vel_b at a NON-identity ROBOT
    orientation with an IDENTITY OBJECT orientation, so the object's own frame is world and the
    only transform under test is quat_rotate_inverse(robot_quat, .) — the mirror of the
    object_lin_vel_b yawed-robot case. (world +x, rotated into a +90-yaw base, is -y.)
    """
    from holosoma.managers.observation.terms.objects import object_ang_vel_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), _YAW90_Z)
    # Identity object orientation: the written ang-vel is the object's world-frame ang-vel.
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), ang=(1.0, 0.0, 0.0))
    av_b = object_ang_vel_b(_TaskShell(sim))
    assert torch.allclose(av_b[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-3)


def _raw_freejoint_qvel(sim, name):
    """Read a free body's RAW freejoint qvel (cols [lin(3), ang(3)]) straight off the CPU backend.

    Bypasses get_actor_states entirely, so it observes MuJoCo's native representation — angular
    velocity in the BODY-LOCAL frame. Used to probe ONE side of the world<->local boundary
    conversion (the unified getter/setter do the rotation; a symmetric round-trip through them
    would hide a missing rotation, so the canary must read/write raw on one side)."""
    qpos_addr, qvel_addr = sim._actor_freejoint_addrs(name)
    return torch.tensor(sim.backend.data.qvel[qvel_addr + 3 : qvel_addr + 6].copy(), dtype=torch.float32)


def _write_raw_freejoint_angvel(sim, name, ang_local):
    """Inject a RAW body-local angular velocity into the freejoint qvel, bypassing set_actor_states."""
    qpos_addr, qvel_addr = sim._actor_freejoint_addrs(name)
    sim.backend.data.qvel[qvel_addr + 3 : qvel_addr + 6] = ang_local


def test_object_ang_vel_get_converts_body_local_to_world():
    """GET side of the boundary: a RAW body-local freejoint spin reads back WORLD-frame.

    The unified actor-state contract (base_simulator) is world-frame for angular velocity, but
    MuJoCo's freejoint qvel stores it BODY-LOCAL. So get_actor_states must rotate body-local->world
    at the backend boundary. We bypass the setter and inject a raw body-local spin (2,0,0) about the
    object's own x-axis, with the object yawed +90 about Z (body-x == world +y), then read through
    the unified getter (object_ang_vel_b, identity robot so base == world): it MUST report (0,2,0).

    This is the real canary — it FAILS if the get-side conversion is removed (a symmetric
    set->get round-trip cannot, since it would cancel the missing rotation on both sides).
    CLUSTER-NEEDED (live MuJoCo).
    """
    from holosoma.managers.observation.terms.objects import object_ang_vel_b

    sim = _single_box_sim()
    _set_robot_pose(sim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))  # identity robot: base == world
    # Pose the object yawed +90 about Z (zero vel via the contract), then inject a RAW body-local spin.
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), quat_xyzw=_YAW90_Z, ang=(0.0, 0.0, 0.0))
    _write_raw_freejoint_angvel(sim, "box", [2.0, 0.0, 0.0])  # body-local x-spin

    av_b = object_ang_vel_b(_TaskShell(sim))
    # body-x spin under a +90-Z object orientation is world +y; identity robot => base == world.
    assert torch.allclose(av_b[0], torch.tensor([0.0, 2.0, 0.0]), atol=1e-3)


def test_object_ang_vel_set_converts_world_to_body_local():
    """SET side of the boundary: a WORLD-frame angular velocity lands as the right BODY-LOCAL qvel.

    Inverse of the get-side canary. Set world ang-vel (0,2,0) through the unified setter on an
    object yawed +90 about Z, then read the RAW freejoint qvel (bypassing the getter): it MUST be
    the body-local (2,0,0) (world +y mapped into the +90-yawed body is body +x). FAILS if the
    set-side world->local conversion is removed. CLUSTER-NEEDED (live MuJoCo).
    """
    sim = _single_box_sim()
    _set_object_state(sim, "box", (1.0, 0.0, 0.0), quat_xyzw=_YAW90_Z, ang=(0.0, 2.0, 0.0))  # world ang
    raw_local = _raw_freejoint_qvel(sim, "box")
    assert torch.allclose(raw_local, torch.tensor([2.0, 0.0, 0.0]), atol=1e-3)


def test_object_pos_b_multi_object_not_scrambled():
    """Two objects keep their own world pose through the k*N reshape (object-major order)."""
    from holosoma.managers.observation.terms.objects import object_pos_b

    sim = build_classic_sim(
        SceneConfig(
            rigid_objects={
                "free0": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
                "free1": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
            }
        )
    )
    _set_robot_pose(sim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    _set_object_state(sim, "free0", (1.0, 0.0, 0.0))
    _set_object_state(sim, "free1", (0.0, 2.0, 0.0))
    pos_b = object_pos_b(_TaskShell(sim)).view(1, 2, 3)
    assert torch.allclose(pos_b[0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(pos_b[0, 1], torch.tensor([0.0, 2.0, 0.0]), atol=1e-4)


def test_reset_objects_callback_noop_robot_only_scene():
    """BaseTask._reset_objects_callback no-ops cleanly on a robot-only scene (no free bodies)."""
    from holosoma.envs.base_task.base_task import BaseTask

    sim = build_classic_sim(SceneConfig())
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == []

    task = BaseTask.__new__(BaseTask)
    task.simulator = sim
    task.device = "cpu"
    task.num_envs = sim.num_envs

    calls = []
    orig = sim.set_actor_states
    sim.set_actor_states = lambda *a, **_k: calls.append(a)  # type: ignore[method-assign]
    try:
        task._reset_objects_callback(torch.arange(sim.num_envs, device=sim.sim_device))
    finally:
        sim.set_actor_states = orig  # type: ignore[method-assign]
    assert calls == []  # robot-only scene -> no object writes
