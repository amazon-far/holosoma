"""GPU (WarpBackend, multi-env) tests for the object-reset callback and pose-jitter term.

These exercise the live multi-object set_actor_states path end-to-end against the real
WarpBackend — the handoff's Part 1 (subset-env reset) and Part 2c (pose jitter) — which the
CPU/fake-sim unit tests cannot reach. Skipped without CUDA / the MuJoCo-Warp stack.

Covered:
- _reset_objects_callback restores free bodies to their initial pose + configured initial
  velocity per env for a SUBSET of env_ids, leaves non-reset envs alone, never touches static.
- a free body configured with a nonzero initial velocity is reset back to THAT velocity
  (not zero) by both the baseline reset and the jitter term.
- jitter_object_pose_on_reset writes within range, per-env distinct, on free bodies only.
- multi-object set/get round-trips without scrambling objects across envs (the name-major
  reshape contract; the analogue of the IsaacGym reshape fix on the Warp backend).
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
from holosoma.envs.base_task.base_task import BaseTask  # noqa: E402
from holosoma.managers.randomization.terms.objects import jitter_object_pose_on_reset  # noqa: E402
from holosoma.simulator.shared.object_registry import ObjectType  # noqa: E402
from tests.simulators._dr_matrix import _sampler  # noqa: E402
from tests.simulators.mujoco._build import build_warp_sim, env_shell  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
NUM_ENVS = 4


def _free_and_static_scene():
    return SceneConfig(
        rigid_objects={
            "free0": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
            "free1": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
            "pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.7, 0.0, 0.3], fixed=True),
        }
    )


def _task_for(sim) -> BaseTask:
    """A BaseTask shell wired to a live sim (bypasses heavy __init__)."""
    task = BaseTask.__new__(BaseTask)
    task.simulator = sim
    task.device = sim.sim_device
    task.num_envs = NUM_ENVS
    return task


@pytest.fixture(scope="module")
def sim():
    return build_warp_sim(_free_and_static_scene(), seed=42)


def test_reset_objects_callback_subset_envs(sim):
    """Reset a SUBSET of envs: free bodies return to initial pose (zero vel) in those envs;
    non-reset envs keep their drifted state; static body untouched everywhere."""
    free = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    assert free == ["free0", "free1"]
    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["pillar"]

    all_envs = torch.arange(NUM_ENVS, device=sim.sim_device)
    init = {n: sim.get_actor_states([n], all_envs)[:, 2].clone() for n in free}
    pillar0 = sim.get_actor_states(["pillar"], all_envs)[:, 2].clone()

    # Let the free bodies fall so a reset is observable.
    for _ in range(40):
        sim.backend.step()
    drifted = {n: sim.get_actor_states([n], all_envs)[:, 2].clone() for n in free}
    for n in free:
        assert torch.all(drifted[n] < init[n] - 1e-3), f"{n} did not fall before reset"

    # Reset only envs {0, 2}.
    reset_envs = torch.tensor([0, 2], device=sim.sim_device)
    task = _task_for(sim)
    task._reset_objects_callback(reset_envs)

    after = {n: sim.get_actor_states([n], all_envs) for n in free}
    for n in free:
        z = after[n][:, 2]
        # Reset envs back at (near) the initial height; velocity restored to the configured
        # initial value (zero for this scene's boxes — see the velocity test for a nonzero case).
        assert torch.allclose(z[reset_envs], init[n][reset_envs], atol=1e-2), f"{n} not reset in {reset_envs.tolist()}"
        assert torch.allclose(after[n][reset_envs][:, 7:], torch.zeros(2, 6, device=sim.sim_device), atol=1e-3)
        # Non-reset envs {1,3} keep their drifted (fallen) state.
        keep = torch.tensor([1, 3], device=sim.sim_device)
        assert torch.allclose(z[keep], drifted[n][keep], atol=1e-2), f"{n} wrongly changed in non-reset envs"

    # Static body unchanged in every env.
    assert torch.allclose(sim.get_actor_states(["pillar"], all_envs)[:, 2], pillar0, atol=1e-3)


def test_jitter_reset_term_multi_env(sim):
    """jitter_object_pose_on_reset jitters free bodies within range, per-env distinct, leaves
    Z/velocity/quat-norm intact, applies a pure yaw, and never touches the static body."""
    all_envs = torch.arange(NUM_ENVS, device=sim.sim_device)
    # Start from a clean baseline reset so we jitter around the initial pose.
    _task_for(sim)._reset_objects_callback(all_envs)
    base = {n: sim.get_actor_states([n], all_envs).clone() for n in ("free0", "free1")}
    pillar_xy = sim.get_actor_states(["pillar"], all_envs)[:, :2].clone()

    env = env_shell(sim, NUM_ENVS)
    xy_range = 0.15
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=xy_range, yaw_range=0.5)

    for n in ("free0", "free1"):
        st = sim.get_actor_states([n], all_envs)
        xy = st[:, :2]
        dxy = (xy - base[n][:, :2]).abs()
        assert dxy.max() <= xy_range + 1e-3, f"{n} jitter exceeded range"
        assert dxy.max() > 1e-4, f"{n} was not jittered at all"
        # Per-env distinct in x.
        assert len({round(float(xy[e, 0]), 4) for e in range(NUM_ENVS)}) > 1
        # Z untouched (jitter is XY + yaw only).
        assert torch.allclose(st[:, 2], base[n][:, 2], atol=1e-4), f"{n} Z changed by jitter"
        # Velocity unchanged from the baseline (this scene's boxes start at rest, so zero here;
        # the term preserves the configured initial velocity rather than forcing zero).
        assert torch.allclose(st[:, 7:], base[n][:, 7:], atol=1e-3)
        # Quaternion stays unit-norm and is a pure-yaw (z-axis) rotation: x,y components ~0
        # (preset orientation is identity, so the composed result is a bare z-rotation, xyzw).
        q = st[:, 3:7]
        assert torch.allclose(q.norm(dim=-1), torch.ones(NUM_ENVS, device=sim.sim_device), atol=1e-3)
        assert q[:, 0].abs().max() < 1e-3 and q[:, 1].abs().max() < 1e-3, f"{n} not a pure-yaw rotation"

    # Static body never moved.
    assert torch.allclose(sim.get_actor_states(["pillar"], all_envs)[:, :2], pillar_xy, atol=1e-4)


def test_jitter_noop_disabled_zero_range_and_object_subset(sim):
    """jitter is a no-op when disabled or both ranges are zero; object_names narrows targeting."""
    all_envs = torch.arange(NUM_ENVS, device=sim.sim_device)
    _task_for(sim)._reset_objects_callback(all_envs)
    env = env_shell(sim, NUM_ENVS)

    def snap():
        return {n: sim.get_actor_states([n], all_envs).clone() for n in ("free0", "free1")}

    # enabled=False -> nothing moves.
    before = snap()
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.2, yaw_range=0.5, enabled=False)
    for n in ("free0", "free1"):
        assert torch.allclose(sim.get_actor_states([n], all_envs), before[n], atol=1e-5)

    # Both ranges zero -> nothing moves (clean reset == omitting the term).
    before = snap()
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.0, yaw_range=0.0)
    for n in ("free0", "free1"):
        assert torch.allclose(sim.get_actor_states([n], all_envs), before[n], atol=1e-5)

    # object_names subset -> only free1 jittered, free0 untouched.
    _task_for(sim)._reset_objects_callback(all_envs)
    before = snap()
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.15, object_names=["free1"])
    assert torch.allclose(sim.get_actor_states(["free0"], all_envs), before["free0"], atol=1e-5), (
        "free0 wrongly jittered"
    )
    assert (sim.get_actor_states(["free1"], all_envs)[:, :2] - before["free1"][:, :2]).abs().max() > 1e-4


def test_jitter_robot_only_scene_noop():
    """jitter_object_pose_on_reset no-ops (no raise) on a robot-only scene."""
    s = build_warp_sim(SceneConfig(), seed=42)
    assert s.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == []
    all_envs = torch.arange(NUM_ENVS, device=s.sim_device)
    env = env_shell(s, NUM_ENVS)
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.2, yaw_range=0.5)  # must not raise


def test_jitter_composes_onto_nonidentity_orientation():
    """A free body with a non-identity baseline yaw: the jitter yaw COMPOSES onto it (result
    stays a pure z-rotation) rather than overwriting — and differs from the baseline quat."""
    import math

    c, sn = math.cos(math.pi / 4), math.sin(math.pi / 4)  # 90deg about Z, wxyz config order
    scene = SceneConfig(
        rigid_objects={
            "rbox": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6], orientation=[c, 0.0, 0.0, sn])
        }
    )
    s = build_warp_sim(scene, seed=42)
    all_envs = torch.arange(NUM_ENVS, device=s.sim_device)
    _task_for(s)._reset_objects_callback(all_envs)
    base_q = s.get_actor_states(["rbox"], all_envs)[:, 3:7].clone()  # xyzw

    env = env_shell(s, NUM_ENVS)
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.0, yaw_range=0.3)
    q = s.get_actor_states(["rbox"], all_envs)[:, 3:7]
    # Still a pure z-rotation (x,y ~0) and unit-norm, but rotated away from the baseline.
    assert q[:, 0].abs().max() < 1e-3 and q[:, 1].abs().max() < 1e-3
    assert torch.allclose(q.norm(dim=-1), torch.ones(NUM_ENVS, device=s.sim_device), atol=1e-3)
    assert (q - base_q).abs().max() > 1e-3, "yaw delta was not applied (quat == baseline)"


def test_multi_object_set_get_roundtrip_no_scramble(sim):
    """set_actor_states on TWO free bodies round-trips per (object, env) without crossing
    objects — the name-major contract (analogue of the IsaacGym reshape fix)."""
    all_envs = torch.arange(NUM_ENVS, device=sim.sim_device)
    names = ["free0", "free1"]

    # Distinct, recognizable target per (object, env): object index in x, env index in y.
    target = torch.zeros(len(names) * NUM_ENVS, 13, device=sim.sim_device)
    for oi in range(len(names)):
        for e in range(NUM_ENVS):
            row = oi * NUM_ENVS + e
            target[row, 0] = 1.0 + oi  # x encodes object
            target[row, 1] = 0.1 * e  # y encodes env
            target[row, 2] = 0.6
            target[row, 6] = 1.0  # identity quat (xyzw)

    sim.set_actor_states(names, all_envs, target)
    got = sim.get_actor_states(names, all_envs)  # [2*NUM_ENVS, 13], name-major

    # Each (object, env) row must read back its own encoded x and y, not a neighbor's.
    assert torch.allclose(got[:, 0], target[:, 0], atol=1e-3), "object identity (x) scrambled"
    assert torch.allclose(got[:, 1], target[:, 1], atol=1e-3), "env identity (y) scrambled"


def test_reset_restores_configured_initial_velocity():
    """A free body configured with a nonzero initial velocity must be reset back to THAT
    velocity (not zero), by both _reset_objects_callback and the pose-jitter term."""
    lin, ang = [0.7, -0.3, 0.0], [0.0, 0.0, 1.5]
    s = build_warp_sim(
        SceneConfig(
            rigid_objects={
                "vbox": RigidObjectConfig(
                    urdf_file=SMALL_BOX,
                    position=[0.4, 0.0, 0.6],
                    linear_velocity=lin,
                    angular_velocity=ang,
                )
            }
        ),
        seed=42,
    )
    all_envs = torch.arange(NUM_ENVS, device=s.sim_device)
    want_vel = torch.tensor(lin + ang, device=s.sim_device)

    # Evolve so velocity drifts away from the configured value (gravity + integration).
    for _ in range(40):
        s.backend.step()
    assert not torch.allclose(s.get_actor_states(["vbox"], all_envs)[:, 7:], want_vel.expand(NUM_ENVS, 6), atol=1e-2)

    # Baseline reset must restore the configured initial velocity in every env (not zero it).
    _task_for(s)._reset_objects_callback(all_envs)
    vel_after_reset = s.get_actor_states(["vbox"], all_envs)[:, 7:]
    assert torch.allclose(vel_after_reset, want_vel.expand(NUM_ENVS, 6), atol=1e-3), (
        f"reset did not restore configured velocity: got {vel_after_reset[0].cpu()}, want {want_vel.cpu()}"
    )

    # The pose-jitter term jitters pose but keeps that same configured velocity. Jitter XY
    # only (yaw_range=0) so orientation stays identity and the world-frame configured velocity
    # equals the body-local qvel read-back directly (see test_initial_velocity_takes_effect).
    env = env_shell(s, NUM_ENVS)
    jitter_object_pose_on_reset(env, all_envs, sampler=_sampler(env), xy_range=0.1, yaw_range=0.0)
    vel_after_jitter = s.get_actor_states(["vbox"], all_envs)[:, 7:]
    assert torch.allclose(vel_after_jitter, want_vel.expand(NUM_ENVS, 6), atol=1e-3), (
        f"jitter did not preserve configured velocity: got {vel_after_jitter[0].cpu()}, want {want_vel.cpu()}"
    )
