"""``MotionCommand.reset`` clamps the noised initial DOF pose to soft limits it derives itself (no sim).

``simulator.dof_pos_limits`` is the hard range on every backend; the term applies
``soft_limits(hard, init_pose_cfg.soft_dof_pos_limit)`` (default 0.95) on its first reset and caches the
result. Driven on a ``MotionCommand`` built without ``__init__`` over a one-clip stub motion and a stub
env/simulator; the root and DOF writes into the simulator tensors are read back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.simulator.shared.dof_limits import soft_limits

pytestmark = pytest.mark.no_sim

NUM_ENVS, NUM_DOF, NUM_FRAMES, NUM_BODIES = 3, 4, 10, 2
ENV_IDS = torch.tensor([0, 2])
# Asymmetric hard ranges so the midpoint contraction is pinned, not just a scale about zero.
HARD_LIMITS = torch.tensor([[-1.0, 1.0], [-2.0, 0.0], [0.0, 4.0], [-0.5, 0.5]])
SOFT_95 = torch.tensor([[-0.95, 0.95], [-1.95, -0.05], [0.1, 3.9], [-0.475, 0.475]])
ROOT_QUAT_XYZW = [0.0, 0.0, 0.70710678, 0.70710678]
JOINT_VEL = 0.25


def _term(*, soft_factor: float, dof_pos_noise: float) -> tuple[Any, Any]:
    """A ``MotionCommand`` with the reset-path state set by hand; returns ``(term, simulator)``."""
    term: Any = MotionCommand.__new__(MotionCommand)
    term.motion_cfg = SimpleNamespace(use_adaptive_timesteps_sampler=False, start_at_timestep_zero_prob=0.0)
    # Root noise stays zero so the root write can be checked exactly; only the DOF noise is exercised.
    term.init_pose_cfg = NoiseToInitialPoseConfig(
        overall_noise_scale=1.0, dof_pos=dof_pos_noise, soft_dof_pos_limit=soft_factor
    )
    term._soft_dof_pos_limits = None
    term.root_body_index = 0
    term.num_envs, term.device = NUM_ENVS, "cpu"
    term.time_steps = torch.zeros(NUM_ENVS, dtype=torch.long)
    term.motion_ids = torch.zeros(NUM_ENVS, dtype=torch.long)
    body_pos_w = torch.zeros(NUM_FRAMES, NUM_BODIES, 3)
    body_pos_w[:, 0, 2] = torch.linspace(0.7, 0.8, NUM_FRAMES)  # root height varies per frame
    body_quat_w = torch.zeros(NUM_FRAMES, NUM_BODIES, 4)
    body_quat_w[..., :] = torch.tensor(ROOT_QUAT_XYZW)
    term.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([NUM_FRAMES]),
        has_object=False,
        joint_pos=torch.zeros(NUM_FRAMES, NUM_DOF),  # the clip sits at the midpoint of every range
        joint_vel=torch.full((NUM_FRAMES, NUM_DOF), JOINT_VEL),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=torch.zeros(NUM_FRAMES, NUM_BODIES, 3),
        body_ang_vel_w=torch.zeros(NUM_FRAMES, NUM_BODIES, 3),
    )
    simulator = SimpleNamespace(
        dof_pos_limits=HARD_LIMITS.clone(),
        dof_pos=torch.full((NUM_ENVS, NUM_DOF), 7.0),
        dof_vel=torch.full((NUM_ENVS, NUM_DOF), 7.0),
        robot_root_states=torch.zeros(NUM_ENVS, 13),
        scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])),
    )
    term._env = SimpleNamespace(simulator=simulator, is_evaluating=False)
    return term, simulator


def test_dof_targets_are_clipped_to_the_soft_range_and_cached() -> None:
    torch.manual_seed(0)
    term, simulator = _term(soft_factor=0.95, dof_pos_noise=10.0)  # noise far beyond every hard range

    term.reset(ENV_IDS)

    assert torch.allclose(term._soft_dof_pos_limits, SOFT_95)
    assert torch.equal(term._soft_dof_pos_limits, soft_limits(HARD_LIMITS, 0.95))
    written = simulator.dof_pos[ENV_IDS]
    lower, upper = SOFT_95[:, 0], SOFT_95[:, 1]
    assert bool((written >= lower - 1e-6).all()) and bool((written <= upper + 1e-6).all())
    at_a_bound = torch.isclose(written, lower.expand_as(written)) | torch.isclose(written, upper.expand_as(written))
    assert bool(at_a_bound.any()), "10-wide noise must hit the soft bounds"
    assert torch.equal(simulator.dof_pos[1], torch.full((NUM_DOF,), 7.0))  # untouched env
    # The DOF velocity comes from the clip unnoised; the root pose is the clip's root at the env origin.
    assert torch.equal(simulator.dof_vel[ENV_IDS], torch.full((len(ENV_IDS), NUM_DOF), JOINT_VEL))
    expected_root_pos = term.motion.body_pos_w[term.time_steps[ENV_IDS], 0] + simulator.scene.env_origins[ENV_IDS]
    assert torch.allclose(simulator.robot_root_states[ENV_IDS, :3], expected_root_pos)
    assert torch.allclose(simulator.robot_root_states[ENV_IDS, 3:7], torch.tensor([ROOT_QUAT_XYZW] * len(ENV_IDS)))
    assert torch.equal(simulator.robot_root_states[ENV_IDS, 7:], torch.zeros(len(ENV_IDS), 6))

    # Cached: the soft range is built once; a later change to the simulator's limits is not re-read.
    cached = term._soft_dof_pos_limits
    simulator.dof_pos_limits = HARD_LIMITS * 100.0
    term.reset(ENV_IDS)
    assert term._soft_dof_pos_limits is cached
    written = simulator.dof_pos[ENV_IDS]
    assert bool((written >= lower - 1e-6).all()) and bool((written <= upper + 1e-6).all())


def test_soft_factor_one_is_the_hard_range() -> None:
    torch.manual_seed(0)
    term, simulator = _term(soft_factor=1.0, dof_pos_noise=10.0)

    term.reset(ENV_IDS)

    assert torch.equal(term._soft_dof_pos_limits, HARD_LIMITS)
    written = simulator.dof_pos[ENV_IDS]
    assert bool((written >= HARD_LIMITS[:, 0]).all()) and bool((written <= HARD_LIMITS[:, 1]).all())
    assert bool((written.abs() > 0.95).any())  # values the 0.95 range would have clipped survive


def test_zero_noise_writes_the_clip_pose_unclipped() -> None:
    term, simulator = _term(soft_factor=0.95, dof_pos_noise=0.0)
    term.motion.joint_pos[:] = torch.tensor([0.5, -1.5, 3.0, 0.2])  # inside the soft range

    term.reset(None)  # None -> every env

    assert torch.allclose(simulator.dof_pos, torch.tensor([[0.5, -1.5, 3.0, 0.2]] * NUM_ENVS))


@pytest.mark.parametrize("env_ids", [torch.tensor([], dtype=torch.long), None])
def test_reset_of_no_or_all_envs_touches_only_those_rows(env_ids: torch.Tensor | None) -> None:
    term, simulator = _term(soft_factor=0.95, dof_pos_noise=0.0)

    term.reset(env_ids)

    touched = simulator.dof_pos != 7.0
    assert bool(touched.all()) if env_ids is None else not bool(touched.any())
