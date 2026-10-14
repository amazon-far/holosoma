"""Locomotion feet reward terms resolve their bodies through env.robot_names.resolve_body_indices (N-generic)."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.managers.observation.terms.locomotion import base_forward_vector, gravity_vector
from holosoma.managers.reward.terms.locomotion import (
    _expected_foot_height,
    feet_phase,
    limits_dof_pos,
    penalty_close_feet_xy,
    penalty_feet_ori,
    penalty_stumble,
)
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 5
NUM_BODIES = 8
NUM_DOF = 4
FEET_BY_SPEC: dict[tuple[str, ...], list[int]] = {
    ("#feet",): [2, 5],
    ("#feet4",): [1, 2, 5, 6],
    ("front_foot",): [7],
}


def _unit_quats(*shape: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(*shape, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


class _StubSimulator:
    def __init__(self, gen: torch.Generator) -> None:
        self._rigid_body_rot = _unit_quats(NUM_ENVS, NUM_BODIES, gen=gen)
        self._rigid_body_pos = torch.randn(NUM_ENVS, NUM_BODIES, 3, generator=gen)
        self.contact_forces = torch.randn(NUM_ENVS, NUM_BODIES, 3, generator=gen)
        self.dof_pos = 2.0 * torch.randn(NUM_ENVS, NUM_DOF, generator=gen)
        lower = -0.5 - torch.rand(NUM_DOF, generator=gen)
        upper = 0.5 + torch.rand(NUM_DOF, generator=gen)
        self.dof_pos_limits = torch.stack([lower, upper], dim=1)


class _StubEnv:
    def __init__(self, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.simulator = _StubSimulator(gen)
        self.base_quat = _unit_quats(NUM_ENVS, gen=gen)
        self.resolve_calls: list[tuple[str, ...]] = []
        self.feet_heights = torch.rand(NUM_ENVS, 2, generator=gen) * 0.1
        self.phase = torch.rand(NUM_ENVS, 2, generator=gen) * 2 * torch.pi - torch.pi
        self.terrain_manager = SimpleNamespace(get_state=lambda _name: SimpleNamespace(feet_heights=self.feet_heights))
        self.command_manager = SimpleNamespace(get_state=lambda _name: SimpleNamespace(phase=self.phase))

    def resolve_body_indices(self, spec: Sequence[str]) -> torch.Tensor:
        self.resolve_calls.append(tuple(spec))
        return torch.tensor(FEET_BY_SPEC[tuple(spec)], dtype=torch.long)

    def resolve_body_names(self, spec: Sequence[str]) -> list[str]:
        return [f"body_{i}" for i in FEET_BY_SPEC[tuple(spec)]]

    @property
    def robot_names(self) -> _StubEnv:
        return self


def _env() -> Any:
    return _StubEnv()


# Pre-refactor two-foot formulas (env.feet_indices[0] / [1]), kept here as the reference.


def _old_penalty_feet_ori(env: Any, left: int, right: int) -> torch.Tensor:
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(env.simulator._rigid_body_rot[:, left], gravity, w_last=True)
    right_gravity = quat_rotate_inverse(env.simulator._rigid_body_rot[:, right], gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


def _old_penalty_close_feet_xy(env: Any, left: int, right: int, close_feet_threshold: float) -> torch.Tensor:
    left_foot_xy = env.simulator._rigid_body_pos[:, left, :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, right, :2]
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )
    return (feet_distance < close_feet_threshold).float()


def _old_feet_phase(env: Any, swing_height: float, tracking_sigma: float) -> torch.Tensor:
    foot_z_left = env.feet_heights[:, 0]
    foot_z_right = env.feet_heights[:, 1]
    rz_left = _expected_foot_height(env.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(env.phase[:, 1], swing_height)
    total_error = torch.square(foot_z_left - rz_left) + torch.square(foot_z_right - rz_right)
    return torch.exp(-total_error / tracking_sigma)


def test_penalty_feet_ori_two_feet_equals_old_formula() -> None:
    env = _env()

    out = penalty_feet_ori(env)

    assert env.resolve_calls == [("#feet",)]
    assert torch.equal(out, _old_penalty_feet_ori(env, 2, 5))


def test_penalty_feet_ori_sole_axes_selects_the_projected_components() -> None:
    env = _env()
    quats = env.simulator._rigid_body_rot[:, [2, 5]].reshape(-1, 4)
    gravity = gravity_vector(env).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 3)
    foot_gravity = quat_rotate_inverse(quats, gravity, w_last=True).reshape(NUM_ENVS, 2, 3)
    expected = torch.sum(torch.sum(torch.square(foot_gravity[..., [0, 2]]), dim=-1) ** 0.5, dim=1)

    out = penalty_feet_ori(env, sole_axes=(0, 2))

    assert torch.allclose(out, expected, atol=1e-6)
    assert not torch.allclose(out, penalty_feet_ori(env))  # the default (0, 1) projection differs


@pytest.mark.parametrize("sole_axes", [(0, 0), (0,), (0, 3), (1, 2, 0), (0.5, 1.5), ("0", "2"), (True, 2)])
def test_penalty_feet_ori_rejects_invalid_sole_axes(sole_axes: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError, match="sole_axes must be two distinct integer axes"):
        penalty_feet_ori(_env(), sole_axes=sole_axes)


def test_penalty_feet_ori_sums_over_four_feet() -> None:
    env = _env()

    out = penalty_feet_ori(env, feet=["#feet4"])

    assert out.shape == (NUM_ENVS,)
    expected = _old_penalty_feet_ori(env, 1, 2) + _old_penalty_feet_ori(env, 5, 6)
    assert torch.allclose(out, expected, atol=1e-6)


def test_penalty_close_feet_xy_two_feet_equals_old_formula() -> None:
    env = _env()
    threshold = 1.0  # near the median seeded foot distance so both outcomes appear

    out = penalty_close_feet_xy(env, close_feet_threshold=threshold)

    assert env.resolve_calls == [("#feet",)]
    assert torch.equal(out, _old_penalty_close_feet_xy(env, 2, 5, threshold))
    assert out.any() and not out.all()  # the reference comparison constrains both sides of the threshold


@pytest.mark.parametrize("feet", [["#feet4"], ["front_foot"]])
def test_penalty_close_feet_xy_requires_exactly_two_feet(feet: list[str]) -> None:
    env = _env()

    with pytest.raises(ValueError, match=r"left/right pair.*expected exactly 2"):
        penalty_close_feet_xy(env, feet=feet)


def test_penalty_stumble_any_over_resolved_feet() -> None:
    env = _env()
    forces = torch.zeros(NUM_ENVS, NUM_BODIES, 3)
    forces[..., 2] = 1.0  # vertical load everywhere
    forces[0, 5, 0] = 10.0  # env 0: a resolved foot hits a wall
    forces[1, 0, 0] = 10.0  # env 1: a non-foot body hits a wall
    forces[2, 1, :2] = torch.tensor([3.9 * 0.6, 3.9 * 0.8])  # env 2: lateral 3.9x the vertical load: no stumble
    forces[3, 6, :2] = torch.tensor([4.1 * 0.6, 4.1 * 0.8])  # env 3: lateral 4.1x: stumble (the ratio is 4)
    forces[4, 2] = torch.tensor([-4.1, 0.0, -1.0])  # env 4: sign-free (|Fz|), a pulling contact still counts
    env.simulator.contact_forces = forces

    out = penalty_stumble(env, feet=["#feet4"])

    assert env.resolve_calls == [("#feet4",)]
    assert out.shape == (NUM_ENVS,)
    assert out.dtype == torch.bool
    assert out.tolist() == [True, False, False, True, True]


def test_feet_phase_two_feet_equals_old_formula() -> None:
    env = _env()

    out = feet_phase(env, swing_height=0.09, tracking_sigma=0.008)

    assert torch.equal(out, _old_feet_phase(env, 0.09, 0.008))


def test_feet_phase_sums_squared_error_over_feet() -> None:
    env = _env()
    env.feet_heights = torch.rand(NUM_ENVS, 4) * 0.1
    env.phase = torch.rand(NUM_ENVS, 4) * 2 * torch.pi - torch.pi

    out = feet_phase(env, swing_height=0.05, tracking_sigma=0.01)

    expected = torch.exp(
        -torch.sum(torch.square(env.feet_heights - _expected_foot_height(env.phase, 0.05)), dim=1) / 0.01
    )
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("feet_shape", [(NUM_ENVS, 0), (NUM_ENVS, 3), (0,)])
def test_feet_phase_shape_mismatch_names_the_feet_height_tag(feet_shape: tuple[int, ...]) -> None:
    env = _env()
    env.feet_heights = torch.zeros(*feet_shape)

    with pytest.raises(ValueError, match=r"'feet_height' body tag.*TerrainTermCfg\.feet_height_bodies"):
        feet_phase(env)


def test_limits_dof_pos_reads_dof_pos_limits() -> None:
    # The stub carries the hard limits under the simulator's one name, dof_pos_limits (the old alias is checked
    # absent on a live simulator in test_fixed_base_arm_e2e_mujoco).
    env = _env()
    limits = env.simulator.dof_pos_limits
    soft = 0.9

    out = limits_dof_pos(env, soft_dof_pos_limit=soft)

    mid = (limits[:, 0] + limits[:, 1]) / 2
    half = 0.5 * (limits[:, 1] - limits[:, 0]) * soft
    lower, upper = mid - half, mid + half
    expected = torch.sum(
        (lower - env.simulator.dof_pos).clip(min=0.0) + (env.simulator.dof_pos - upper).clip(min=0.0), dim=1
    )
    assert torch.allclose(out, expected)
    assert (out > 0).any(), "test data should put some DOFs outside the soft range"
