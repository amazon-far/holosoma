"""Locomotion feet reward terms resolve their bodies through env.robot_names.resolve_body_indices (N-generic)."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.managers.observation.terms.locomotion import gravity_vector
from holosoma.managers.reward.terms.locomotion import (
    feet_phase,
    limits_dof_pos,
    penalty_close_feet_xy,
    penalty_feet_ori,
    penalty_stumble,
)
from holosoma.utils.rotations import quat_rotate_inverse
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
    base_quat: torch.Tensor

    def __init__(self, gen: torch.Generator) -> None:
        self._rigid_body_rot = _unit_quats(NUM_ENVS, NUM_BODIES, gen=gen)
        self._rigid_body_pos = torch.randn(NUM_ENVS, NUM_BODIES, 3, generator=gen)
        self.contact_forces = torch.randn(NUM_ENVS, NUM_BODIES, 3, generator=gen)
        self.dof_pos = 2.0 * torch.randn(NUM_ENVS, NUM_DOF, generator=gen)
        lower = -0.5 - torch.rand(NUM_DOF, generator=gen)
        upper = 0.5 + torch.rand(NUM_DOF, generator=gen)
        self.dof_pos_limits = torch.stack([lower, upper], dim=1)

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unused = torch.zeros(NUM_DOF)
        return self.dof_pos_limits, unused, unused


class _StubEnv:
    def __init__(self, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.simulator = _StubSimulator(gen)
        self.simulator.base_quat = _unit_quats(NUM_ENVS, gen=gen)
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


@pytest.mark.parametrize("feet", [["#feet"], ["#feet4"]], ids=["two-feet", "four-feet"])
def test_penalty_feet_ori_sums_selected_foot_tilts(feet: list[str]) -> None:
    env = _env()
    indices = FEET_BY_SPEC[tuple(feet)]
    # Every unselected body tilts 90 degrees. Only selected feet contribute.
    quarter_turn = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    env.simulator._rigid_body_rot[:] = quarter_turn
    env.simulator._rigid_body_rot[:, indices] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env.simulator._rigid_body_rot[1, indices[0]] = quarter_turn
    env.simulator._rigid_body_rot[2, indices] = quarter_turn
    env.simulator._rigid_body_rot[4, indices[0]] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])

    out = penalty_feet_ori(env, feet=feet)

    assert env.resolve_calls == [tuple(feet)]
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0, float(len(indices)), 0.0, 0.0]))


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


@pytest.mark.parametrize(
    "yaw_quat", [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2**-0.5, 2**-0.5]], ids=["forward-x", "forward-y"]
)
def test_penalty_close_feet_uses_lateral_distance_and_strict_threshold(yaw_quat: list[float]) -> None:
    env = _env()
    env.simulator.base_quat[:] = torch.tensor(yaw_quat)
    env.simulator._rigid_body_pos.zero_()
    # Distances bracket the threshold, including equality and negative separation.
    distance = torch.tensor([0.05, 0.099, 0.1, 0.101, -0.05])
    lateral_axis = 1 if yaw_quat[2] == 0 else 0
    env.simulator._rigid_body_pos[:, 5, lateral_axis] = distance
    env.simulator._rigid_body_pos[:, 5, 1 - lateral_axis] = 2.0
    # Keep the equality case free of roundoff from the longitudinal projection.
    env.simulator._rigid_body_pos[2, 5, 1 - lateral_axis] = 0.0

    out = penalty_close_feet_xy(env, close_feet_threshold=0.1)

    assert env.resolve_calls == [("#feet",)]
    assert out.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]


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


@pytest.mark.parametrize("num_feet", [2, 4])
def test_feet_phase_scores_known_gait_keyframes_and_sums_feet(num_feet: int) -> None:
    env = _env()
    env.phase = torch.tensor([-torch.pi, -torch.pi / 2, 0.0, torch.pi / 2, torch.pi]).unsqueeze(1).repeat(1, num_feet)
    env.feet_heights = torch.full((NUM_ENVS, num_feet), 0.02)

    out = feet_phase(env, swing_height=0.08, tracking_sigma=0.01)

    # The gait heights are 0, 0.04, 0.08, 0.04, 0 at these phases.
    expected = torch.exp(-num_feet * torch.tensor([0.04, 0.04, 0.36, 0.04, 0.04]))
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("feet_shape", [(NUM_ENVS, 0), (NUM_ENVS, 3), (0,)])
def test_feet_phase_shape_mismatch_names_the_feet_height_tag(feet_shape: tuple[int, ...]) -> None:
    env = _env()
    env.feet_heights = torch.zeros(*feet_shape)

    with pytest.raises(ValueError, match=r"'feet_height' body tag.*TerrainTermCfg\.feet_height_bodies"):
        feet_phase(env)


def test_limits_dof_pos_reads_dof_pos_limits() -> None:
    # Derive the reward interval from the simulator's hard position limits.
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
