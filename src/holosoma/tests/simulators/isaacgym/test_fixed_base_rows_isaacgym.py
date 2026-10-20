"""Unit tests for the IsaacGym fixed-base spawn pose and root-write rule (CPU, no GPU, no IsaacGym SDK).

``robot_start_pose`` is the pure-torch half of ``IsaacGym._build_each_env``: a fixed base spawns at env origin +
``base_init_state`` position with the configured xyzw orientation and no XY jitter (every same-pose reset depends
on it); a floating base keeps the legacy +-1 m XY jitter, the init height and the SDK's identity orientation.
``fixed_base_robot_rows`` is the pure-torch half of ``IsaacGym._set_actor_root_state_tensor_by_index``: robot
rows (``actor_indices`` in ``robot_indices``, env = index // actors per env) whose pose equals the registered spawn
pose are rewritten as spawn pose + zero velocity, a robot row that changes the pose raises ``NotImplementedError``
before anything is written (a mixed robot+object batch is all-or-nothing), object rows pass through.
``fixed_base_robot_assert.py`` covers the live spawn and writer on the GPU lane.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.isaacgym.root_state import fixed_base_robot_rows, robot_start_pose
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 3
ACTORS_PER_ENV = 2  # robot is slot 0 of each env block, one object in slot 1
ROBOT_INDICES = torch.arange(NUM_ENVS) * ACTORS_PER_ENV
# A 90 deg yaw: q and -q spell the same rotation, so the sign-free compare is exercised.
YAW90_XYZW = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678])
SPAWN = torch.zeros(NUM_ENVS, 7)
SPAWN[:, 0] = torch.arange(NUM_ENVS, dtype=torch.float)  # distinct per-env origins
SPAWN[:, 2] = 0.5
SPAWN[:, 3:7] = YAW90_XYZW
ENV_ORIGINS = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.5], [5.0, 6.0, 1.0]])
BASE_INIT_STATE = torch.cat([torch.tensor([0.1, 0.2, 0.8]), YAW90_XYZW, torch.zeros(6)])


def _start_pose(env_id: int, *, fixed_base: bool) -> tuple[torch.Tensor, list[float] | None]:
    return robot_start_pose(
        env_id, env_origins=ENV_ORIGINS, base_init_state=BASE_INIT_STATE, fixed_base=fixed_base, device="cpu"
    )


def test_fixed_base_spawns_at_origin_plus_init_pose_without_jitter() -> None:
    torch.manual_seed(0)
    for env_id in range(NUM_ENVS):
        pos, quat = _start_pose(env_id, fixed_base=True)
        assert torch.equal(pos, ENV_ORIGINS[env_id] + BASE_INIT_STATE[:3])
        assert quat == YAW90_XYZW.tolist()  # xyzw, the gymapi.Quat argument order


def test_floating_base_spawn_jitters_xy_and_keeps_init_height() -> None:
    torch.manual_seed(0)
    origins_before = ENV_ORIGINS.clone()
    offsets = []
    for env_id in range(NUM_ENVS):
        pos, quat = _start_pose(env_id, fixed_base=False)
        assert quat is None  # the SDK default (identity) orientation
        assert pos[2] == BASE_INIT_STATE[2]
        offsets.append(pos[:2] - ENV_ORIGINS[env_id, :2])
    offset = torch.stack(offsets)
    assert (offset.abs() <= 1.0).all() and offset.abs().sum() > 0
    assert torch.equal(ENV_ORIGINS, origins_before)  # the jitter is applied to a copy


class _SpawnLookup:
    """Registered spawn poses; records the env ids looked up."""

    def __init__(self) -> None:
        self.lookups: list[list[int]] = []

    def __call__(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.lookups.append(env_ids.tolist())
        return SPAWN[env_ids]


def _rows(
    actor_indices: torch.Tensor, states: torch.Tensor, lookup: _SpawnLookup, actors_per_env: int = ACTORS_PER_ENV
):
    return fixed_base_robot_rows(
        actor_indices,
        states,
        robot_indices=torch.arange(NUM_ENVS) * actors_per_env,
        actors_per_env=actors_per_env,
        spawn_poses_for=lookup,
    )


def _row(pose: torch.Tensor, velocity: float = 0.0) -> torch.Tensor:
    return torch.cat([pose, torch.full((6,), velocity)])


def test_object_rows_pass_through_untouched() -> None:
    lookup = _SpawnLookup()
    states = torch.randn(2, 13, generator=torch.Generator().manual_seed(0))

    out = _rows(ROBOT_INDICES[:2] + 1, states, lookup)

    assert out is states
    assert lookup.lookups == []


def test_same_pose_robot_rows_are_rewritten_as_spawn_plus_zero_velocity() -> None:
    lookup = _SpawnLookup()
    moved_object = _row(torch.tensor([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]), velocity=2.0)
    negated = SPAWN[2].clone()
    negated[3:7] = -negated[3:7]
    states = torch.stack([_row(SPAWN[0], velocity=1.0), moved_object, _row(negated, velocity=-1.0)])
    before = states.clone()

    out = _rows(torch.tensor([0, 3, 4]), states, lookup)  # robot env 0, object env 1, robot env 2

    assert out is not states and torch.equal(states, before)  # the input buffer is never mutated
    assert torch.equal(out[[0, 2], :7], SPAWN[[0, 2]])  # -q comes back as the registered sign
    assert torch.equal(out[[0, 2], 7:], torch.zeros(2, 6))
    assert torch.equal(out[1], moved_object)
    assert lookup.lookups == [[0, 2]]


def test_pose_change_raises_before_any_write() -> None:
    target = SPAWN[1].clone()
    target[2] += 0.5
    states = _row(target).unsqueeze(0)
    before = states.clone()

    with pytest.raises(NotImplementedError, match="IsaacGym cannot move a fixed-base robot"):
        _rows(ROBOT_INDICES[1:2], states, _SpawnLookup())

    assert torch.equal(states, before)


def test_mixed_batch_with_one_moved_robot_row_is_all_or_nothing() -> None:
    moved_robot = SPAWN[2].clone()
    moved_robot[0] += 1e-3  # above ROOT_POSE_ATOL
    states = torch.stack([_row(SPAWN[0]), _row(torch.ones(7)), _row(moved_robot)])
    before = states.clone()

    with pytest.raises(NotImplementedError, match="write the spawn pose or use MuJoCo/IsaacSim"):
        _rows(torch.tensor([0, 1, 4]), states, _SpawnLookup())

    assert torch.equal(states, before)  # the same-pose robot row and the object row were not written either


def test_env_id_is_actor_index_over_actors_per_env() -> None:
    lookup = _SpawnLookup()
    states = _row(SPAWN[2], velocity=0.3).unsqueeze(0)

    out = _rows(torch.tensor([6]), states, lookup, actors_per_env=3)  # robot indices 0, 3, 6

    assert lookup.lookups == [[2]]
    assert torch.equal(out[0, :7], SPAWN[2]) and torch.equal(out[0, 7:], torch.zeros(6))
    with pytest.raises(NotImplementedError):
        _rows(torch.tensor([3]), states, lookup, actors_per_env=3)  # env 1's spawn differs from env 2's pose
