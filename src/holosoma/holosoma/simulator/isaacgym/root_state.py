"""Pure-torch pieces of IsaacGym's robot root handling (spawn pose, fixed-base root-write rule)."""

from __future__ import annotations

from collections.abc import Callable

from holosoma.simulator.shared.root_states import root_pose_changed
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


def robot_start_pose(
    env_id: int,
    *,
    env_origins: torch.Tensor,
    base_init_state: torch.Tensor,
    fixed_base: bool,
    device: str,
) -> tuple[torch.Tensor, list[float] | None]:
    """Spawn position and xyzw orientation of env ``env_id``'s robot (``None``: the SDK's identity).

    A welded base cannot be repositioned at reset, so it spawns exactly at the registered pose (env origin +
    configured position and orientation, no XY jitter); a floating base keeps the legacy +-1 m XY jitter at
    the configured height.
    """
    if fixed_base:
        return env_origins[env_id] + base_init_state[:3], base_init_state[3:7].tolist()
    pos = env_origins[env_id].clone()
    pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=device).squeeze(1)
    pos[2] = base_init_state[2]
    return pos, None


def fixed_base_robot_rows(
    actor_indices: torch.Tensor,
    states: torch.Tensor,
    *,
    robot_indices: torch.Tensor,
    actors_per_env: int,
    spawn_poses_for: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply the fixed-base rule to the robot rows of a root write; returns the rows to write.

    Robot rows (``actor_indices`` in ``robot_indices``; the robot is slot 0 of each env block, so
    ``index // actors_per_env`` is the env) must carry the registered spawn pose (``spawn_poses_for(env_ids)``,
    ``[n, 7]`` world xyzw); otherwise raise before any write, so a mixed robot+object write is all-or-nothing.
    Accepted rows are rewritten as spawn pose + zero velocity so the raw buffer stays truthful.
    """
    robot_rows = torch.isin(actor_indices, robot_indices)
    if not bool(robot_rows.any()):
        return states
    spawn = spawn_poses_for(actor_indices[robot_rows] // actors_per_env)
    if root_pose_changed(states[robot_rows, :7], spawn):
        raise NotImplementedError(
            "IsaacGym cannot move a fixed-base robot: PhysX keeps a welded articulation root where it spawned; "
            "write the spawn pose or use MuJoCo/IsaacSim"
        )
    states = states.clone()
    states[robot_rows, :7] = spawn
    states[robot_rows, 7:] = 0.0
    return states
