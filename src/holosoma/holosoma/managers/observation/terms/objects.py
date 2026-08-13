"""Object-state observation terms.

These terms expose the state of registered free bodies (``ObjectType.INDIVIDUAL``) to a
policy in the **robot base frame**: translation- and yaw-aware relative to the robot.

Each term returns ``[num_envs, k * N]`` for ``N`` objects (``k`` = 3 for positions/
velocities, 4 for orientation). The dimension is derived from the number of objects.

Quaternions are xyzw throughout, matching the actor-state API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.rotations import quat_conjugate, quat_mul, quat_rotate_inverse_batched

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


def _resolve_names(env: BaseTask, object_names: Sequence[str] | None) -> list[str]:
    """Free-body names to observe — explicit list, else every registered free body."""
    if object_names is not None:
        return list(object_names)
    return env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)


def _object_states_env_major(env: BaseTask, names: list[str]) -> torch.Tensor:
    """Live object states reshaped to ``[num_envs, N, 13]`` (xyzw, world frame).

    ``get_actor_states`` returns ``[N * num_envs, 13]`` object-major / env-minor; reshape
    to ``[N, num_envs, 13]`` then permute so each env's objects are contiguous.
    """
    env_ids = torch.arange(env.num_envs, device=env.device)
    states = env.simulator.get_actor_states(names, env_ids)  # [N*num_envs, 13]
    return states.view(len(names), env.num_envs, 13).permute(1, 0, 2)


def _object_vec3_b(
    env: BaseTask, object_names: Sequence[str] | None, sl: slice, *, relative_to_root: bool
) -> torch.Tensor:
    """A ``[num_envs, 3 * N]`` per-object 3-vector rotated into the robot base frame.

    Reads the ``sl`` slice of each object's 13-state (positions 0:3, lin-vel 7:10, ang-vel
    10:13), optionally subtracts the robot root position first (``relative_to_root`` for
    positions), then rotates world->base. Empty (``[num_envs, 0]``) for a robot-only scene.
    """
    names = _resolve_names(env, object_names)
    if not names:
        return torch.empty(env.num_envs, 0, device=env.device)

    st = _object_states_env_major(env, names)  # [num_envs, N, 13]
    robot_root = env.simulator.robot_root_states  # [num_envs, 13]
    vec_w = st[..., sl]  # [num_envs, N, 3], world frame
    if relative_to_root:
        vec_w = vec_w - robot_root[:, None, :3]
    vec_b = quat_rotate_inverse_batched(robot_root[:, 3:7], vec_w)  # base frame (xyzw root quat)
    return vec_b.reshape(env.num_envs, -1)


def object_pos_b(env: BaseTask, object_names: Sequence[str] | None = None) -> torch.Tensor:
    """Object positions in the robot base frame, ``[num_envs, 3 * N]`` (empty if no objects)."""
    return _object_vec3_b(env, object_names, slice(0, 3), relative_to_root=True)


def object_lin_vel_b(env: BaseTask, object_names: Sequence[str] | None = None) -> torch.Tensor:
    """Object linear velocity in the robot base frame, ``[num_envs, 3 * N]`` (empty if no objects)."""
    return _object_vec3_b(env, object_names, slice(7, 10), relative_to_root=False)


def object_ang_vel_b(env: BaseTask, object_names: Sequence[str] | None = None) -> torch.Tensor:
    """Object angular velocity in the robot base frame, ``[num_envs, 3 * N]`` (empty if no objects).

    Relies on ``get_actor_states`` returning WORLD-frame angular velocity (the unified
    actor-state contract); each backend converts from its native storage at its own boundary
    (MuJoCo's freejoint qvel is body-local, so its get/set_actor_states rotate world<->local).
    This term then only applies the robot world->base rotation, like the linear-velocity term.
    """
    return _object_vec3_b(env, object_names, slice(10, 13), relative_to_root=False)


def object_quat_b(env: BaseTask, object_names: Sequence[str] | None = None) -> torch.Tensor:
    """Object orientation relative to the robot base, xyzw, ``[num_envs, 4 * N]``.

    Empty (``[num_envs, 0]``) for a robot-only scene. (Kept separate from the vec3 terms: an
    orientation is composed, ``conj(q_base) * q_obj``, not rotated like a vector.)
    """
    names = _resolve_names(env, object_names)
    if not names:
        return torch.empty(env.num_envs, 0, device=env.device)

    st = _object_states_env_major(env, names)  # [num_envs, N, 13]
    obj_quat_w = st[..., 3:7]  # [num_envs, N, 4]
    base_quat = env.simulator.robot_root_states[:, 3:7]  # [num_envs, 4]

    # q_base->obj = conj(q_base) * q_obj, broadcast the base quat across the N objects.
    base_quat_exp = base_quat[:, None, :].expand(-1, len(names), -1)
    quat_b = quat_mul(quat_conjugate(base_quat_exp, w_last=True), obj_quat_w, w_last=True)
    return quat_b.reshape(env.num_envs, -1)
