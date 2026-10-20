"""Root-state helpers shared by every backend's robot root writer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# Absolute tolerance under which two root poses count as equal. Used by every backend's fixed-base
# rule: a root write whose pose equals the current pose is a no-op.
ROOT_POSE_ATOL = 1e-5


def root_pose_changed(requested: torch.Tensor, current: torch.Tensor, atol: float = ROOT_POSE_ATOL) -> bool:
    """True when any row of two ``[n, 7]`` ``[pos, quat]`` tensors differs by more than ``atol``.

    ``q`` and ``-q`` are the same rotation, so a quaternion row matches when either sign agrees.
    Both tensors must use the same quaternion order. Empty inputs count as unchanged.
    """
    if requested.shape[0] == 0:
        return False
    pos_same = torch.all((requested[:, :3] - current[:, :3]).abs() <= atol, dim=1)
    quat, cur = requested[:, 3:7], current[:, 3:7]
    quat_same = torch.all((quat - cur).abs() <= atol, dim=1) | torch.all((quat + cur).abs() <= atol, dim=1)
    return not bool((pos_same & quat_same).all())


def robot_rows(sim: BaseSimulator, env_ids: torch.Tensor, root_states: Any) -> torch.Tensor:
    """Select the ``[len(env_ids), 13]`` robot rows handed to ``set_actor_root_state_tensor_robots``.

    ``None``, the unified ``all_root_states`` view and ``robot_root_states`` itself select
    ``robot_root_states[env_ids]``; a ``[num_envs, 13]`` tensor is the global robot tensor and is indexed by
    ``env_ids``; otherwise a ``[len(env_ids), 13]`` tensor is taken as pre-sliced; any other length raises
    ``ValueError``. Same precedence as ``set_dof_state_tensor_robots`` (global first), so a full-length tensor
    is never mistaken for pre-sliced rows.
    """
    if root_states is None or root_states is sim.all_root_states or root_states is sim.robot_root_states:
        return sim.robot_root_states[env_ids]
    if not isinstance(root_states, torch.Tensor):
        raise TypeError(
            "root_states must be a torch.Tensor, all_root_states or robot_root_states, "
            f"got {type(root_states).__name__}"
        )
    if len(root_states) == sim.num_envs:
        return root_states[env_ids]
    if len(root_states) == len(env_ids):
        return root_states
    raise ValueError(
        f"root_states dimension mismatch: got {len(root_states)}, "
        f"expected either {sim.num_envs} (global) or {len(env_ids)} (pre-sliced)"
    )
