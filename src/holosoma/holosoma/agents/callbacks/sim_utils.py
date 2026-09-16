"""Utilities for interfacing with IsaacSim's external force API.

Provides body resolution and force application helpers used by eval callbacks.
"""

from __future__ import annotations

from typing import Any

from holosoma.utils.safe_torch_import import torch


def resolve_body(sim: Any, body_name: str) -> tuple[str, int]:
    """Resolve body name to (name, isaac_body_id).

    ``find_rigid_body_indice`` returns a holosoma-ordered index (BFS order
    used consistently across simulators, for contact sensors and other
    holosoma-level operations).  ``sim.body_ids[idx]`` maps that to the raw
    IsaacLab body ID used by ``_robot.data.body_pos_w``,
    ``_robot.data.body_quat_w``, and ``set_external_force_and_torque``.

    Raises ValueError if not found.
    """
    body_names_idx = sim.find_rigid_body_indice(body_name)
    if isinstance(body_names_idx, int) and body_names_idx >= 0:
        return (body_name, sim.body_ids[body_names_idx])
    available = list(getattr(sim, "body_names", []))
    raise ValueError(f"Body '{body_name}' not found. Available: {available}")


def apply_body_force_world(
    sim: Any,
    env_ids: torch.Tensor,
    isaac_body_id: int,
    force_world: torch.Tensor,
) -> None:
    """Apply world-frame force on a body across envs.

    Converts from world frame to body-local frame (required by
    ``set_external_force_and_torque``) using the body's current orientation.

    Args:
        sim: Simulator instance.
        env_ids: [N] env indices.
        isaac_body_id: Body to apply force to.
        force_world: [N, 3] or [3] force vector in world frame.
    """
    from isaaclab.utils.math import quat_apply_inverse

    if force_world.dim() == 1:
        force_world = force_world.unsqueeze(0).expand(len(env_ids), -1)
    body_quats = sim._robot.data.body_quat_w[env_ids, isaac_body_id]
    force_body = quat_apply_inverse(body_quats, force_world)
    forces = force_body.unsqueeze(1)  # [N, 1, 3]
    torques = torch.zeros_like(forces)
    body_ids = torch.tensor([isaac_body_id], device=env_ids.device)
    sim._robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        env_ids=env_ids,
        body_ids=body_ids,
    )


def clear_body_force(
    sim: Any,
    env_ids: torch.Tensor,
    isaac_body_id: int,
) -> None:
    """Clear external forces on a body for given envs."""
    zero = torch.zeros(len(env_ids), 1, 3, device=env_ids.device)
    body_ids = torch.tensor([isaac_body_id], device=env_ids.device)
    sim._robot.set_external_force_and_torque(
        forces=zero,
        torques=zero,
        env_ids=env_ids,
        body_ids=body_ids,
    )
