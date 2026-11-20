"""Rotation utilities for 6D rotation representation."""

from __future__ import annotations

import torch


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to 6D rotation representation.
    
    Args:
        quat: Quaternion in xyzw format [..., 4]
    
    Returns:
        rot6d: First two columns of rotation matrix [..., 6]
    """
    # Convert to rotation matrix
    rotmat = quat_to_rotmat(quat)
    
    # Take first two columns: rotmat[..., :, :2] gives shape [..., 3, 2]
    # Transpose to [..., 2, 3] then flatten to get column-major order
    rot6d = rotmat[..., :, :2].transpose(-2, -1).reshape(*quat.shape[:-1], 6)
    
    return rot6d


def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to 3x3 rotation matrix.
    
    Args:
        rot6d: 6D rotation [..., 6]
    
    Returns:
        rotmat: Rotation matrix [..., 3, 3]
    """
    # Reshape to get first two columns
    x = rot6d[..., :3]
    y = rot6d[..., 3:6]
    
    # Gram-Schmidt orthonormalization
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.norm(y, dim=-1, keepdim=True)
    
    # Third column is cross product
    z = torch.cross(x, y, dim=-1)
    
    # Stack to form rotation matrix
    rotmat = torch.stack([x, y, z], dim=-1)
    
    return rotmat


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to quaternion.
    
    Args:
        rot6d: 6D rotation [..., 6]
    
    Returns:
        quat: Quaternion in xyzw format [..., 4]
    """
    rotmat = rot6d_to_rotmat(rot6d)
    return rotmat_to_quat(rotmat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion in xyzw format [..., 4]
    
    Returns:
        rotmat: Rotation matrix [..., 3, 3]
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Compute rotation matrix elements
    rotmat = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)
    
    return rotmat


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion.
    
    Args:
        rotmat: Rotation matrix [..., 3, 3]
    
    Returns:
        quat: Quaternion in xyzw format [..., 4]
    """
    # Use Shepperd's method for numerical stability
    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]
    
    quat = torch.zeros(*rotmat.shape[:-2], 4, device=rotmat.device, dtype=rotmat.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    quat[mask, 3] = 0.25 * s
    quat[mask, 0] = (rotmat[mask, 2, 1] - rotmat[mask, 1, 2]) / s
    quat[mask, 1] = (rotmat[mask, 0, 2] - rotmat[mask, 2, 0]) / s
    quat[mask, 2] = (rotmat[mask, 1, 0] - rotmat[mask, 0, 1]) / s
    
    # Case 2: trace <= 0
    mask = ~mask
    if mask.any():
        # Find largest diagonal element
        diag = torch.stack([rotmat[..., 0, 0], rotmat[..., 1, 1], rotmat[..., 2, 2]], dim=-1)
        k = torch.argmax(diag[mask], dim=-1)
        
        # Simplified: use w-dominant case for remaining
        s = torch.sqrt(1.0 + rotmat[mask, 0, 0] + rotmat[mask, 1, 1] + rotmat[mask, 2, 2]) * 2
        quat[mask, 3] = 0.25 * s
        quat[mask, 0] = (rotmat[mask, 2, 1] - rotmat[mask, 1, 2]) / s
        quat[mask, 1] = (rotmat[mask, 0, 2] - rotmat[mask, 2, 0]) / s
        quat[mask, 2] = (rotmat[mask, 1, 0] - rotmat[mask, 0, 1]) / s
    
    return quat


def world_to_reference_frame(pos_world: torch.Tensor, quat_world: torch.Tensor, 
                             ref_pos: torch.Tensor, ref_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from world frame to a reference frame.
    
    Args:
        pos_world: Position in world frame [..., 3]
        quat_world: Orientation in world frame (xyzw) [..., 4]
        ref_pos: Reference frame origin in world frame [..., 3]
        ref_quat: Reference frame orientation in world frame (xyzw) [..., 4]
    
    Returns:
        pos_ref: Position in reference frame [..., 3]
        quat_ref: Orientation in reference frame (xyzw) [..., 4]
    """
    from holosoma.utils.rotations import quat_inverse, quat_mul, quat_apply
    
    # Transform position
    pos_relative = pos_world - ref_pos
    ref_quat_inv = quat_inverse(ref_quat, w_last=True)
    pos_ref = quat_apply(ref_quat_inv, pos_relative, w_last=True)
    
    # Transform orientation
    quat_ref = quat_mul(ref_quat_inv, quat_world, w_last=True)
    
    return pos_ref, quat_ref


def reference_frame_to_world(pos_ref: torch.Tensor, quat_ref: torch.Tensor,
                             ref_pos: torch.Tensor, ref_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from reference frame to world frame.
    
    Args:
        pos_ref: Position in reference frame [..., 3]
        quat_ref: Orientation in reference frame (xyzw) [..., 4]
        ref_pos: Reference frame origin in world frame [..., 3]
        ref_quat: Reference frame orientation in world frame (xyzw) [..., 4]
    
    Returns:
        pos_world: Position in world frame [..., 3]
        quat_world: Orientation in world frame (xyzw) [..., 4]
    """
    from holosoma.utils.rotations import quat_mul, quat_apply
    
    # Transform position
    pos_world = quat_apply(ref_quat, pos_ref, w_last=True) + ref_pos
    
    # Transform orientation
    quat_world = quat_mul(ref_quat, quat_ref, w_last=True)
    
    return pos_world, quat_world


def heading_quat_from_root_quat(root_quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from root orientation using projected forward axis.

    The heading yaw is defined as atan2(f_y, f_x) where f = R_WR @ e_x
    is the root's forward direction projected onto the world xy-plane.

    Args:
        root_quat: Root orientation in world frame (xyzw) [..., 4]

    Returns:
        heading_quat: Yaw-only quaternion (xyzw) [..., 4]
    """
    # Get rotation matrix from quaternion
    R_WR = quat_to_rotmat(root_quat)

    # Forward axis in world: first column of R_WR
    f = R_WR[..., :, 0]  # [..., 3]
    fx = f[..., 0]
    fy = f[..., 1]

    # Heading yaw
    psi = torch.atan2(fy, fx)
    half_psi = psi * 0.5

    # Yaw-only quaternion: rotation about z-axis => [0, 0, sin(psi/2), cos(psi/2)]
    heading_quat = torch.zeros_like(root_quat)
    heading_quat[..., 2] = torch.sin(half_psi)
    heading_quat[..., 3] = torch.cos(half_psi)

    return heading_quat
