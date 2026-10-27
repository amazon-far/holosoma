"""Zero-copy tensor views for MuJoCo Warp backend.

ARCHITECTURE OVERVIEW
---------------------
This module provides specialized views for data transformations that cannot be
handled by raw PyTorch tensors alone. Views are ONLY used when meaningful semantic
transformation is required (format conversion, data assembly, etc.).

For simple array access (qpos, qvel, ctrl, cfrc_ext), WarpBackend returns raw PyTorch
tensors directly via wp.to_torch() - no wrapper needed. These already:
- Share memory with Warp arrays (zero-copy)
- Support all PyTorch operations natively
- Work seamlessly on GPU

VIEW CLASSES
------------
1. **MjwRootStateView**: Handles quaternion reordering and composite state assembly
   - Converts between MuJoCo [w,x,y,z] and holosoma [x,y,z,w] quaternion formats
   - Assembles 13-element state from separate qpos/qvel arrays
   - Critical for correct robot orientation (prevents physics instability)

USAGE PATTERNS
--------------
Views are designed for explicit read/write, not arithmetic:

✅ GOOD - Explicit tensor conversion:
    states = view[:]  # Get as tensor
    new_states = states + offset  # Compute on tensor
    view[:] = new_states  # Write back

❌ BAD - Direct arithmetic (NOT supported)
    view += offset  # Will fail - requires operators


ZERO-COPY SEMANTICS
-------------------
All views maintain zero-copy access to underlying Warp GPU arrays:
- Reads return fresh tensors (data gathered/transformed on demand)
- Writes directly modify Warp arrays (changes visible immediately in simulation)
- No CPU-GPU transfers except for viewer sync
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import torch

from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse

# Index passed to __getitem__/__setitem__ (forwarded to torch tensor indexing).
IndexKey = Union[int, slice, "torch.Tensor", tuple[Any, ...], Sequence[int]]
# Value accepted by __setitem__ / copy_ (converted to a tensor if not already one).
IndexValue = Union["torch.Tensor", "np.ndarray[Any, Any]", Sequence[Any], "MjwRootStateView"]


class MjwQuaternionView:
    """Zero-copy view for quaternions with format conversion.

    Why this view is necessary
    --------------------------
    Raw PyTorch tensor slices cannot handle the quaternion convention mismatch:

    - **MuJoCo format**: [w, x, y, z] (scalar-first)
    - **holosoma expects**: [x, y, z, w] (scalar-last, matching IsaacSim)

    Without this view:
    - Robot orientations would be incorrectly interpreted
    - Physics would become unstable due to wrong quaternion interpretation
    - Training would fail due to incorrect state observations

    What it does:
    - Reads: Reorders quaternion [w,x,y,z] -> [x,y,z,w]
    - Writes: Reorders quaternion [x,y,z,w] -> [w,x,y,z]
    - Zero-copy: Direct GPU memory access to underlying Warp array slice
    """

    # Flag for PyTorch compatibility
    _is_tensor_proxy: bool = True

    def __init__(self, qpos: torch.Tensor, quat_slice: slice, num_envs: int) -> None:
        """Initialize quaternion view.

        Parameters
        ----------
        qpos : torch.Tensor
            Position state tensor [num_envs, nq] (tethered to Warp)
        quat_slice : slice
            Slice for extracting quaternion [w, x, y, z]
        num_envs : int
            Number of parallel environments
        """
        self.qpos = qpos
        self.quat_slice = quat_slice
        self.num_envs = num_envs
        self.device = str(qpos.device)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape [num_envs, 4]."""
        return (self.num_envs, 4)

    def __getitem__(self, key: IndexKey) -> torch.Tensor:
        """Read quaternion with format conversion [w,x,y,z] -> [x,y,z,w].

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification

        Returns
        -------
        torch.Tensor
            Quaternion in holosoma format [x, y, z, w]
        """
        quat_mj = self.qpos[:, self.quat_slice]  # [N, 4] - [w, x, y, z]
        quat_holo = quat_mj[:, [1, 2, 3, 0]]  # [x, y, z, w]
        return quat_holo[key]

    def __setitem__(self, key: IndexKey, value: IndexValue) -> None:
        """Write quaternion with format conversion [x,y,z,w] -> [w,x,y,z].

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification
        value : torch.Tensor, np.ndarray, or sequence
            Quaternion data in holosoma format [x, y, z, w]
        """
        if isinstance(value, torch.Tensor):
            val = value.to(self.device)
        else:
            val = torch.tensor(value, device=self.device)

        # Optimized path for full slice [:] (most common case)
        if key == slice(None):
            # Input: [N, 4] in [x, y, z, w] format
            # Convert to MuJoCo [w, x, y, z]
            quat_mj = val[:, [3, 0, 1, 2]]
            self.qpos[:, self.quat_slice] = quat_mj
        else:
            # Partial indexing requires read-modify-write
            current = self[:]
            current[key] = val
            self[:] = current

    def __len__(self) -> int:
        """Return the number of environments."""
        return self.num_envs

    def dim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the quaternion."""
        return self[:].clone()

    def __repr__(self) -> str:
        return f"<MjwQuaternionView shape={self.shape} device={self.device}>"


class MjwAngularVelocityView:
    """Zero-copy view for angular velocity with proper multi-env reshaping.

    Why this view is necessary
    --------------------------
    While angular velocity doesn't require format conversion like quaternions,
    it needs proper reshaping for multi-environment access:

    - **Raw access**: qvel[:, slice] gives [num_envs, 3] but in flattened form
    - **holosoma expects**: Clean [num_envs, 3] tensor with proper indexing

    What it does:
    - Reads: Extracts and reshapes angular velocity to [num_envs, 3]
    - Writes: Writes angular velocity back with proper shape handling
    - Zero-copy: Direct GPU memory access to underlying Warp array slice
    """

    # Flag for PyTorch compatibility
    _is_tensor_proxy: bool = True

    def __init__(self, qvel: torch.Tensor, ang_vel_slice: slice, num_envs: int) -> None:
        """Initialize angular velocity view.

        Parameters
        ----------
        qvel : torch.Tensor
            Velocity state tensor [num_envs, nv] (tethered to Warp)
        ang_vel_slice : slice
            Slice for extracting angular velocity
        num_envs : int
            Number of parallel environments
        """
        self.qvel = qvel
        self.ang_vel_slice = ang_vel_slice
        self.num_envs = num_envs
        self.device = str(qvel.device)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape [num_envs, 3]."""
        return (self.num_envs, 3)

    def __getitem__(self, key: IndexKey) -> torch.Tensor:
        """Read angular velocity.

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification

        Returns
        -------
        torch.Tensor
            Angular velocity [num_envs, 3]
        """
        ang_vel = self.qvel[:, self.ang_vel_slice]  # [N, 3]
        return ang_vel[key]

    def __setitem__(self, key: IndexKey, value: IndexValue) -> None:
        """Write angular velocity.

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification
        value : torch.Tensor, np.ndarray, or sequence
            Angular velocity data
        """
        if isinstance(value, torch.Tensor):
            val = value.to(self.device)
        else:
            val = torch.tensor(value, device=self.device)

        # Optimized path for full slice [:] (most common case)
        if key == slice(None):
            self.qvel[:, self.ang_vel_slice] = val
        else:
            # Partial indexing requires read-modify-write
            current = self[:]
            current[key] = val
            self[:] = current

    def __len__(self) -> int:
        """Return the number of environments."""
        return self.num_envs

    def dim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the angular velocity."""
        return self[:].clone()

    def __repr__(self) -> str:
        return f"<MjwAngularVelocityView shape={self.shape} device={self.device}>"


class MjwRootStateView:
    """Composite view for robot root state with quaternion conversion.

    Why this view is necessary
    --------------------------
    Raw PyTorch tensors cannot handle the dual transformations required:

    1. **Quaternion Convention Mismatch**:
       - MuJoCo uses: [w, x, y, z] (scalar-first)
       - holosoma expects: [x, y, z, w] (scalar-last, matching IsaacSim)

    2. **Composite State Assembly**:
       - MuJoCo stores: pos/quat in qpos[nq], velocities in qvel[nv] (separate arrays)
       - holosoma expects: Single 13-element [pos(3), quat(4), lin_vel(3), ang_vel(3)]

    Without this view:
    - Every root state read would need manual quaternion reordering
    - State resets would break (wrong quaternion format → incorrect orientations)
    - Training code would fail (expects scalar-last quaternions)
    - Manual gathering from multiple arrays required at every access

    What it does:
    - Reads: Gathers from qpos/qvel, reorders quaternion [w,x,y,z] -> [x,y,z,w], concatenates
    - Writes: Splits 13-element state, reorders quaternion [x,y,z,w] -> [w,x,y,z], scatters to arrays
    - Zero-copy: Direct GPU memory access to underlying Warp arrays

    CRITICAL: Without this automatic conversion, robot orientations would be incorrectly
    interpreted, causing physics instability and training failure.
    """

    # Flag for PyTorch compatibility
    _is_tensor_proxy: bool = True

    def __init__(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        pos_slice: slice,
        quat_slice: slice,
        vel_slice: slice,
        ang_vel_slice: slice,
        num_envs: int,
    ) -> None:
        """Initialize root state view.

        Parameters
        ----------
        qpos : torch.Tensor
            Position state tensor [num_envs, nq] (tethered to Warp)
        qvel : torch.Tensor
            Velocity state tensor [num_envs, nv] (tethered to Warp)
        pos_slice : slice
            Slice for extracting position [x, y, z]
        quat_slice : slice
            Slice for extracting quaternion [w, x, y, z] (MuJoCo convention)
        vel_slice : slice
            Slice for extracting linear velocity
        ang_vel_slice : slice
            Slice for extracting angular velocity
        num_envs : int
            Number of parallel environments
        """
        self.qpos = qpos
        self.qvel = qvel
        self.pos_slice = pos_slice
        self.quat_slice = quat_slice
        self.vel_slice = vel_slice
        self.ang_vel_slice = ang_vel_slice
        self.num_envs = num_envs
        self.device = str(qpos.device)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape [num_envs, 13]."""
        return (self.num_envs, 13)

    def __getitem__(self, key: IndexKey) -> torch.Tensor:
        """Read root state with quaternion conversion.

        Returns 13-element state in holosoma convention:
        [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification

        Returns
        -------
        torch.Tensor
            Root state with quaternion in holosoma format
        """
        # Extract components
        pos = self.qpos[:, self.pos_slice]  # [N, 3]
        quat_mj = self.qpos[:, self.quat_slice]  # [N, 4] - [w, x, y, z]
        lin_vel = self.qvel[:, self.vel_slice]  # [N, 3]
        ang_vel = self.qvel[:, self.ang_vel_slice]  # [N, 3]

        # Convert quaternion: [w, x, y, z] -> [x, y, z, w]
        quat_holo = quat_mj[:, [1, 2, 3, 0]]

        # MuJoCo's freejoint qvel stores angular velocity in the BODY-LOCAL frame, but the
        # unified robot_root_states contract is WORLD-frame for both velocities (matching
        # IsaacSim, and what every consumer of robot_root_states[:, 10:13] expects).
        # Rotate the body-local angular velocity into world using the root's world orientation.
        ang_vel = quat_rotate(quat_holo, ang_vel, w_last=True)

        # Assemble full state
        root_state = torch.cat([pos, quat_holo, lin_vel, ang_vel], dim=1)

        return root_state[key]

    def __setitem__(self, key: IndexKey, value: IndexValue) -> None:
        """Write root state with quaternion conversion.

        Expects 13-element state in holosoma convention:
        [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification
        value : torch.Tensor, np.ndarray, or sequence
            Root state data to write
        """
        if isinstance(value, torch.Tensor):
            val = value.to(self.device)
        else:
            val = torch.tensor(value, device=self.device)

        # Optimized path for full slice [:] (most common case)
        if key == slice(None):
            # Input: [N, 13]
            pos = val[:, 0:3]
            quat_holo = val[:, 3:7]  # [x, y, z, w]
            lin_vel = val[:, 7:10]
            # Incoming angular velocity is WORLD-frame (unified contract, symmetric with
            # __getitem__); MuJoCo's freejoint qvel wants it BODY-LOCAL, so rotate it back
            # using the root's world orientation before scattering into qvel.
            ang_vel = quat_rotate_inverse(quat_holo, val[:, 10:13], w_last=True)

            # Convert quaternion: [x, y, z, w] -> [w, x, y, z]
            quat_mj = quat_holo[:, [3, 0, 1, 2]]

            # Write to underlying tensors (modifies Warp arrays via zero-copy)
            self.qpos[:, self.pos_slice] = pos
            self.qpos[:, self.quat_slice] = quat_mj
            self.qvel[:, self.vel_slice] = lin_vel
            self.qvel[:, self.ang_vel_slice] = ang_vel
        else:
            # Partial indexing requires read-modify-write
            current = self[:]
            current[key] = val
            self[:] = current

    def __len__(self) -> int:
        """Return the number of environments."""
        return self.num_envs

    def dim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the root state."""
        return self[:].clone()

    def __repr__(self) -> str:
        return f"<MjwRootStateView shape={self.shape} device={self.device}>"
