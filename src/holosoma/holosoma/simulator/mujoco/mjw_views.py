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

2. **MjwDofStateView**: Handles IsaacGym-compatible DOF state flattening
   - Converts between MuJoCo's separate arrays and holosoma's flattened format
   - Enables compatibility with existing training code
   - Required for state resets and DOF manipulation

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

from typing import Tuple

import torch

from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse


class MjwDofStateView:
    """Zero-copy view for DOF states in IsaacGym flattened format.

    Why this view is necessary
    --------------------------
    Raw PyTorch tensors cannot handle the format transformation between MuJoCo's
    native DOF storage and holosoma's expected IsaacGym-compatible format:

    - **MuJoCo format**: Separate qpos[num_envs, nq] and qvel[num_envs, nv] arrays
    - **holosoma expects**: Flattened [num_envs * num_dof, 2] where [:, 0] = pos, [:, 1] = vel

    Without this view:
    - Code would need manual reshaping at every DOF access point
    - Training code would break (expects IsaacGym-compatible interface)
    - State resets would require complex scatter operations

    What it does:
    - Reads: Automatically stacks qpos + qvel and flattens to [N*D, 2] format
    - Writes: Automatically unflattens and scatters to separate qpos/qvel arrays
    - Zero-copy: Direct GPU memory access to underlying Warp arrays

    This view wraps GPU tensors from WarpBackend and provides zero-copy access
    compatible with the IsaacGym interface used throughout holosoma.
    """

    # Flag for PyTorch compatibility
    _is_tensor_proxy: bool = True

    def __init__(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        dof_pos_indices: slice,
        dof_vel_indices: slice,
        num_envs: int,
        num_dof: int,
    ):
        """Initialize DOF state view.

        Parameters
        ----------
        qpos : torch.Tensor
            Position state tensor [num_envs, nq] (tethered to Warp)
        qvel : torch.Tensor
            Velocity state tensor [num_envs, nv] (tethered to Warp)
        dof_pos_indices : slice
            Slice for extracting DOF positions
        dof_vel_indices : slice
            Slice for extracting DOF velocities
        num_envs : int
            Number of parallel environments
        num_dof : int
            Number of degrees of freedom
        """
        self.qpos = qpos
        self.qvel = qvel
        self.dof_pos_indices = dof_pos_indices
        self.dof_vel_indices = dof_vel_indices
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.device = str(qpos.device)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape [num_envs * num_dof, 2]."""
        return (self.num_envs * self.num_dof, 2)

    def __getitem__(self, key) -> torch.Tensor:
        """Read DOF states in IsaacGym flattened format.

        Returns states as [num_envs * num_dof, 2] where:
        - [:, 0] = positions
        - [:, 1] = velocities

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification

        Returns
        -------
        torch.Tensor
            DOF states in IsaacGym format
        """
        # Extract DOF positions and velocities
        dof_pos = self.qpos[:, self.dof_pos_indices]  # [num_envs, num_dof]
        dof_vel = self.qvel[:, self.dof_vel_indices]  # [num_envs, num_dof]

        # Stack into IsaacGym format: [num_envs, num_dof, 2]
        dof_state = torch.stack([dof_pos, dof_vel], dim=2)

        # Flatten to [num_envs * num_dof, 2]
        dof_state_flat = dof_state.reshape(-1, 2)

        return dof_state_flat[key]

    def __setitem__(self, key, value):
        """Write DOF states from IsaacGym flattened format.

        Expects states as [num_envs * num_dof, 2] where:
        - [:, 0] = positions
        - [:, 1] = velocities

        Parameters
        ----------
        key : int, slice, tuple, or tensor
            Index or slice specification
        value : torch.Tensor, np.ndarray, or sequence
            DOF state data to write
        """
        if isinstance(value, torch.Tensor):
            val = value.to(self.device)
        else:
            val = torch.tensor(value, device=self.device)

        # Optimized path for full slice [:] (most common case)
        if key == slice(None):
            # Input: [num_envs * num_dof, 2]
            # Reshape to [num_envs, num_dof, 2]
            dof_state = val.reshape(self.num_envs, self.num_dof, 2)

            dof_pos = dof_state[:, :, 0]  # [num_envs, num_dof]
            dof_vel = dof_state[:, :, 1]  # [num_envs, num_dof]

            # Write to underlying tensors (modifies Warp arrays via zero-copy)
            self.qpos[:, self.dof_pos_indices] = dof_pos
            self.qvel[:, self.dof_vel_indices] = dof_vel
        else:
            # Partial indexing requires read-modify-write
            current = self[:]
            current[key] = val
            self[:] = current

    def __len__(self) -> int:
        """Return the total number of DOF states."""
        return self.num_envs * self.num_dof

    def dim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the DOF states."""
        return self[:].clone()

    def __repr__(self) -> str:
        return f"<MjwDofStateView shape={self.shape} device={self.device}>"

    def copy_(self, other):
        """Support in-place copy: self.copy_(other)

        Mimics PyTorch's tensor.copy_() API for compatibility with
        code that uses in-place tensor operations.

        Parameters
        ----------
        other : torch.Tensor, np.ndarray, or view
            Source data to copy from

        Returns
        -------
        self
            Returns self for method chaining
        """
        self[:] = other
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Enable PyTorch operations to work with our views.

        This allows PyTorch functions to accept our view objects by automatically
        converting them to tensors when used as arguments.

        Parameters
        ----------
        func : callable
            PyTorch function being called
        types : tuple
            Types of all tensor-like arguments
        args : tuple
            Positional arguments to the function
        kwargs : dict
            Keyword arguments to the function

        Returns
        -------
        result
            Result of the PyTorch function with converted arguments
        """
        if kwargs is None:
            kwargs = {}

        # Convert any of our view instances in args to tensors
        args = tuple(arg[:] if isinstance(arg, (MjwDofStateView, MjwRootStateView)) else arg for arg in args)

        # Let PyTorch handle the actual operation with real tensors
        return func(*args, **kwargs)


class MjwQuaternionView:
    """Zero-copy view for quaternions with format conversion.

    Why this view is necessary
    --------------------------
    Raw PyTorch tensor slices cannot handle the quaternion convention mismatch:

    - **MuJoCo format**: [w, x, y, z] (scalar-first)
    - **holosoma expects**: [x, y, z, w] (scalar-last, matching IsaacGym/IsaacSim)

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

    def __init__(self, qpos: torch.Tensor, quat_slice: slice, num_envs: int):
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
    def shape(self) -> Tuple[int, ...]:
        """Return shape [num_envs, 4]."""
        return (self.num_envs, 4)

    def __getitem__(self, key) -> torch.Tensor:
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

    def __setitem__(self, key, value):
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

    def __init__(self, qvel: torch.Tensor, ang_vel_slice: slice, num_envs: int):
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
    def shape(self) -> Tuple[int, ...]:
        """Return shape [num_envs, 3]."""
        return (self.num_envs, 3)

    def __getitem__(self, key) -> torch.Tensor:
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

    def __setitem__(self, key, value):
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
       - holosoma expects: [x, y, z, w] (scalar-last, matching IsaacGym/IsaacSim)

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
    ):
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
    def shape(self) -> Tuple[int, ...]:
        """Return shape [num_envs, 13]."""
        return (self.num_envs, 13)

    def __getitem__(self, key) -> torch.Tensor:
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
        # IsaacGym/IsaacSim, and what every consumer of robot_root_states[:, 10:13] expects).
        # Rotate the body-local angular velocity into world using the root's world orientation.
        ang_vel = quat_rotate(quat_holo, ang_vel, w_last=True)

        # Assemble full state
        root_state = torch.cat([pos, quat_holo, lin_vel, ang_vel], dim=1)

        return root_state[key]

    def __setitem__(self, key, value):
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
