"""
MuJoCo Proxy System

This module provides a tensor-like interface into Mujoco's state so envs/tasks have a unified interface
across IsaacGym, IsaacLab and Mujoco.

The current implementation is intended for evaluation only. The performance may be suboptimal
for training many parallel environments and requires profiling.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Tuple

import numpy as np
import torch

from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse


def quat_mujoco_to_holosoma(quat_mujoco: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from MuJoCo format [w, x, y, z] to holosoma format [x, y, z, w].
    """
    return quat_mujoco[..., [1, 2, 3, 0]]


def quat_holosoma_to_mujoco(quat_holosoma: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from holosoma format [x, y, z, w] to MuJoCo format [w, x, y, z].
    """
    return quat_holosoma[..., [3, 0, 1, 2]]


class BaseMujocoView(Protocol):
    """
    Protocol defining the interface for MuJoCo view objects.

    This uses structural subtyping (Protocol) rather than inheritance,
    allowing different backend implementations to satisfy the interface
    without coupling them together.
    """

    device: str
    _is_tensor_proxy: bool

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the view."""
        ...

    def __getitem__(self, key) -> torch.Tensor:
        """Get data from the view, returning a PyTorch tensor."""
        ...

    def __setitem__(self, key, value) -> None:
        """Set data in the view from a PyTorch tensor or array."""
        ...

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        ...

    def dim(self) -> int:
        """Return the number of dimensions."""
        ...

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the view data."""
        ...


class BaseMujocoViewMixin:
    """
    Mixin providing common functionality for MuJoCo views.

    This class provides default implementations of common methods
    that views can inherit if desired. Views don't need to inherit
    from this - they just need to satisfy the BaseMujocoView Protocol.
    """

    device: str
    _is_tensor_proxy: bool = True

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the view. Must be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement shape")

    def __getitem__(self, key) -> torch.Tensor:
        """Get data from the view. Must be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self.shape[0]

    def dim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the view data."""
        return self[:].clone()

    def repeat(self, *sizes):
        """Repeat the tensor along specified dimensions."""
        return self[:].repeat(*sizes)

    def unsqueeze(self, dim):
        """Add a dimension of size 1 at the specified position."""
        return self[:].unsqueeze(dim)

    def view(self, *shape):
        """Return a new tensor with the same data but different shape."""
        return self[:].view(*shape)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape} device={self.device}>"

    @classmethod
    def __torch_function__(cls, func: Callable, types: tuple, args: tuple = (), kwargs: dict | None = None) -> Any:
        """
        PyTorch compatibility for critical operations.

        Converts any view instance to a tensor before calling the PyTorch function.
        """
        if kwargs is None:
            kwargs = {}

        new_args = []
        for arg in args:
            # Check if the argument is one of our proxy views
            if getattr(arg, "_is_tensor_proxy", False):
                new_args.append(arg[:])  # Convert to tensor via __getitem__
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if getattr(value, "_is_tensor_proxy", False):
                new_kwargs[key] = value[:]  # Convert to tensor via __getitem__
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)


class MujocoView(BaseMujocoViewMixin):
    """
    View into MuJoCo arrays with tensor-like interface.

    Provides tensor-like access to slices of MuJoCo data arrays with optional
    data transformation functions for format conversion.

    Allows tasks to interface with Mujoco like they do with IsaacLab and IsaacGym, albeit potentially
    sub-optimally depending on conversions.
    """

    def __init__(
        self,
        base_array: np.ndarray,
        indices: slice | np.ndarray,
        transform_get: Callable | None = None,
        transform_set: Callable | None = None,
        device: str = "cpu",
    ):
        """
        Initialize a view into a MuJoCo data array.

        Args:
            base_array: The source NumPy array (e.g., mjData.qpos)
            indices: Slice or array indices defining the view
            transform_get: Optional function to transform data when reading
            transform_set: Optional function to transform data when writing
            device: PyTorch device for tensor operations
        """
        self.base_array = base_array
        self.indices = indices
        self.transform_get = transform_get
        self.transform_set = transform_set
        self.device = device

    def __getitem__(self, key) -> torch.Tensor:
        """Get data from the view, returning a PyTorch tensor."""
        raw_data = self.base_array[self.indices]

        if self.transform_get is not None:
            raw_data = self.transform_get(raw_data)

        # NOTE: We avoid .copy() for performance. This may fail if the base
        # MuJoCo array is not writable or has negative strides.
        tensor_data = torch.from_numpy(raw_data).to(self.device, dtype=torch.float32)

        # Handle partial indexing like [0, :3], [0], or full slicing [:]
        return tensor_data[key]

    def __setitem__(self, key, value):
        """Set data in the view from a PyTorch tensor or numpy array."""
        if isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = np.asarray(value)

        if key == slice(None, None, None):
            # Full slicing [:] - set all data
            if self.transform_set is not None:
                np_value = self.transform_set(np_value)
            self.base_array[self.indices] = np_value
        else:
            # Partial indexing - this read-modify-write is less efficient but
            # correctly handles transforms that rely on the full view context.
            current_data = self.base_array[self.indices]
            if self.transform_get is not None:
                current_data = self.transform_get(current_data)

            if isinstance(current_data, torch.Tensor):
                current_data = current_data.detach().cpu().numpy()

            if isinstance(key, torch.Tensor):
                key = key.detach().cpu().numpy()

            current_data[key] = np_value

            if self.transform_set is not None:
                current_data = self.transform_set(current_data)

            self.base_array[self.indices] = current_data

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the view."""
        if self.transform_get:
            # If a transform exists, its output shape is authoritative
            return self.transform_get(self.base_array[self.indices]).shape

        # Calculate shape based on indices as a fallback
        if isinstance(self.indices, slice):
            start, stop, step = self.indices.indices(self.base_array.shape[0])
            length = len(range(start, stop, step))
            if self.base_array.ndim > 1:
                return (length,) + self.base_array.shape[1:]
            return (length,)
        return self.base_array[self.indices].shape


class MujocoRootStateView:
    """
    Specialized view for robot root states with quaternion format conversion.

    Handles the 13-element root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]

    Note: This class satisfies the BaseMujocoView Protocol through structural subtyping.
    """

    _is_tensor_proxy: bool = True

    def __init__(
        self,
        qpos_array: np.ndarray,
        qvel_array: np.ndarray,
        pos_indices: slice,
        quat_indices: slice,
        vel_indices: slice,
        ang_vel_indices: slice,
        num_envs: int,
        device: str = "cpu",
    ):
        self.qpos_array = qpos_array
        self.qvel_array = qvel_array
        self.pos_indices = pos_indices
        self.quat_indices = quat_indices
        self.vel_indices = vel_indices
        self.ang_vel_indices = ang_vel_indices
        self.num_envs = num_envs
        self.device = device

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_envs, 13)

    def __getitem__(self, key) -> torch.Tensor:
        """Get 13-element root state with quaternion conversion."""
        pos = self.qpos_array[self.pos_indices].reshape(self.num_envs, 3)
        quat_mujoco = self.qpos_array[self.quat_indices].reshape(self.num_envs, 4)
        lin_vel = self.qvel_array[self.vel_indices].reshape(self.num_envs, 3)
        ang_vel = self.qvel_array[self.ang_vel_indices].reshape(self.num_envs, 3)

        quat_holosoma = quat_mujoco_to_holosoma(quat_mujoco)
        root_state = np.column_stack([pos, quat_holosoma, lin_vel, ang_vel])

        tensor_data = torch.from_numpy(root_state).to(self.device, dtype=torch.float32)

        # MuJoCo's freejoint qvel stores angular velocity in the BODY-LOCAL frame, but the
        # unified robot_root_states contract is WORLD-frame for both velocities (matching
        # IsaacGym/IsaacSim, and what every consumer of robot_root_states[:, 10:13] expects).
        # Rotate the body-local angular velocity into world using the root's world orientation.
        tensor_data[:, 10:13] = quat_rotate(tensor_data[:, 3:7], tensor_data[:, 10:13], w_last=True)

        return tensor_data[key]

    def __setitem__(self, key, value):
        """Set 13-element root state with quaternion conversion."""
        if isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = np.asarray(value)

        if key == slice(None, None, None):
            # Full slicing [:] - set all environments
            np_value = np_value.reshape(self.num_envs, 13)

            pos = np_value[:, 0:3]
            quat_holosoma = np_value[:, 3:7]
            lin_vel = np_value[:, 7:10]
            # Incoming angular velocity is WORLD-frame (unified contract, symmetric with
            # __getitem__); MuJoCo's freejoint qvel wants it BODY-LOCAL, so rotate it back
            # using the root's world orientation before scattering into qvel.
            ang_vel = (
                quat_rotate_inverse(
                    torch.from_numpy(quat_holosoma).float(),
                    torch.from_numpy(np_value[:, 10:13]).float(),
                    w_last=True,
                )
                .cpu()
                .numpy()
            )

            quat_mujoco = quat_holosoma_to_mujoco(quat_holosoma)

            self.qpos_array[self.pos_indices] = pos.flatten()
            self.qpos_array[self.quat_indices] = quat_mujoco.flatten()
            self.qvel_array[self.vel_indices] = lin_vel.flatten()
            self.qvel_array[self.ang_vel_indices] = ang_vel.flatten()
        else:
            # Partial indexing: Read-modify-write is inefficient but safe.
            current_root_state = self[:]  # Gets full state as a tensor
            current_root_state[key] = torch.as_tensor(value, device=self.device)
            self[:] = current_root_state  # Write back the full state

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self.num_envs

    def dim(self) -> int:
        """Return the number of dimensions."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the root state."""
        return self[:].clone()


class MujocoDofStateView:
    """
    Specialized view for DOF states compatible with IsaacGym's flattened format.

    Provides access to joint positions and velocities as a (num_envs * num_dof, 2) array.

    Note: This class satisfies the BaseMujocoView Protocol through structural subtyping.
    """

    _is_tensor_proxy: bool = True

    def __init__(
        self,
        qpos_array: np.ndarray,
        qvel_array: np.ndarray,
        dof_pos_indices: slice,
        dof_vel_indices: slice,
        num_envs: int,
        num_dof: int,
        device: str = "cpu",
    ):
        self.qpos_array = qpos_array
        self.qvel_array = qvel_array
        self.dof_pos_indices = dof_pos_indices
        self.dof_vel_indices = dof_vel_indices
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.device = device

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_envs * self.num_dof, 2)

    def __getitem__(self, key) -> torch.Tensor:
        """Get DOF states in IsaacGym flattened format."""
        dof_pos = self.qpos_array[self.dof_pos_indices].reshape(self.num_envs, self.num_dof)
        dof_vel = self.qvel_array[self.dof_vel_indices].reshape(self.num_envs, self.num_dof)

        dof_state = np.stack([dof_pos, dof_vel], axis=-1)
        dof_state_flat = dof_state.reshape(-1, 2)

        tensor_data = torch.from_numpy(dof_state_flat).to(self.device, dtype=torch.float32)

        return tensor_data[key]

    def __setitem__(self, key, value):
        """Set DOF states from IsaacGym flattened format."""
        if key != slice(None, None, None):
            raise IndexError("Only full slicing `[:]` is supported for setting DOF state.")

        if isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = np.asarray(value)

        dof_state = np_value.reshape(self.num_envs, self.num_dof, 2)
        dof_pos = dof_state[:, :, 0]
        dof_vel = dof_state[:, :, 1]

        self.qpos_array[self.dof_pos_indices] = dof_pos.flatten()
        self.qvel_array[self.dof_vel_indices] = dof_vel.flatten()

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self.num_envs * self.num_dof

    def dim(self) -> int:
        """Return the number of dimensions."""
        return 2

    def clone(self) -> torch.Tensor:
        """Return a cloned tensor of the DOF state."""
        return self[:].clone()


# --- Factory functions for common view types ---


def create_quaternion_view(qpos_array: np.ndarray, indices: slice, num_envs: int, device: str = "cpu") -> MujocoView:
    """Create a view for quaternions with format conversion."""

    def reshape_quat_get(data):
        reshaped_data = data.reshape(num_envs, 4)
        return quat_mujoco_to_holosoma(reshaped_data)

    def reshape_quat_set(data):
        quat_mujoco = quat_holosoma_to_mujoco(data)
        return quat_mujoco.flatten()

    return MujocoView(
        qpos_array, indices, transform_get=reshape_quat_get, transform_set=reshape_quat_set, device=device
    )


def create_dof_position_view(
    qpos_array: np.ndarray, indices: slice, num_envs: int, num_dof: int, device: str = "cpu"
) -> MujocoView:
    """Create a view for DOF positions reshaped for multi-env."""

    def reshape_get(data):
        return data.reshape(num_envs, num_dof)

    def reshape_set(data):
        return data.flatten()

    return MujocoView(qpos_array, indices, transform_get=reshape_get, transform_set=reshape_set, device=device)


def create_dof_velocity_view(
    qvel_array: np.ndarray, indices: slice, num_envs: int, num_dof: int, device: str = "cpu"
) -> MujocoView:
    """Create a view for DOF velocities reshaped for multi-env."""

    def reshape_get(data):
        return data.reshape(num_envs, num_dof)

    def reshape_set(data):
        return data.flatten()

    return MujocoView(qvel_array, indices, transform_get=reshape_get, transform_set=reshape_set, device=device)


def create_dof_acceleration_view(
    qacc_array: np.ndarray, indices: slice, num_envs: int, num_dof: int, device: str = "cpu"
) -> MujocoView:
    """Create a view for DOF accelerations reshaped for multi-env.

    Parameters
    ----------
    qacc_array : np.ndarray
        MuJoCo's qacc array
    indices : slice
        Slice object defining which elements to view
    num_envs : int
        Number of environments
    num_dof : int
        Number of degrees of freedom
    device : str
        Device for the tensor

    Returns
    -------
    MujocoView
        View of DOF accelerations with shape [num_envs, num_dof]
    """

    def reshape_get(data):
        return data.reshape(num_envs, num_dof)

    def reshape_set(data):
        return data.flatten()

    return MujocoView(qacc_array, indices, transform_get=reshape_get, transform_set=reshape_set, device=device)


def create_base_angular_velocity_view(
    qvel_array: np.ndarray, indices: slice, num_envs: int, device: str = "cpu"
) -> MujocoView:
    """Create a view for base angular velocity reshaped for multi-env.

    Parameters
    ----------
    qvel_array : np.ndarray
        MuJoCo's qvel array
    indices : slice
        Slice object defining which elements to view
    num_envs : int
        Number of environments
    device : str
        Device for the tensor

    Returns
    -------
    MujocoView
        View of base angular velocity with shape [num_envs, 3]
    """

    def reshape_get(data):
        return data.reshape(num_envs, 3)

    def reshape_set(data):
        return data.flatten()

    return MujocoView(qvel_array, indices, transform_get=reshape_get, transform_set=reshape_set, device=device)


def create_base_linear_acceleration_view(
    qacc_array: np.ndarray, indices: slice, num_envs: int, device: str = "cpu"
) -> MujocoView:
    """Create a view for base linear acceleration reshaped for multi-env.

    Parameters
    ----------
    qacc_array : np.ndarray
        MuJoCo's qacc array
    indices : slice
        Slice object defining which elements to view
    num_envs : int
        Number of environments
    device : str
        Device for the tensor

    Returns
    -------
    MujocoView
        View of base linear acceleration with shape [num_envs, 3]
    """

    def reshape_get(data):
        return data.reshape(num_envs, 3)

    def reshape_set(data):
        return data.flatten()

    return MujocoView(qacc_array, indices, transform_get=reshape_get, transform_set=reshape_set, device=device)
