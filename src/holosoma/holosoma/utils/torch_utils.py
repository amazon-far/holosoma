# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md

"""Torch utility functions for tensor operations and RL-specific helpers.

This module contains utility functions for:
- Tensor conversion and random number generation
- RL trajectory processing (split, pad, unpad)
- Custom batched quaternion operations for [N, M, 3] shapes

Note:
    For general rotation and quaternion math, use holosoma.isaac_utils.rotations instead.
    This module only contains utilities and RL-specific helpers.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import numpy.typing as npt

from holosoma.utils.safe_torch_import import torch

# ============================================================================
# Math Utilities (from maths.py)
# ============================================================================


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a tensor along the last dimension.

    Args:
        x: Input tensor to normalize.
        eps: Small epsilon value to prevent division by zero.

    Returns:
        Normalized tensor.
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    """Copy the sign of tensor b to scalar a.

    Args:
        a: Scalar value.
        b: Tensor whose signs to copy.

    Returns:
        Tensor with magnitude of a and signs of b.
    """
    a_tensor = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_tensor) * torch.sign(b)


def set_seed(seed: int, torch_deterministic: bool = False) -> int:
    """Set random seed across all modules for reproducibility.

    Args:
        seed: Random seed value. If -1, generates random seed.
        torch_deterministic: If True, enables deterministic operations.

    Returns:
        The seed that was set.
    """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


# ============================================================================
# Tensor Utilities
# ============================================================================


def to_torch(
    x: npt.NDArray[np.float64] | list[Any] | tuple[Any, ...],
    dtype: torch.dtype = torch.float,
    device: str | torch.device = "cuda:0",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert input to torch tensor.

    Parameters
    ----------
    x : Union[npt.NDArray[np.float64], list[Any], tuple[Any, ...]]
        Input data to convert to tensor. Can be numpy array, list, or tuple.
    dtype : torch.dtype, optional
        Desired data type of the tensor, by default torch.float
    device : Union[str, torch.device], optional
        Device to place the tensor on, by default "cuda:0"
    requires_grad : bool, optional
        Whether to track gradients, by default False

    Returns
    -------
    torch.Tensor
        Converted tensor with specified dtype and device
    """
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate random float tensor.

    Parameters
    ----------
    lower : float
        Lower bound
    upper : float
        Upper bound
    shape : tuple[int, int] | torch.Size
        Shape of output tensor. Can be a tuple or torch.Size object.
    device : str
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Random tensor of specified shape
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def get_axis_params(value: float, axis_idx: int, dtype: npt.DTypeLike = np.float64, n_dims: int = 3) -> list[float]:
    """Construct arguments to `Vec` according to axis index.

    Parameters
    ----------
    value : float
        Value to set at axis_idx
    axis_idx : int
        Index of axis to set value
    dtype : npt.DTypeLike, optional
        Output dtype, by default np.float64
    n_dims : int, optional
        Number of dimensions, by default 3

    Returns
    -------
    list[float]
        list of parameters with specified values
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    return list(params.astype(dtype))
