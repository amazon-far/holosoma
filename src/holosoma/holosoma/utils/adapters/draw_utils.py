"""Shared utilities for draw adapters across different simulators."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from holosoma.utils.safe_torch_import import torch


def convert_to_numpy(
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
) -> npt.NDArray[np.float64]:
    """Convert position input to numpy array.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as numpy array with dtype float64
    """
    if torch is not None and isinstance(pos, torch.Tensor):
        return cast("npt.NDArray[np.float64]", pos.cpu().numpy().astype(np.float64))
    if isinstance(pos, (list, tuple)):
        return np.array(pos, dtype=np.float64)
    return np.array(pos, dtype=np.float64)


def convert_to_list(
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
) -> list[float]:
    """Convert position input to list of floats.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as list of floats
    """
    if torch is not None and isinstance(pos, torch.Tensor):
        return cast("list[float]", pos.cpu().numpy().astype(np.float64).tolist())
    if isinstance(pos, np.ndarray):
        return cast("list[float]", pos.astype(np.float64).tolist())
    if isinstance(pos, (list, tuple)):
        return list(pos)
    return list(pos)


def convert_to_tuple(
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
) -> tuple[float, ...]:
    """Convert position input to tuple of floats.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as tuple of 3 floats
    """
    converted = convert_to_list(pos)
    if len(converted) != 3:
        raise ValueError(f"Expected 3D position, got {len(converted)} dimensions")
    return tuple(converted)
