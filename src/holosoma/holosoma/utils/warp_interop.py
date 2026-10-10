"""Checked Torch interoperability helpers for Warp arrays."""

from __future__ import annotations

from typing import Any

import torch


def from_torch_vec3(tensor: torch.Tensor, device: Any) -> Any:
    """Wrap a contiguous float32 ``[..., 3]`` Torch tensor as a zero-copy Warp array."""
    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected a float32 Torch tensor, got {tensor.dtype}")
    if tensor.ndim < 2 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected a Torch tensor with shape [..., 3], got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError("Expected a contiguous Torch tensor")

    import warp as wp

    tensor_wp = wp.from_torch(tensor, dtype=wp.vec3)
    if str(tensor_wp.device) != str(device):
        raise ValueError(f"Torch tensor is on {tensor_wp.device}, but the Warp data is on {device}")
    return tensor_wp
