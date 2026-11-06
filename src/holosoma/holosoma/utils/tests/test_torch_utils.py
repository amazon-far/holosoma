from __future__ import annotations

import torch

from holosoma.utils.torch_utils import (
    normalize,
)


def test_normalize() -> None:
    # Test normalization of vector
    v = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)
    norm_v = normalize(v)
    assert torch.allclose(torch.norm(norm_v), torch.tensor(1.0), atol=1e-6)

    # Test with zero vector - should return zero vector
    zero = torch.zeros(3, dtype=torch.float32)
    norm_zero = normalize(zero)
    expected = torch.zeros(3, dtype=torch.float32)
    assert torch.allclose(norm_zero, expected, atol=1e-6)
