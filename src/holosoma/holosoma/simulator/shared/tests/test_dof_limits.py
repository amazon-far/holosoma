"""``dof_limits.soft_limits``: the consumer-side soft contraction of hard limits (pure)."""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.shared.dof_limits import soft_limits

pytestmark = pytest.mark.no_sim


def test_soft_limits_contracts_each_row_toward_its_midpoint() -> None:
    hard = torch.tensor([[-1.0, 1.0], [0.0, 4.0], [-3.0, -1.0]])
    assert torch.allclose(soft_limits(hard, 0.5), torch.tensor([[-0.5, 0.5], [1.0, 3.0], [-2.5, -1.5]]))
    assert torch.equal(soft_limits(hard, 1.0), hard)


def test_soft_limits_preserve_unbounded_and_one_sided_rows_without_nan() -> None:
    hard = torch.tensor(
        [
            [float("-inf"), float("inf")],
            [float("-inf"), 2.0],
            [-3.0, float("inf")],
            [-1.0, 1.0],
        ]
    )

    soft = soft_limits(hard, 0.5)

    assert torch.equal(soft[:3], hard[:3])
    assert torch.equal(soft[3], torch.tensor([-0.5, 0.5]))
    assert not torch.isnan(soft).any()
