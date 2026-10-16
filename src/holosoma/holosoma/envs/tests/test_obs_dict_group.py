"""Tests for dict/dtype-aware observation handling in BaseTask.

A ``concatenate=False`` observation group computes to a dict of per-term tensors. These tests pin
that ``_store_final_observations`` handles a dict group value (and a uint8 image term) without
flattening it, by calling the unbound method on a stub.
"""

from __future__ import annotations

import types
from typing import cast

import pytest

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _make_stub(obs_buf_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> BaseTask:
    """Minimal object carrying just what the method under test touches."""
    stub = types.SimpleNamespace()
    stub.obs_buf_dict = obs_buf_dict
    stub.extras = {}
    return cast("BaseTask", stub)


def test_store_final_observations_handles_dict_group() -> None:
    """Final-obs copy works for both a flat group and a dict (image) group, per env."""
    num_envs = 3
    cur = {
        "actor_obs": torch.zeros(num_envs, 2),
        "image_group": {"front_rgb": torch.zeros(num_envs, 2, 2, 3, dtype=torch.uint8)},
    }
    stub = _make_stub(cur)

    env_ids = torch.tensor([0, 2])  # store envs 0 and 2 only
    # Final observations hold one row per entry of env_ids.
    final = {
        "actor_obs": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "image_group": {"front_rgb": torch.full((2, 2, 2, 3), 7, dtype=torch.uint8)},
    }

    BaseTask._store_final_observations(stub, env_ids, final)

    store = stub.extras["final_observations"]
    # Flat group: env 0 and 2 copied from final, env 1 left zero.
    assert torch.equal(store["actor_obs"][env_ids], final["actor_obs"])
    assert torch.equal(store["actor_obs"][1], torch.zeros(2))
    # Dict group recursed: image copied for the selected envs, dtype preserved.
    img_store = store["image_group"]["front_rgb"]
    assert img_store.dtype == torch.uint8
    assert torch.equal(img_store[env_ids], final["image_group"]["front_rgb"])
    assert int(img_store[1].sum()) == 0  # env 1 untouched


def test_store_final_observations_empty_is_noop() -> None:
    """An empty final-obs dict stores nothing."""
    stub = _make_stub({"actor_obs": torch.zeros(2, 2)})
    BaseTask._store_final_observations(stub, torch.tensor([0]), {})
    assert "final_observations" not in stub.extras
