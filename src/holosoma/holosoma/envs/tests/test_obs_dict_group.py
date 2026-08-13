"""Tests for dict/dtype-aware observation handling in BaseTask.

A ``concatenate=False`` observation group computes to a dict of per-term tensors. These tests pin
that ``_clip_observations`` and ``_store_final_observations`` handle a dict group value (and a
uint8 image term) without flattening or clipping it, by calling the unbound methods on a stub.
"""

from __future__ import annotations

import types

import pytest

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _make_stub(obs_buf_dict, clip_limit=100.0):
    """Minimal object carrying just what the two methods touch."""
    stub = types.SimpleNamespace()
    stub.obs_buf_dict = obs_buf_dict
    stub.extras = {}
    stub.observation_manager = types.SimpleNamespace(cfg=types.SimpleNamespace(clip_observations=clip_limit))
    return stub


def test_clip_observations_handles_dict_group_and_uint8():
    """A flat float group is clipped; a dict group recurses; uint8 images pass through unclipped."""
    flat = torch.tensor([[500.0, -500.0, 1.0]])  # exceeds +/-100, gets clipped
    image = torch.full((1, 4, 4, 3), 200, dtype=torch.uint8)  # uint8 image, stays unclipped
    proprio = torch.tensor([[5.0, -5.0]])  # within range, unchanged
    obs = {
        "actor_obs": flat,
        "image_group": {"front_rgb": image, "proprio": proprio},
    }
    stub = _make_stub(obs)

    BaseTask._clip_observations(stub)

    # Flat float group clipped to +/-100.
    assert torch.equal(stub.obs_buf_dict["actor_obs"], torch.tensor([[100.0, -100.0, 1.0]]))
    # Dict group preserved as a dict.
    assert isinstance(stub.obs_buf_dict["image_group"], dict)
    # uint8 image untouched (not clipped to +/-100).
    assert torch.equal(stub.obs_buf_dict["image_group"]["front_rgb"], image)
    assert stub.obs_buf_dict["image_group"]["front_rgb"].dtype == torch.uint8
    # Float term inside the dict group within range, unchanged.
    assert torch.equal(stub.obs_buf_dict["image_group"]["proprio"], proprio)


def test_clip_observations_preserves_depth_inf_sentinel():
    """A float32 depth image (4-D) keeps its +inf no-hit sentinel; clipping would corrupt it."""
    depth = torch.full((1, 4, 4, 1), 250.0)  # finite far depth, beyond the +/-100 clip
    depth[0, 0, 0, 0] = float("inf")  # no-hit sentinel
    obs = {"image_group": {"front_depth": depth.clone()}}
    stub = _make_stub(obs, clip_limit=100.0)

    BaseTask._clip_observations(stub)

    out = stub.obs_buf_dict["image_group"]["front_depth"]
    assert torch.isinf(out[0, 0, 0, 0])  # sentinel survives (not clamped to 100)
    assert float(out[0, 1, 1, 0]) == 250.0  # far depth not clamped


def test_store_final_observations_handles_dict_group():
    """Final-obs copy works for both a flat group and a dict (image) group, per env."""
    num_envs = 3
    cur = {
        "actor_obs": torch.zeros(num_envs, 2),
        "image_group": {"front_rgb": torch.zeros(num_envs, 2, 2, 3, dtype=torch.uint8)},
    }
    stub = _make_stub(cur)

    final = {
        "actor_obs": torch.arange(num_envs * 2, dtype=torch.float32).reshape(num_envs, 2),
        "image_group": {"front_rgb": torch.full((num_envs, 2, 2, 3), 7, dtype=torch.uint8)},
    }
    env_ids = torch.tensor([0, 2])  # store envs 0 and 2 only

    BaseTask._store_final_observations(stub, env_ids, final)

    store = stub.extras["final_observations"]
    # Flat group: env 0 and 2 copied from final, env 1 left zero.
    assert torch.equal(store["actor_obs"][env_ids], final["actor_obs"][env_ids])
    assert torch.equal(store["actor_obs"][1], torch.zeros(2))
    # Dict group recursed: image copied for the selected envs, dtype preserved.
    img_store = store["image_group"]["front_rgb"]
    assert img_store.dtype == torch.uint8
    assert torch.equal(img_store[env_ids], final["image_group"]["front_rgb"][env_ids])
    assert int(img_store[1].sum()) == 0  # env 1 untouched


def test_store_final_observations_empty_is_noop():
    """An empty final-obs dict stores nothing."""
    stub = _make_stub({"actor_obs": torch.zeros(2, 2)})
    BaseTask._store_final_observations(stub, torch.tensor([0]), {})
    assert "final_observations" not in stub.extras
