"""ObservationManager.clip_observations: flat float groups are clipped, image terms are not."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _manager(clip_limit: float = 100.0) -> ObservationManager:
    return ObservationManager(ObservationManagerCfg(clip_observations=clip_limit), SimpleNamespace(num_envs=1), "cpu")


def test_clip_observations_handles_dict_group_and_uint8() -> None:
    """A flat float group is clipped; a dict group recurses; uint8 images pass through unclipped."""
    flat = torch.tensor([[500.0, -500.0, 1.0]])  # exceeds +/-100, gets clipped
    image = torch.full((1, 4, 4, 3), 200, dtype=torch.uint8)  # uint8 image, stays unclipped
    proprio = torch.tensor([[5.0, -5.0]])  # within range, unchanged
    obs = {
        "actor_obs": flat,
        "image_group": {"front_rgb": image, "proprio": proprio},
    }

    out = _manager().clip_observations(obs)

    # Flat float group clipped to +/-100.
    assert torch.equal(out["actor_obs"], torch.tensor([[100.0, -100.0, 1.0]]))
    # Dict group preserved as a dict.
    assert isinstance(out["image_group"], dict)
    # uint8 image untouched (not clipped to +/-100).
    assert torch.equal(out["image_group"]["front_rgb"], image)
    assert out["image_group"]["front_rgb"].dtype == torch.uint8
    # Float term inside the dict group within range, unchanged.
    assert torch.equal(out["image_group"]["proprio"], proprio)


def test_clip_observations_preserves_depth_inf_sentinel() -> None:
    """A float32 depth image (4-D) keeps its +inf no-hit sentinel; clipping would corrupt it."""
    depth = torch.full((1, 4, 4, 1), 250.0)  # finite far depth, beyond the +/-100 clip
    depth[0, 0, 0, 0] = float("inf")  # no-hit sentinel

    out = _manager().clip_observations({"image_group": {"front_depth": depth.clone()}})["image_group"]["front_depth"]

    assert torch.isinf(out[0, 0, 0, 0])  # sentinel survives (not clamped to 100)
    assert float(out[0, 1, 1, 0]) == 250.0  # far depth not clamped
