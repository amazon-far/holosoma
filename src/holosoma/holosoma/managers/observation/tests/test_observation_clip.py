"""Global observation clipping: applied per term, unless the term sets ``global_clip=False``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_HERE = "holosoma.managers.observation.tests.test_observation_clip"


def flat(env: Any) -> torch.Tensor:
    return env.flat


def image(env: Any) -> torch.Tensor:
    return env.image


def _step(groups: dict[str, ObsGroupCfg], **values: torch.Tensor) -> dict[str, Any]:
    env = SimpleNamespace(num_envs=1, device="cpu", **values)
    manager = ObservationManager(ObservationManagerCfg(groups=groups, clip_observations=100.0), env, "cpu")
    manager.new_frame()
    return manager.step()


def test_global_clip_applies_to_flagged_terms_only() -> None:
    value = torch.tensor([[500.0, -500.0, 1.0]])
    terms = {
        "clipped": ObsTermCfg(func=f"{_HERE}:flat"),
        "unclipped": ObsTermCfg(func=f"{_HERE}:flat", global_clip=False),
    }

    obs = _step({"g": ObsGroupCfg(concatenate=False, terms=terms)}, flat=value)["g"]

    assert obs["clipped"].tolist() == [[100.0, -100.0, 1.0]]
    assert torch.equal(obs["unclipped"], value)


def test_flattened_depth_keeps_its_sentinel_when_opted_out() -> None:
    """A flattened image in a concatenated group is 2-D float, but opting out keeps +inf and far depths."""
    depth = torch.tensor([[float("inf"), 250.0, 1.0]])
    terms = {"depth": ObsTermCfg(func=f"{_HERE}:flat", global_clip=False)}

    obs = _step({"g": ObsGroupCfg(terms=terms)}, flat=depth)["g"]

    assert torch.isinf(obs[0, 0]) and float(obs[0, 1]) == 250.0


def test_opted_out_uint8_image_is_untouched() -> None:
    rgb = torch.full((1, 4, 4, 3), 200, dtype=torch.uint8)
    terms = {"rgb": ObsTermCfg(func=f"{_HERE}:image", global_clip=False)}

    out = _step({"g": ObsGroupCfg(concatenate=False, terms=terms)}, image=rgb)["g"]["rgb"]

    assert out.dtype == torch.uint8 and torch.equal(out, rgb)


@pytest.mark.parametrize(
    "value",
    [
        torch.full((1, 4, 4, 3), 200, dtype=torch.uint8),  # RGB image
        torch.full((1, 4, 4, 1), float("inf")),  # depth image
        torch.tensor([[True, False]]),  # contact flags
    ],
    ids=["uint8_image", "float_image", "bool"],
)
def test_global_clip_on_a_term_that_is_not_2d_float_raises(value: torch.Tensor) -> None:
    terms = {"term": ObsTermCfg(func=f"{_HERE}:image")}

    with pytest.raises(ValueError, match="'term' in group 'g': .*Set global_clip=False"):
        _step({"g": ObsGroupCfg(concatenate=False, terms=terms)}, image=value)


def test_global_clip_follows_scale_and_term_clip() -> None:
    """Order: scale, then the term's clip, then the global limit."""
    terms = {"x": ObsTermCfg(func=f"{_HERE}:flat", scale=10.0, clip=(-50.0, 50.0))}

    obs = _step({"g": ObsGroupCfg(terms=terms)}, flat=torch.tensor([[20.0, 2.0]]))["g"]

    assert obs.tolist() == [[50.0, 20.0]]


def test_layout_reports_global_clip() -> None:
    terms = {"a": ObsTermCfg(func=f"{_HERE}:flat"), "b": ObsTermCfg(func=f"{_HERE}:flat", global_clip=False)}
    env = SimpleNamespace(num_envs=1, device="cpu", flat=torch.zeros(1, 2))
    manager = ObservationManager(ObservationManagerCfg(groups={"g": ObsGroupCfg(terms=terms)}), env, "cpu")

    assert [entry["global_clip"] for entry in manager.get_observation_layout(["g"])] == [True, False]
