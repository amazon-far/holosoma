"""``setup_raycast_depth_camera_mount``: per-env camera mount offsets drawn through the term sampler."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.managers.randomization.terms.sensors import setup_raycast_depth_camera_mount
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sampler import STAGE_SETUP, TermSampler

pytestmark = pytest.mark.no_sim

NUM_ENVS = 64
POSITION = [[-0.025, 0.025], [-0.01, 0.01], [0.0, 0.05]]
EULER_DEG = [[-2.5, 2.5], [-3.0, 3.0], [-1.0, 1.0]]


class _Camera:
    def __init__(self) -> None:
        self.offsets: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_mount_offset(self, position: torch.Tensor, euler: torch.Tensor) -> None:
        self.offsets = (position, euler)


def _env(camera: Any) -> Any:
    simulator = SimpleNamespace(raycast_cameras={"head_depth": camera})
    return SimpleNamespace(num_envs=NUM_ENVS, device="cpu", simulator=simulator)


def _apply(camera: Any, seed: int = 7, **overrides: Any) -> None:
    params: dict[str, Any] = {
        "sensor": "head_depth",
        "position_range": POSITION,
        "euler_deg_range": EULER_DEG,
        "enabled": True,
    }
    params.update(overrides)
    setup_raycast_depth_camera_mount(
        _env(camera), TermSampler.bind(seed, "setup_raycast_depth_camera_mount", STAGE_SETUP), **params
    )


def test_offsets_are_per_env_within_each_axis_range() -> None:
    camera = _Camera()
    _apply(camera)

    assert camera.offsets is not None
    position, euler = camera.offsets
    assert position.shape == euler.shape == (NUM_ENVS, 3)
    for axis, (lo, hi) in enumerate(POSITION):
        assert torch.all((position[:, axis] >= lo) & (position[:, axis] <= hi))
    for axis, (lo, hi) in enumerate(EULER_DEG):
        lo_rad, hi_rad = torch.deg2rad(torch.tensor([lo, hi]))
        assert torch.all((euler[:, axis] >= lo_rad - 1e-7) & (euler[:, axis] <= hi_rad + 1e-7))
    assert position[:, 0].unique().numel() == NUM_ENVS
    assert not torch.allclose(position[:, 0] / 0.025, euler[:, 0] / torch.deg2rad(torch.tensor(2.5)))


def test_offsets_are_reproducible_for_a_seed() -> None:
    first, second, other = _Camera(), _Camera(), _Camera()
    _apply(first)
    _apply(second)
    _apply(other, seed=8)

    assert first.offsets is not None and second.offsets is not None and other.offsets is not None
    assert all(torch.equal(a, b) for a, b in zip(first.offsets, second.offsets, strict=True))
    assert not torch.equal(first.offsets[0], other.offsets[0])


def test_disabled_term_leaves_the_mount_alone() -> None:
    camera = _Camera()
    _apply(camera, enabled=False)

    assert camera.offsets is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sensor": "missing"}, "no ray-cast depth camera named 'missing'"),
        ({"position_range": POSITION[:2]}, "per axis"),
    ],
)
def test_bad_term_configs_are_rejected(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _apply(_Camera(), **overrides)
