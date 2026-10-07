from __future__ import annotations

import pytest

from holosoma.managers.observation.terms.depth import WarpDepthImageObsTerm
from holosoma.sensors.warp.camera_config.d435i_depth_config import G1FlatRsD435iConfig
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_d435i_camera_owns_its_training_crop() -> None:
    config = G1FlatRsD435iConfig()

    assert (
        config.crop_top,
        config.crop_bottom,
        config.crop_left,
        config.crop_right,
    ) == (2, 0, 4, 4)


def test_depth_term_applies_configured_crop() -> None:
    term = object.__new__(WarpDepthImageObsTerm)
    term._crop = (1, 2, 3, 4)
    image = torch.arange(2 * 8 * 10).reshape(2, 8, 10)

    cropped = term._crop_depth_images(image)

    torch.testing.assert_close(cropped, image[:, 1:-2, 3:-4])


def test_zero_crop_keeps_full_image() -> None:
    term = object.__new__(WarpDepthImageObsTerm)
    term._crop = (0, 0, 0, 0)
    image = torch.arange(12).reshape(1, 3, 4)

    torch.testing.assert_close(term._crop_depth_images(image), image)
