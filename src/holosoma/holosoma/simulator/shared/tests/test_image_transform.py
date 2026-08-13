"""Unit tests for the camera image-transform helper (pure torch, no simulator).

Pins the fixed transform order (resize -> layout -> scale) and the frame semantics a
visual policy depends on: HWC<->CHW layout, uint8/float[0,1]/float[-1,1] scaling, depth range
normalization with the +inf no-hit mapped to the far end.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor import ImageTransformConfig
from holosoma.simulator.shared.image_transform import apply_image_transform
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _rgb(n=2, h=8, w=6):
    return torch.randint(0, 256, (n, h, w, 3), dtype=torch.uint8)


def test_default_is_identity_passthrough():
    rgb = _rgb()
    out = apply_image_transform(rgb, ImageTransformConfig(), "rgb")
    assert out.dtype == torch.uint8 and tuple(out.shape) == (2, 8, 6, 3)
    assert torch.equal(out, rgb)


def test_chw_layout_permutes():
    out = apply_image_transform(_rgb(), ImageTransformConfig(layout="CHW"), "rgb")
    assert tuple(out.shape) == (2, 3, 8, 6)  # [N, C, H, W]


def test_flatten_produces_1d_per_env():
    out = apply_image_transform(_rgb(n=2, h=8, w=6), ImageTransformConfig(layout="CHW", flatten=True), "rgb")
    assert tuple(out.shape) == (2, 3 * 8 * 6)  # [N, C*H*W]


def test_flatten_roundtrips_through_cnnwrapper_view():
    # The flat vector must be CHW-ordered so CNNWrapper's view(N, C, H, W) reconstructs the image.
    rgb = _rgb(n=2, h=8, w=6)
    chw = apply_image_transform(rgb, ImageTransformConfig(layout="CHW"), "rgb")  # [N, C, H, W]
    flat = apply_image_transform(rgb, ImageTransformConfig(layout="CHW", flatten=True), "rgb")
    reconstructed = flat.view(2, 3, 8, 6)  # exactly what CNNWrapper.forward does
    assert torch.equal(reconstructed, chw)


def test_flatten_requires_chw_layout():
    with pytest.raises(ValueError, match="flatten requires layout='CHW'"):
        ImageTransformConfig(layout="HWC", flatten=True)


def test_resize_rgb_bilinear_keeps_uint8():
    out = apply_image_transform(_rgb(), ImageTransformConfig(resize=[4, 4]), "rgb")
    assert tuple(out.shape) == (2, 4, 4, 3) and out.dtype == torch.uint8


def test_resize_nonsquare_target_and_axis_order():
    # Non-square target [H=4, W=8] from a non-square input [6, 10]: a (W, H) axis swap in the
    # interpolate call would yield (2, 8, 4, 3) and fail. A vertical brightness ramp (rows differ,
    # columns constant) also catches a transpose by VALUE: after resize the gradient must stay
    # along rows (row 0 darkest, last row brightest), not along columns.
    ramp = torch.zeros((2, 6, 10, 3), dtype=torch.uint8)
    for r in range(6):
        ramp[:, r, :, :] = r * 40  # brightness increases down the rows, constant across columns
    out = apply_image_transform(ramp, ImageTransformConfig(resize=[4, 8]), "rgb")
    assert tuple(out.shape) == (2, 4, 8, 3)
    col_mean = out[0, :, :, 0].float().mean(dim=1)  # mean per row
    assert torch.all(col_mean[1:] >= col_mean[:-1])  # ramp preserved along rows (no transpose)
    assert float(out[0, :, 0, 0].float().std()) > 1.0  # rows vary
    assert float(out[0, 0, :, 0].float().std()) < 1e-3  # columns constant within a row


def test_float01_scaling_range():
    rgb = torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
    out = apply_image_transform(rgb, ImageTransformConfig(scale="float01"), "rgb")
    assert out.dtype == torch.float32 and float(out.max()) == 1.0 and float(out.min()) == 1.0


def test_float_pm1_scaling_range():
    rgb = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
    out = apply_image_transform(rgb, ImageTransformConfig(scale="float_pm1"), "rgb")
    assert float(out.min()) == -1.0  # 0 -> -1


def test_full_chain_chw_resize_float01():
    out = apply_image_transform(_rgb(), ImageTransformConfig(resize=[4, 4], layout="CHW", scale="float01"), "rgb")
    assert tuple(out.shape) == (2, 3, 4, 4) and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_depth_float01_maps_range_and_inf_to_far():
    depth = torch.full((1, 4, 4, 1), 2.0)
    depth[0, 0, 0, 0] = float("inf")  # no-hit
    out = apply_image_transform(depth, ImageTransformConfig(scale="float01", depth_range=[0.5, 5.0]), "depth")
    assert float(out[0, 0, 0, 0]) == 1.0  # +inf -> far end
    assert abs(float(out[0, 1, 1, 0]) - (2.0 - 0.5) / 4.5) < 1e-5  # mid distance normalized


def test_depth_float_pm1_maps_range_and_inf_to_plus1():
    depth = torch.full((1, 4, 4, 1), 0.5)  # at the near end of [0.5, 5.0] -> -1
    depth[0, 0, 0, 0] = float("inf")  # no-hit -> far end -> +1
    out = apply_image_transform(depth, ImageTransformConfig(scale="float_pm1", depth_range=[0.5, 5.0]), "depth")
    assert abs(float(out[0, 1, 1, 0]) - (-1.0)) < 1e-5  # near distance -> -1
    assert abs(float(out[0, 0, 0, 0]) - 1.0) < 1e-5  # +inf -> far end -> +1


def test_depth_resize_nearest_preserves_inf():
    depth = torch.full((1, 8, 8, 1), 3.0)
    depth[0, 0, 0, 0] = float("inf")
    out = apply_image_transform(depth, ImageTransformConfig(resize=[4, 4]), "depth")
    assert tuple(out.shape) == (1, 4, 4, 1)
    assert torch.isinf(out).any()  # nearest keeps the no-hit sentinel (bilinear would blend it away)


def test_depth_float_without_range_raises():
    with pytest.raises(ValueError, match="depth_range is required"):
        apply_image_transform(torch.zeros((1, 2, 2, 1)), ImageTransformConfig(scale="float01"), "depth")


def test_config_rejects_bad_resize_and_range():
    with pytest.raises(ValueError, match="resize must be positive"):
        ImageTransformConfig(resize=[0, 4])
    with pytest.raises(ValueError, match="depth_range must be"):
        ImageTransformConfig(depth_range=[5.0, 1.0])
