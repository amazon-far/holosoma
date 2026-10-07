"""Numeric-parity tests for DepthShmPlugin's depth preprocessing (pure, no simulator).

The plugin's crop/clip/resize/normalize chain is a *contract with the training pipeline*
(``holosoma.managers.observation.terms.depth.WarpDepthImageObsTerm``). Every step must match, and
none of the plausible mistakes raise: a wrong ``far_clip`` rescales every pixel, a missing crop
changes the field of view, and resizing before clamping smears the far-plane sentinel across
neighbouring pixels. All of those yield a plausible-looking but wrong depth latent, so this module
reimplements the training math independently and asserts the plugin agrees bit-for-bit.

The stochastic augmentations in training's ``_process_depth_images`` (Gaussian noise, Sobel-edge
shuffling, Perlin holes, per-env depth offset) are deliberately excluded: they are train-time domain
randomization, not part of the deployed input transform.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.config_types.plugin import DepthShmPluginConfig
from holosoma.simulator.plugins.depth_shm_plugin import DepthShmPlugin
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

# The D435i training rig: 106x60 raycast, [0.3, 3.0] m range, crop [2:, 4:-4] -> 58x98 -> 58x87.
RENDER_H, RENDER_W = 60, 106
OUT_H, OUT_W = 58, 87
NEAR, FAR = 0.3, 3.0


def _d435i_config(**overrides: Any) -> DepthShmPluginConfig:
    kwargs: dict[str, Any] = {
        "camera": "cam",
        "resized_height": OUT_H,
        "resized_width": OUT_W,
        "crop_top": 2,
        "crop_left": 4,
        "crop_right": 4,
        "near_clip": NEAR,
        "far_clip": FAR,
    }
    kwargs.update(overrides)
    return DepthShmPluginConfig(**kwargs)


def _plugin(cfg: DepthShmPluginConfig) -> DepthShmPlugin:
    """A plugin instance with only ``cfg`` populated.

    ``__init__`` needs a live simulator to resolve camera streams; the preprocessing under test is a
    pure function of ``cfg``, so bypass construction rather than standing up a fake simulator.
    """
    plugin = object.__new__(DepthShmPlugin)
    plugin.cfg = cfg
    return plugin


def _training_reference(depth: npt.NDArray[np.float32], cfg: DepthShmPluginConfig) -> npt.NDArray[np.float32]:
    """Training's deployed depth transform, transcribed from ``WarpDepthImageObsTerm``.

    Mirrors, in order: the capture-time clamp to the camera's ``[min_range, max_range]``
    (``_get_depth_images``), ``_crop_depth_images``'s ``[:, 2:, 4:-4]``, the bicubic
    ``F.interpolate(..., antialias=True)`` to the backbone's size, the far-side clamp plus the
    ``< 0.15 -> max_depth`` empty rule, and ``_normalize_depth_images``.
    """
    near, far = cfg.near_clip, cfg.far_clip
    t = torch.from_numpy(depth.astype(np.float32).copy())[None]  # (N, H, W)
    t = torch.clamp(t, min=near, max=far)
    t = t[:, cfg.crop_top : t.shape[1] - cfg.crop_bottom, cfg.crop_left : t.shape[2] - cfg.crop_right]
    t = torch.nn.functional.interpolate(
        t.unsqueeze(1),
        size=(cfg.resized_height, cfg.resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(1)
    t = torch.clamp(t, max=far)
    t[t < cfg.empty_threshold] = far
    return cast("npt.NDArray[np.float32]", ((t - near) / (far - near) - 0.5).numpy()[0])


def _synthetic_depth(seed: int) -> npt.NDArray[np.float32]:
    """A metric-depth frame with in-range values, a far-plane sentinel and a no-hit patch."""
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.05, 4.0, size=(RENDER_H, RENDER_W)).astype(np.float32)
    depth[10:20, 30:40] = 39.0  # MuJoCo reports the scene extent where a ray escapes
    depth[40:50, 60:80] = np.inf  # raycast no-hit
    # The plugin's caller (``publish``) does this substitution before preprocessing.
    return np.nan_to_num(depth, nan=FAR, posinf=FAR, neginf=NEAR)


@pytest.mark.parametrize("seed", range(4))
def test_matches_training_pipeline(seed: int) -> None:
    """The plugin's output must equal the training transform to floating-point precision."""
    cfg = _d435i_config()
    depth = _synthetic_depth(seed)

    actual = _plugin(cfg)._resize_clip_normalize(depth.copy())

    assert actual.shape == (OUT_H, OUT_W)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, _training_reference(depth, cfg), atol=1e-6)


def test_crop_precedes_resize() -> None:
    """Cropping after the resize would sample a different field of view.

    Bright-line check: put a distinctive value in a row the crop removes. If the crop ran after the
    resize (or not at all), that row's depth would bleed into the output.
    """
    cfg = _d435i_config()
    depth = np.full((RENDER_H, RENDER_W), FAR, dtype=np.float32)
    depth[0:2, :] = NEAR  # exactly the rows crop_top drops

    actual = _plugin(cfg)._resize_clip_normalize(depth)

    # Everything surviving the crop is at far -> +0.5 uniformly.
    np.testing.assert_allclose(actual, 0.5, atol=1e-6)


def test_far_sentinel_is_clamped_before_resize() -> None:
    """A far-plane sentinel must not smear into its neighbours through the resize kernel."""
    cfg = _d435i_config()
    depth = np.full((RENDER_H, RENDER_W), FAR, dtype=np.float32)
    depth[30, 50] = 1e4  # a single escaped ray

    actual = _plugin(cfg)._resize_clip_normalize(depth)

    # Clamped first, so the frame is uniformly far; an unclamped 1e4 would ring far past +0.5.
    np.testing.assert_allclose(actual, 0.5, atol=1e-6)
    assert actual.max() <= 0.5 + 1e-6


def test_normalizes_range_endpoints() -> None:
    """near_clip maps to -0.5 and far_clip to +0.5 — the backbone's expected input scale."""
    cfg = _d435i_config()
    plugin = _plugin(cfg)

    at_near = plugin._resize_clip_normalize(np.full((RENDER_H, RENDER_W), NEAR, dtype=np.float32))
    at_far = plugin._resize_clip_normalize(np.full((RENDER_H, RENDER_W), FAR, dtype=np.float32))

    np.testing.assert_allclose(at_near, -0.5, atol=1e-6)
    np.testing.assert_allclose(at_far, 0.5, atol=1e-6)


def test_far_clip_rescales_every_pixel() -> None:
    """Guards the D435i-vs-ZED far_clip distinction (3.0 vs 2.0), which fails silently."""
    depth = np.full((RENDER_H, RENDER_W), 1.5, dtype=np.float32)

    at_3m = _plugin(_d435i_config(far_clip=3.0))._resize_clip_normalize(depth.copy())
    at_2m = _plugin(_d435i_config(far_clip=2.0))._resize_clip_normalize(depth.copy())

    # 1.5m sits mid-range at far=3.0 but near the far end at far=2.0.
    np.testing.assert_allclose(at_3m, (1.5 - NEAR) / (3.0 - NEAR) - 0.5, atol=1e-6)
    np.testing.assert_allclose(at_2m, (1.5 - NEAR) / (2.0 - NEAR) - 0.5, atol=1e-6)
    assert not np.allclose(at_3m, at_2m)


def test_no_crop_config_is_a_passthrough() -> None:
    """The ZED preset configures no crop, so only the resize should change the frame."""
    cfg = _d435i_config(crop_top=0, crop_left=0, crop_right=0)
    depth = _synthetic_depth(0)

    plugin = _plugin(cfg)
    np.testing.assert_array_equal(plugin._crop(depth), depth)
    np.testing.assert_allclose(plugin._resize_clip_normalize(depth.copy()), _training_reference(depth, cfg), atol=1e-6)


def test_crop_that_removes_everything_fails_loud() -> None:
    """A crop wider than the frame should raise, not silently yield an empty tensor."""
    cfg = _d435i_config(crop_top=RENDER_H)
    with pytest.raises(ValueError, match="crop removed the whole"):
        _plugin(cfg)._resize_clip_normalize(np.zeros((RENDER_H, RENDER_W), dtype=np.float32))
