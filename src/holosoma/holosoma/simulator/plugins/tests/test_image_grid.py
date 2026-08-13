"""Unit tests for tile_images (pure numpy/cv2; no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from holosoma.simulator.plugins.viz.image_grid import _BORDER, _draw_label, colorize_depth, tile_images

pytestmark = pytest.mark.no_sim

_B = 2 * _BORDER  # per-cell growth in each dimension from the white border


def test_single_image_unchanged_size():
    g = tile_images([np.zeros((10, 12, 3), np.uint8)])
    assert g.shape == (10 + _B, 12 + _B, 3)
    assert g.dtype == np.uint8


def test_two_equal_images_row():
    g = tile_images([np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)])
    assert g.shape == (8 + _B, 2 * (8 + _B), 3)  # auto near-square: cols=ceil(sqrt(2))=2 -> 1 row


def test_three_images_padded_grid():
    g = tile_images([np.zeros((8, 8, 3), np.uint8)] * 3)
    assert g.shape == (2 * (8 + _B), 2 * (8 + _B), 3)  # 2x2 grid, 1 padded cell


def test_mixed_sizes_resized_to_common_cell():
    imgs = [
        np.zeros((6, 10, 3), np.uint8),
        np.zeros((12, 4, 3), np.uint8),
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((5, 5, 3), np.uint8),
    ]
    g = tile_images(imgs)  # cell = max(h)=12, max(w)=10; 2x2 -> 2*(12+_B) x 2*(10+_B)
    assert g.shape == (2 * (12 + _B), 2 * (10 + _B), 3)


def test_explicit_layout():
    g = tile_images([np.zeros((8, 8, 3), np.uint8)] * 2, layout=(1, 3))
    assert g.shape == (8 + _B, 3 * (8 + _B), 3)


def test_letterbox_preserves_aspect_with_pad_bars():
    # A wide image (4x12) tiled with a tall one (12x4) -> 12x12 cells. The wide image keeps its
    # 1:3 aspect: scaled to 4x12 inside a 12x12 cell, leaving pad_value bars above and below.
    wide = np.full((4, 12, 3), 200, np.uint8)
    tall = np.full((12, 4, 3), 200, np.uint8)
    g = tile_images([wide, tall], layout=(1, 2), pad_value=0)
    cell = 12 + _B
    # First cell interior (drop the white border): the wide image fills the full 12 width but only
    # the middle 4 rows; the top rows are pad_value (0) letterbox bars, the centre band is the image.
    interior = g[_BORDER : cell - _BORDER, _BORDER : cell - _BORDER]  # 12x12 cell interior
    assert int(interior[0].max()) == 0  # top row is a letterbox bar (would be 200 if stretched)
    assert int(interior[6].max()) == 200  # centre band holds the (aspect-preserved) image


def test_pad_value_fills_trailing_cells():
    g = tile_images([np.full((4, 4, 3), 100, np.uint8)] * 3, layout=(2, 2), pad_value=7)
    # The 4th (padded) cell is bottom-right; its interior pixels (inside the border) equal pad_value.
    cell = 4 + _B
    interior = g[cell + _BORDER : 2 * cell - _BORDER, cell + _BORDER : 2 * cell - _BORDER]
    assert int(interior.min()) == 7 and int(interior.max()) == 7


def test_labels_none_matches_no_labels():
    imgs = [np.full((8, 8, 3), 50, np.uint8), np.full((8, 8, 3), 200, np.uint8)]
    assert np.array_equal(tile_images(imgs), tile_images(imgs, labels=None))


def test_wrong_length_labels_raises():
    with pytest.raises(ValueError, match="labels has 1 entries but there are 2 images"):
        tile_images([np.zeros((8, 8, 3), np.uint8)] * 2, labels=["only_one"])


def test_labels_draw_pixels():
    # Drawing a label must alter the cell (text is white-on-black over a mid-grey fill).
    img = np.full((40, 80, 3), 128, np.uint8)
    plain = tile_images([img.copy()])
    labeled = tile_images([img.copy()], labels=["cam0"])
    assert not np.array_equal(plain, labeled)


def test_short_label_stays_one_line():
    # A continuous label that fits draws on a single line (only the top band has text).
    cell = np.full((48, 240, 3), 0, np.uint8)
    _draw_label(cell, "env0/front")
    assert cell[4:18].max() > 200  # first line drawn
    assert cell[22:].max() == 0  # nothing wrapped onto a second line


def test_label_wraps_at_delimiter_when_too_wide():
    # Too wide for one line -> wraps at "/"/":" onto stacked lines (text in two bands).
    cell = np.full((64, 90, 3), 0, np.uint8)
    _draw_label(cell, "env0/front_camera:rgb")
    assert cell[4:18].max() > 200 and cell[20:40].max() > 200  # two lines present


def test_label_never_clips_right_edge():
    # An unbreakable over-long segment must shrink to fit, never clip past the right edge.
    cell = np.full((40, 60, 3), 0, np.uint8)
    _draw_label(cell, "env12/extremely_long_camera_name_xyz")
    assert not (cell[:, -2:] > 50).any()  # rightmost columns untouched by text


def test_colorize_depth_shape_dtype_and_flows_through_tile():
    # A [H,W,1] float depth -> HxWx3 uint8, usable as a tile_images panel like an RGB image.
    dep = np.full((12, 20, 1), 1.0, np.float32)
    out = colorize_depth(dep, (0.1, 5.0), "inferno")
    assert out.shape == (12, 20, 3) and out.dtype == np.uint8
    tile_images([out])  # must not raise (valid HxWx3 panel)


def test_colorize_depth_near_brighter_than_far_grayscale():
    # Grayscale: NEAR maps bright (255), FAR maps dark (0).
    near = colorize_depth(np.full((4, 4), 0.1, np.float32), (0.1, 5.0), "gray")
    far = colorize_depth(np.full((4, 4), 5.0, np.float32), (0.1, 5.0), "gray")
    assert int(near.mean()) > 240 and int(far.mean()) < 15


def test_colorize_depth_inf_no_hit_maps_to_far():
    # +inf / no-hit must read as the far plane (background), identical to an at-far pixel.
    inf = colorize_depth(np.full((4, 4), np.inf, np.float32), (0.1, 5.0), "gray")
    far = colorize_depth(np.full((4, 4), 5.0, np.float32), (0.1, 5.0), "gray")
    assert np.array_equal(inf, far)


def test_colorize_depth_clamps_out_of_range():
    # Beyond the range clamps (nearer-than-min -> brightest, farther-than-max -> darkest).
    nearer = colorize_depth(np.full((4, 4), 0.0, np.float32), (0.1, 5.0), "gray")
    farther = colorize_depth(np.full((4, 4), 99.0, np.float32), (0.1, 5.0), "gray")
    assert int(nearer.mean()) == 255 and int(farther.mean()) == 0


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        tile_images([])


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match="H.W.3"):
        tile_images([np.zeros((8, 8), np.uint8)])


def test_layout_too_small_raises():
    with pytest.raises(ValueError, match="cells"):
        tile_images([np.zeros((4, 4, 3), np.uint8)] * 5, layout=(2, 2))
