"""Unit tests for tile_images (pure numpy/cv2; no simulator)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.simulator.plugins.viz.image_grid import _BORDER, _draw_label, colorize_depth, tile_images

pytestmark = pytest.mark.no_sim

_B = 2 * _BORDER  # per-cell growth in each dimension from the white border


def test_single_image_unchanged_size() -> None:
    g = tile_images([np.zeros((10, 12, 3), np.uint8)])
    assert tuple(g.shape) == (10 + _B, 12 + _B, 3)
    assert g.dtype == np.uint8


def test_two_equal_images_row() -> None:
    g = tile_images([np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)])
    assert tuple(g.shape) == (8 + _B, 2 * (8 + _B), 3)  # auto near-square: cols=ceil(sqrt(2))=2 -> 1 row


def test_three_images_padded_grid() -> None:
    g = tile_images([np.zeros((8, 8, 3), np.uint8)] * 3)
    assert tuple(g.shape) == (2 * (8 + _B), 2 * (8 + _B), 3)  # 2x2 grid, 1 padded cell


def test_mixed_sizes_resized_to_common_cell() -> None:
    imgs: list[npt.NDArray[np.uint8]] = [
        np.zeros((6, 10, 3), np.uint8),
        np.zeros((12, 4, 3), np.uint8),
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((5, 5, 3), np.uint8),
    ]
    g = tile_images(imgs)  # cell = max(h)=12, max(w)=10; 2x2 -> 2*(12+_B) x 2*(10+_B)
    assert tuple(g.shape) == (2 * (12 + _B), 2 * (10 + _B), 3)


def test_explicit_layout() -> None:
    g = tile_images([np.zeros((8, 8, 3), np.uint8)] * 2, layout=(1, 3))
    assert tuple(g.shape) == (8 + _B, 3 * (8 + _B), 3)


def test_letterbox_preserves_aspect_with_pad_bars() -> None:
    # A wide image (4x12) tiled with a tall one (12x4) -> 12x12 cells. The wide image keeps its
    # 1:3 aspect: scaled to 4x12 inside a 12x12 cell, leaving pad_value bars above and below.
    wide: npt.NDArray[np.uint8] = np.full((4, 12, 3), 200, np.uint8)
    tall: npt.NDArray[np.uint8] = np.full((12, 4, 3), 200, np.uint8)
    g = tile_images([wide, tall], layout=(1, 2), pad_value=0)
    cell = 12 + _B
    # First cell interior (drop the white border): the wide image fills the full 12 width but only
    # the middle 4 rows; the top rows are pad_value (0) letterbox bars, the centre band is the image.
    interior = g[_BORDER : cell - _BORDER, _BORDER : cell - _BORDER]  # 12x12 cell interior
    assert int(interior[0].max()) == 0  # top row is a letterbox bar (would be 200 if stretched)
    assert int(interior[6].max()) == 200  # centre band holds the (aspect-preserved) image


def test_pad_value_fills_trailing_cells() -> None:
    g = tile_images([np.full((4, 4, 3), 100, np.uint8)] * 3, layout=(2, 2), pad_value=7)
    # The 4th (padded) cell is bottom-right; its interior pixels (inside the border) equal pad_value.
    cell = 4 + _B
    interior = g[cell + _BORDER : 2 * cell - _BORDER, cell + _BORDER : 2 * cell - _BORDER]
    assert int(interior.min()) == 7 and int(interior.max()) == 7


def test_wrong_length_labels_raises() -> None:
    with pytest.raises(ValueError, match="labels has 1 entries but there are 2 images"):
        tile_images([np.zeros((8, 8, 3), np.uint8)] * 2, labels=["only_one"])


def test_labels_draw_pixels() -> None:
    # Drawing a label must alter the cell (text is white-on-black over a mid-grey fill).
    img: npt.NDArray[np.uint8] = np.full((40, 80, 3), 128, np.uint8)
    plain = tile_images([img.copy()])
    labeled = tile_images([img.copy()], labels=["cam0"])
    assert not np.array_equal(plain, labeled)


def test_short_label_stays_one_line() -> None:
    # A continuous label that fits draws on a single line (only the top band has text).
    cell: npt.NDArray[np.uint8] = np.full((48, 240, 3), 0, np.uint8)
    _draw_label(cell, "env0/front")
    assert cell[4:18].max() > 200  # first line drawn
    assert cell[22:].max() == 0  # nothing wrapped onto a second line


def test_label_wraps_at_delimiter_when_too_wide() -> None:
    # Too wide for one line -> wraps at "/"/":" onto stacked lines (text in two bands).
    cell: npt.NDArray[np.uint8] = np.full((64, 90, 3), 0, np.uint8)
    _draw_label(cell, "env0/front_camera:rgb")
    assert cell[4:18].max() > 200 and cell[20:40].max() > 200  # two lines present


def test_label_never_clips_right_edge() -> None:
    # An unbreakable over-long segment must shrink to fit, never clip past the right edge.
    cell: npt.NDArray[np.uint8] = np.full((40, 60, 3), 0, np.uint8)
    _draw_label(cell, "env12/extremely_long_camera_name_xyz")
    assert not (cell[:, -2:] > 50).any()  # rightmost columns untouched by text


def test_colorize_depth_shape_dtype_and_flows_through_tile() -> None:
    # A [H,W,1] float depth -> HxWx3 uint8, usable as a tile_images panel like an RGB image.
    dep: npt.NDArray[np.float32] = np.full((12, 20, 1), 1.0, np.float32)
    out = colorize_depth(dep, (0.1, 5.0), "inferno")
    assert tuple(out.shape) == (12, 20, 3) and out.dtype == np.uint8
    tile_images([out])  # must not raise (valid HxWx3 panel)


def test_colorize_depth_near_brighter_than_far_grayscale() -> None:
    # Grayscale: NEAR maps bright (255), FAR maps dark (0).
    near = colorize_depth(np.full((4, 4), 0.1, np.float32), (0.1, 5.0), "gray")
    far = colorize_depth(np.full((4, 4), 5.0, np.float32), (0.1, 5.0), "gray")
    assert int(near.mean()) > 240 and int(far.mean()) < 15


def test_colorize_depth_clamps_out_of_range() -> None:
    # Beyond the range clamps (nearer-than-min -> brightest, farther-than-max -> darkest).
    nearer = colorize_depth(np.full((4, 4), 0.0, np.float32), (0.1, 5.0), "gray")
    farther = colorize_depth(np.full((4, 4), 99.0, np.float32), (0.1, 5.0), "gray")
    assert int(nearer.mean()) == 255 and int(farther.mean()) == 0


@pytest.mark.parametrize(("depth_range", "allowed_differing"), [((0.1, 5.0), 0), ((0.5, 100.0), 1)])
def test_colorize_depth_tracks_the_reference_ramp(depth_range: tuple[float, float], allowed_differing: int) -> None:
    # Compared against the ROUNDED reference, i.e. what the previous implementation actually
    # produced. An unrounded reference would carry 0.5 of quantization slack and let a systematic
    # brightness bias or a switch from rounding to truncation pass unnoticed.
    #
    # cv2.convertScaleAbs evaluates the ramp in a different float32 order (255*(hi-d)/(hi-lo) rather
    # than (1-(d-lo)/(hi-lo))*255), so a value within float32 epsilon of a half step can land on the
    # neighbouring level. Both libraries round half to even, so this is evaluation order, not a
    # rounding mode. ``allowed_differing`` records what was MEASURED here (0 and 1 of 20000); the
    # +8 added below is a cross-architecture allowance, since which side of a half step a
    # near-boundary sample lands on depends on SIMD dispatch (arm64 NEON vs x86 FMA).
    lo, hi = depth_range
    sweep: npt.NDArray[np.float32] = np.linspace(lo - 1.0, hi + 1.0, 20000, dtype=np.float32).reshape(100, 200)
    reference = ((1.0 - (np.clip(sweep, lo, hi) - lo) / (hi - lo)) * 255.0).round().astype(np.uint8)
    got = colorize_depth(sweep, depth_range, "gray")[..., 0]

    deviation = np.abs(got.astype(np.int16) - reference.astype(np.int16))
    assert deviation.max() <= 1, "the ramp may differ by at most one level"
    # A handful of samples sit within a couple of float32 ULPs of a half step, and which side they
    # land on depends on the SIMD dispatch (arm64 NEON vs x86 FMA), so this is a small budget
    # rather than the count measured on one machine. It stays far below what a real defect costs:
    # truncation instead of rounding, or a sub-level brightness bias, moves thousands of the 20000.
    assert int(np.count_nonzero(deviation)) <= allowed_differing + 8
    # Monotone and spanning the full range: near stays brightest, far darkest.
    assert got[0, 0] == 255 and got[-1, -1] == 0
    assert np.all(np.diff(got.ravel().astype(np.int16)) <= 0)


@pytest.mark.parametrize("channel_axis", [False, True], ids=["hw", "hw1"])
def test_colorize_depth_leaves_the_caller_array_untouched(channel_axis: bool) -> None:
    # The input is the per-step host snapshot SHARED with every other consumer of this stream, so
    # the normalization must stay on colorize_depth's own copy. It normalizes in place internally.
    # Both shapes are covered because [H,W,1] is what get_camera_data actually hands out and it
    # reaches the squeeze path; an [H,W] float32 input aliases the caller just as directly, since
    # np.asarray returns it unchanged, so either shape catches an in-place step.
    dep: npt.NDArray[np.float32] = np.array([[0.5, np.inf], [np.nan, 7.0]], np.float32)
    if channel_axis:
        dep = dep[..., None]
    before = dep.copy()
    colorize_depth(dep, (0.1, 5.0), "gray")
    assert np.array_equal(dep, before, equal_nan=True)


def test_colorize_depth_colormapped_output_is_the_ramp_byte_in_rgb() -> None:
    # Every other case here uses "gray", which skips both the colormap lookup and the BGR->RGB
    # reorder. This pins the colorized path end to end: the mid-range depth must produce exactly
    # cv2's inferno entry for its ramp byte, in RGB order.
    import cv2

    lo, hi = 0.0, 4.0
    depth: npt.NDArray[np.float32] = np.full((2, 3), 1.0, np.float32)
    expected_gray = round((1.0 - (1.0 - lo) / (hi - lo)) * 255.0)  # 191
    bgr = cv2.applyColorMap(np.full((1, 1), expected_gray, np.uint8), cv2.COLORMAP_INFERNO)
    expected_rgb = bgr[0, 0][::-1]

    out = colorize_depth(depth, (lo, hi), "inferno")
    assert out.shape == (2, 3, 3) and out.dtype == np.uint8
    assert np.array_equal(out, np.broadcast_to(expected_rgb, (2, 3, 3)))


@pytest.mark.parametrize("value", [np.nan, -np.inf, np.inf], ids=["nan", "neg_inf", "pos_inf"])
def test_colorize_depth_maps_every_non_finite_to_the_far_plane(value: float) -> None:
    # Asserted against the literal far byte rather than colorize_depth(far), which would compare
    # production against production. 0 is "far" because the ramp is inverted (near -> 255).
    #
    # Honest about what this pins: only [neg_inf] fails if the isfinite pass in depth_color is
    # removed. [pos_inf] is absorbed by the far clamp either way, and [nan] reaches 0 only because
    # cv2's saturate_cast of NaN happens to give 0 here -- float-to-uchar conversion of a
    # non-finite value is unspecified, and was measured to depend on SIMD width on this arch. Those
    # two are a contract record, not a guard; the isfinite pass is defensive for that same reason.
    got = colorize_depth(np.full((4, 4), value, np.float32), (0.1, 5.0), "gray")
    assert np.array_equal(got, np.zeros((4, 4, 3), np.uint8))


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        tile_images([])


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="H.W.3"):
        tile_images([np.zeros((8, 8), np.uint8)])


def test_layout_too_small_raises() -> None:
    with pytest.raises(ValueError, match="cells"):
        tile_images([np.zeros((4, 4, 3), np.uint8)] * 5, layout=(2, 2))
