"""Tile a flat list of images into a single grid image (pure numpy + cv2).

Used by the viz egress to show several mounted-camera views in one window/frame. It tiles whatever
flat list it is handed, in order; the caller decides which cameras and in what order. RGB in, RGB
out (the display/encode boundary converts to BGR). Lives under the viz package since that is its
only consumer. Depth colorization is shared with the ROS2 egress in
``holosoma.simulator.plugins.depth_color``; :func:`colorize_depth` is re-exported here for callers that
still import it from this module.
"""

from __future__ import annotations

import math
import re

import cv2
import numpy as np

from holosoma.simulator.plugins.depth_color import colorize_depth

__all__ = ["colorize_depth", "tile_images"]

_BORDER = 2  # white gutter (px) added around every cell, so adjacent views are visibly separated
_BORDER_COLOR = (255, 255, 255)  # white; symmetric in RGB/BGR so the display boundary needn't care


def _letterbox(img: np.ndarray, cell_h: int, cell_w: int, pad_value: int) -> np.ndarray:
    """Scale ``img`` to fit a ``cell_h x cell_w`` cell preserving aspect ratio, centered.

    The image is resized by the largest factor that keeps it inside the cell (so neither axis
    overflows), then centered on a ``pad_value`` background; letterbox/pillarbox bars fill the
    leftover. A camera whose aspect already matches the cell fills it exactly with no bars.
    """
    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h)).astype(np.uint8)  # cv2 takes (width, height)
    canvas = np.full((cell_h, cell_w, 3), pad_value, dtype=np.uint8)
    y0, x0 = (cell_h - new_h) // 2, (cell_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _wrap_label(text: str, font: int, scale: float, thickness: int, avail: int) -> list[str]:
    """Greedily wrap ``text`` to lines that fit ``avail`` px, breaking only at ``/`` and ``:``.

    Splits at the natural delimiters (keeping each delimiter on the segment it ends) and packs as
    many segments per line as fit, so ``env0/front_camera:rgb`` becomes one line when it fits and
    breaks at ``/``/``:`` only when it would clip, never mid-word. A single segment that is itself
    too wide is left on its own line (the caller's shrink-to-fit handles that residual case).
    """
    segments = [s for s in re.split(r"(?<=[/:])", text) if s]  # keep delimiter with its segment
    lines, cur = [], ""
    for seg in segments:
        candidate = cur + seg
        if cur and cv2.getTextSize(candidate, font, scale, thickness)[0][0] > avail:
            lines.append(cur)
            cur = seg
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines or [text]


def _draw_label(cell: np.ndarray, text: str) -> None:
    """Draw ``text`` in the top-left corner of ``cell`` in place (black outline + white fill).

    cv2.putText neither wraps nor honors newlines, so ``text`` is one continuous label that is
    wrapped here only when it would clip: first at ``/``/``:`` delimiters (:func:`_wrap_label`), then
    (only if a single unbreakable segment is still too wide) the font is shrunk to fit. So a label
    stays one line when it fits, breaks at natural boundaries when it doesn't, and never clips.
    """
    font, margin, base_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 4, 0.4, 1
    avail = max(1, cell.shape[1] - 2 * margin)
    lines = _wrap_label(text, font, base_scale, thickness, avail)
    line_widths = [cv2.getTextSize(ln, font, base_scale, thickness)[0][0] for ln in lines]
    widest = max([1, *line_widths])  # floor of 1 px guards the avail/widest division below
    scale = base_scale * min(1.0, avail / widest)  # shrink only if an unbreakable line still overflows
    (_, text_h), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_h = text_h + baseline + 2
    y = margin + text_h  # baseline of the first line, a few px below the top edge
    for ln in lines:
        # Black outline under white text keeps the label legible over any background.
        cv2.putText(cell, ln, (margin, y), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(cell, ln, (margin, y), font, scale, _BORDER_COLOR, thickness, cv2.LINE_AA)
        y += line_h


def tile_images(
    images: list[np.ndarray],
    layout: tuple[int, int] | None = None,
    pad_value: int = 0,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Tile ``images`` into one ``HxWx3`` uint8 grid.

    Every cell is a common size (the max H and max W across the list) so the grid lines up. Each
    image is LETTERBOXED into its cell (scaled to fit while preserving its aspect ratio, then
    centered on ``pad_value`` bars) so a mix of differently-shaped cameras is never stretched.
    Each cell is wrapped in a thin white border so adjacent views are separated; trailing empty
    cells (when the count doesn't fill the grid) are filled with ``pad_value`` and bordered to match.

    Parameters
    ----------
    images : list[np.ndarray]
        Non-empty list of ``HxWx3`` uint8 RGB images.
    layout : tuple[int, int] | None
        ``(rows, cols)``; ``None`` picks a near-square layout (cols = ceil(sqrt(n))).
    pad_value : int
        Fill value (0-255) for padding trailing cells.
    labels : list[str] | None
        One label per image (same order), drawn in each cell's corner; ``None`` draws no labels.

    Returns
    -------
    np.ndarray
        Contiguous ``(rows*(cell_h+2*border)) x (cols*(cell_w+2*border)) x 3`` uint8 RGB grid.
    """
    if not images:
        raise ValueError("tile_images requires at least one image.")
    for i, img in enumerate(images):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image {i} must be HxWx3, got shape {img.shape}.")

    n = len(images)
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels has {len(labels)} entries but there are {n} images.")
    if layout is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = layout
        if rows * cols < n:
            raise ValueError(f"layout {layout} has {rows * cols} cells < {n} images.")

    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)

    # Letterbox each image into the common cell (aspect-preserving), label it, then wrap it in a
    # white border. _letterbox returns a fresh array, so the in-place label/border touch the cell
    # only, never the caller's source images.
    cells = [_letterbox(img, cell_h, cell_w, pad_value) for img in images]
    if labels is not None:
        for cell, label in zip(cells, labels):
            _draw_label(cell, label)
    t = _BORDER
    cells = [cv2.copyMakeBorder(c, t, t, t, t, cv2.BORDER_CONSTANT, value=_BORDER_COLOR) for c in cells]
    blank = cv2.copyMakeBorder(
        np.full((cell_h, cell_w, 3), pad_value, dtype=np.uint8), t, t, t, t, cv2.BORDER_CONSTANT, value=_BORDER_COLOR
    )
    cells += [blank] * (rows * cols - n)  # pad trailing cells

    grid_rows = [np.hstack(cells[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.ascontiguousarray(np.vstack(grid_rows))
