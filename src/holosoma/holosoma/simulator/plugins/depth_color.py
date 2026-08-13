"""Depth-map colorization shared by the camera-egress plugins (ROS-free; cv2 deferred).

Turns a float32 depth map (meters, ``+inf`` no-hit) into an ``HxWx3`` uint8 RGB image using a
named OpenCV colormap. Both egress plugins consume this: the viz plugin tiles the result into its
grid, and the ROS2 image plugin colorizes a depth route before wire-encoding it as RGB
(jpeg/png/rgb8). It lives here — neutral ground under ``simulator/plugins`` — so neither egress
flavor imports the other's package.

cv2 is imported inside :func:`colorize_depth` (not at module top), so importing this module pulls in
no heavy dep; the colormap names are validated against the config layer at import with plain strings.
"""

from __future__ import annotations

import numpy as np

from holosoma.config_types.plugin import _DEPTH_COLORMAPS

# Default clamp/normalize range (meters) when a caller does not fix one. NEAR maps bright, FAR dark.
DEFAULT_DEPTH_RANGE: tuple[float, float] = (0.01, 5.0)

# Colormap names rendered via ``cv2.COLORMAP_<NAME>`` (looked up lazily in colorize_depth to keep cv2
# deferred). "gray" is handled directly (grayscale, not a cv2 colormap), so the cv2-backed set is
# exactly the config-accepted set minus "gray". Fails loud at import if the two definitions drift.
_CV2_COLORMAP_NAMES = ("inferno", "turbo", "viridis", "magma", "jet")
assert set(_CV2_COLORMAP_NAMES) | {"gray"} == set(_DEPTH_COLORMAPS), (
    f"_CV2_COLORMAP_NAMES {sorted(_CV2_COLORMAP_NAMES)} + 'gray' must match _DEPTH_COLORMAPS {sorted(_DEPTH_COLORMAPS)}"
)


def colorize_depth(
    depth: np.ndarray,
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
    colormap: str = "inferno",
) -> np.ndarray:
    """Colorize a depth map (meters) to an ``HxWx3`` uint8 RGB image.

    Depth is float32 meters, ``+inf`` for no-hit. It is clamped to ``depth_range`` and normalized so
    NEAR -> bright (255) and FAR -> dark (0), then a colormap is applied; ``+inf``/NaN/no-hit reads as
    the far end (0). The output is RGB (``cv2.applyColorMap`` emits BGR, so we reorder) to match the
    RGB-in convention of both consumers (``tile_images`` and the RGB wire encoders).

    Parameters
    ----------
    depth : np.ndarray
        ``[H,W]`` or ``[H,W,1]`` float depth in meters (``+inf`` = no-hit).
    depth_range : tuple[float, float]
        ``(min_m, max_m)`` clamp/normalize range; min maps to bright, max (and no-hit) to dark.
    colormap : str
        ``"gray"`` for grayscale, else a name in :data:`_CV2_COLORMAP_NAMES` (default ``"inferno"``).
    """
    dep = np.asarray(depth, dtype=np.float32)
    if dep.ndim == 3:
        dep = dep[..., 0]
    lo, hi = depth_range
    # +inf/no-hit and NaN -> the far plane so they read as "background", never as near.
    dep = np.nan_to_num(dep, nan=hi, posinf=hi, neginf=hi)
    # Clamp to range, normalize to [0,1] with near->1, then to uint8 [0,255] (near bright).
    norm = 1.0 - (np.clip(dep, lo, hi) - lo) / (hi - lo)
    gray = (norm * 255.0).round().astype(np.uint8)
    if colormap == "gray":
        return np.repeat(gray[..., None], 3, axis=2)

    import cv2

    bgr = cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{colormap.upper()}"))  # cv2 outputs BGR
    return np.ascontiguousarray(bgr[..., ::-1])  # -> RGB to match the RGB-in convention
