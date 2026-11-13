"""Depth-map colorization shared by the camera-egress plugins (ROS-free; cv2 deferred).

Turns a float32 depth map (meters, ``+inf`` no-hit) into an ``HxWx3`` uint8 RGB image using a
named OpenCV colormap. Both egress plugins consume this: the viz plugin tiles the result into its
grid, and the ROS2 image plugin colorizes a depth route before wire-encoding it as RGB
(jpeg/png/rgb8). It lives here — neutral ground under ``simulator/plugins`` — so neither egress
flavor imports the other's package.

cv2 is imported inside :func:`colorize_depth` (not at module top), so importing this module pulls in
no heavy dep; the colormap names are validated against the config layer at import with plain strings.
Every colormap, ``"gray"`` included, now goes through cv2: it carries the normalization itself, not
just the colormap lookup.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from holosoma.config_types.plugin import _DEPTH_COLORMAPS

# Default clamp/normalize range (meters) when a caller does not fix one. NEAR maps bright, FAR dark.
DEFAULT_DEPTH_RANGE: tuple[float, float] = (0.01, 5.0)

# Colormap names rendered via ``cv2.COLORMAP_<NAME>`` (looked up lazily in colorize_depth to keep cv2
# deferred). "gray" needs no colormap lookup, so the cv2-backed set is exactly the config-accepted
# set minus "gray". Fails loud at import if the two definitions drift.
_CV2_COLORMAP_NAMES = ("inferno", "turbo", "viridis", "magma", "jet")
assert set(_CV2_COLORMAP_NAMES) | {"gray"} == set(_DEPTH_COLORMAPS), (
    f"_CV2_COLORMAP_NAMES {sorted(_CV2_COLORMAP_NAMES)} + 'gray' must match _DEPTH_COLORMAPS {sorted(_DEPTH_COLORMAPS)}"
)


def colorize_depth(
    depth: npt.NDArray[np.float32],
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
    colormap: str = "inferno",
) -> npt.NDArray[np.uint8]:
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
    import cv2

    dep: npt.NDArray[np.float32] = np.asarray(depth, dtype=np.float32)
    if dep.ndim == 3:
        dep = dep[..., 0]
    lo, hi = depth_range
    # +inf/no-hit and NaN -> the far plane so they read as "background", never as near. One isfinite
    # pass covers all three non-finite cases, where nan_to_num makes a masked pass per sentinel.
    # The result is a fresh array, so the clamp below runs in place; the caller's array is the host
    # snapshot shared with every other consumer this step and must not be mutated.
    dep = np.where(np.isfinite(dep), dep, hi).astype(np.float32, copy=False)
    # Normalize to uint8 [0,255] with near->bright in one SIMD pass: convertScaleAbs computes
    # saturate_cast<uchar>(|dep * alpha + beta|), which folds the scale, the rounding and both ends
    # of the clamp that an explicit np.clip would need. The negative slope puts near at 255 without
    # a second inverting pass, and clamping only the FAR end keeps ``hi - dep`` non-negative, which
    # is what makes that abs() a no-op; saturate_cast clips the >255 result at the near end.
    np.minimum(dep, hi, out=dep)
    # convertScaleAbs always emits CV_8U; its stub widens the return to integer|floating.
    gray = cast(
        "npt.NDArray[np.uint8]",
        cv2.convertScaleAbs(dep, alpha=-255.0 / (hi - lo), beta=255.0 * hi / (hi - lo)),
    )
    if colormap == "gray":
        return np.repeat(gray[..., None], 3, axis=2)

    bgr = cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{colormap.upper()}"))  # cv2 outputs BGR
    return np.ascontiguousarray(bgr[..., ::-1])  # -> RGB to match the RGB-in convention
