"""Backend-neutral clipping for depth and LiDAR range outputs."""

from __future__ import annotations

from dataclasses import dataclass

from holosoma.config_types.sensor import NoReturnClippingBehavior
from holosoma.utils.safe_torch_import import torch


@dataclass(frozen=True)
class RangeClippingResult:
    """Reported distances and the mask identifying physical in-range returns."""

    ranges: torch.Tensor
    hit_mask: torch.Tensor


def clip_sensor_ranges(
    distances: torch.Tensor,
    *,
    near: float,
    far: float,
    behavior: NoReturnClippingBehavior,
) -> RangeClippingResult:
    """Apply one cross-backend no-return convention to metric sensor distances.

    A physical return is finite and lies in the inclusive ``[near, far]`` interval. Misses and
    out-of-range returns are reported as ``+inf`` (``"none"``), the far endpoint (``"max"``), or
    zero (``"zero"``). ``hit_mask`` remains false for synthetic endpoint and zero values.

    ``near`` and ``far`` must be finite with ``near >= 0``, which every sensor config enforces. The
    interval test relies on it to reject non-finite distances, so an infinite ``far`` would report a
    ``+inf`` miss as a hit.
    """
    # The interval test alone rejects every non-finite value, because the sensor configs validate
    # near and far as finite with near >= 0: NaN fails both comparisons, +inf fails ``<= far`` and
    # -inf fails ``>= near``. A separate isfinite pass would re-read the whole frame to agree.
    hit_mask = (distances >= near) & (distances <= far)
    clip_values = {
        "none": float("inf"),
        "max": far,
        "zero": 0.0,
    }
    # Scalar ``other``, so no full-size fill tensor is allocated per frame.
    ranges = torch.where(hit_mask, distances, clip_values[behavior])
    return RangeClippingResult(ranges=ranges.to(torch.float32), hit_mask=hit_mask)
