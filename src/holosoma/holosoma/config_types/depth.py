"""Depth-image preprocessing and augmentation settings for depth observation terms.

:class:`DepthProcessingConfig` is the deterministic transform from a metric depth frame to the
network input; deployment must apply exactly the same transform to the real camera. The
:class:`DepthAugmentationConfig` effects simulate real depth-sensor artifacts during training and
are off by default.
"""

from __future__ import annotations

import math
from dataclasses import field

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

_FORBID_EXTRA = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthProcessingConfig:
    """Clamp, crop, resize, and normalize a metric depth frame.

    Applied in this order: clamp to ``[near, far]`` (non-finite values become ``far``), crop the
    border, bicubic resize with antialiasing, clamp to ``far`` again (bicubic overshoots at depth
    edges), replace values below ``empty_threshold`` with ``far``, then normalize to
    ``(depth - near) / (far - near) - 0.5``. Bicubic undershoot below ``near`` is kept, so the
    output range is approximately, not strictly, ``[-0.5, 0.5]``.
    """

    near: float = 0.3
    """Closest valid depth in meters; closer values are clamped to it."""

    far: float = 3.0
    """Farthest valid depth in meters; farther values and no-returns are clamped to it."""

    crop_top: int = 0
    """Rows removed from the top of the frame before resizing."""

    crop_bottom: int = 0
    """Rows removed from the bottom of the frame before resizing."""

    crop_left: int = 0
    """Columns removed from the left of the frame before resizing."""

    crop_right: int = 0
    """Columns removed from the right of the frame before resizing."""

    resize: list[int] | None = None
    """Output ``[height, width]`` after cropping, or ``None`` to keep the cropped size."""

    empty_threshold: float = 0.15
    """Depths below this (bicubic undershoot or invalid pixels) are reported as ``far``."""

    @model_validator(mode="after")
    def validate_processing(self) -> DepthProcessingConfig:
        """Validate the depth range, crop, and resize."""
        if not (math.isfinite(self.near) and math.isfinite(self.far) and 0.0 <= self.near < self.far):
            raise ValueError(f"DepthProcessingConfig needs 0 <= near < far, got {self.near}..{self.far}.")
        crops = (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
        if any(value < 0 for value in crops):
            raise ValueError(f"DepthProcessingConfig crops must be non-negative, got {crops}.")
        if self.resize is not None and (len(self.resize) != 2 or min(self.resize) <= 0):
            raise ValueError(f"DepthProcessingConfig.resize must be [height, width] > 0, got {self.resize}.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthEdgeNoiseConfig:
    """Stereo-like artifacts at depth discontinuities, found with a Sobel gradient.

    Pixels near strong edges are replaced by a random 3x3 neighbor, and far-side pixels at
    moderate edges are dropped to ``far``.
    """

    enabled: bool = False
    shuffle_gradient_threshold: float = 1.0
    """Sobel magnitude above which a pixel may be shuffled with a neighbor."""
    shuffle_probability: float = 0.9
    """Probability that an edge pixel seeds a shuffled 3x3 region."""
    drop_gradient_threshold: float = 0.6
    """Sobel magnitude above which a far-side pixel may be dropped."""
    drop_min_depth: float = 2.5
    """A pixel is far-side if the 3x3 max depth around it exceeds this (meters)."""
    drop_probability: float = 0.7
    """Probability that a far-side edge pixel is dropped."""
    border: int = 3
    """Pixels this close to the image border are never modified."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthHolesConfig:
    """Temporally coherent Perlin-noise holes, as from missing stereo matches."""

    enabled: bool = False
    threshold: float = 0.2
    """Fraction of the normalized noise range below which a pixel becomes a hole."""
    min_depth: float = 0.2
    """Only pixels deeper than this (meters) can become holes."""
    max_depth: float = 2.0
    """Only pixels shallower than this (meters) can become holes."""
    noise_shape: list[int] = field(default_factory=lambda: [64, 96])
    """``[height, width]`` the noise is generated at before resizing to the image."""
    seed: int | None = None
    """Offsets the noise sequence. ``None`` reproduces the fixed sequence of earlier releases,
    identical on every run and process."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthGaussianNoiseConfig:
    """Independent per-pixel Gaussian noise on every frame."""

    enabled: bool = False
    std: float = 0.03
    """Standard deviation in meters."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthOffsetConfig:
    """A constant per-env depth offset, like a camera calibration bias, drawn once."""

    enabled: bool = False
    std: float = 0.03
    """Standard deviation of the offset in meters."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DepthAugmentationConfig:
    """Training-time depth artifacts, applied after resizing and before normalization in this
    order: edge noise, holes, Gaussian noise, per-env offset. Each is off unless enabled."""

    edge_noise: DepthEdgeNoiseConfig = field(default_factory=DepthEdgeNoiseConfig)
    holes: DepthHolesConfig = field(default_factory=DepthHolesConfig)
    gaussian_noise: DepthGaussianNoiseConfig = field(default_factory=DepthGaussianNoiseConfig)
    per_env_offset: DepthOffsetConfig = field(default_factory=DepthOffsetConfig)
