"""ROS-free camera-intrinsics math for the ROS2 image egress CameraInfo.

Derives the 3x3 K, 3x4 P, distortion D, and rectification R from ``CameraIntrinsics``. Kept
separate from ``ros2_image_plugin.py`` so the projection math is unit-tested without a ROS
environment; the egress just copies these arrays into a ``sensor_msgs/CameraInfo``.

Pinhole cameras are represented exactly. Native Isaac Sim fisheye cameras use a radial f-theta
polynomial that ROS cannot encode directly, so it is fit to ROS's four-coefficient ``equidistant``
model over the rendered lens's usable radius. The result is a best-effort approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from holosoma.config_types.sensor import IsaacSimCameraConfig, IsaacSimFisheyeConfig
from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics


@dataclass(frozen=True)
class CameraInfoData:
    """Plain CameraInfo payload (lists, ROS-free) the egress copies into the message."""

    width: int
    height: int
    k: list[float]  # 3x3 row-major
    p: list[float]  # 3x4 row-major
    d: list[float]  # distortion (empty model -> zeros)
    r: list[float]  # 3x3 rectification (identity for a monocular pinhole)
    distortion_model: str = "plumb_bob"


def focal_length_px(height: int, vertical_fov_deg: float) -> float:
    """Pixel focal length from image height and vertical FOV: f = (H/2) / tan(vfov/2)."""
    half = math.radians(vertical_fov_deg) / 2.0
    return (height / 2.0) / math.tan(half)


def _camera_info_data(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_model: str,
    distortion: list[float],
) -> CameraInfoData:
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    # Monocular P = [K | 0] (no stereo baseline term on Tx).
    p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return CameraInfoData(
        width=width,
        height=height,
        k=k,
        p=p,
        d=distortion,
        r=r,
        distortion_model=distortion_model,
    )


def _fit_equidistant_inverse(
    fisheye: IsaacSimFisheyeConfig,
    projection_type: str,
) -> npt.NDArray[np.float64]:
    r"""Least-squares invert native ``theta(r)`` into ROS equidistant ``r(theta)``.

    Isaac's coefficients define ray angle from nominal pixel radius:
    ``theta(r) = a + b*r + c*r^2 + ... + f*r^5``. ROS/OpenCV equidistant instead defines
    ``r(theta) = f_px*theta*(1 + k1*theta^2 + ... + k4*theta^8)``. Solving for the five odd-power
    coefficients ``[f_px, f_px*k1, ..., f_px*k4]`` makes the two models agree in nominal pixels.
    """
    polynomial = np.array(
        [
            fisheye.polynomial_a,
            fisheye.polynomial_b,
            fisheye.polynomial_c,
            fisheye.polynomial_d,
            fisheye.polynomial_e,
            fisheye.polynomial_f,
        ],
        dtype=np.float64,
    )
    x_radius = max(abs(fisheye.optical_centre_x), abs(fisheye.nominal_width - fisheye.optical_centre_x))
    y_radius = max(abs(fisheye.optical_centre_y), abs(fisheye.nominal_height - fisheye.optical_centre_y))
    nominal_radius = math.hypot(x_radius, y_radius)

    # max_fov is the full angular extent, while theta is measured from the optical axis.
    theta_limit = math.radians(fisheye.max_fov) / 2.0
    limit_polynomial = polynomial.copy()
    limit_polynomial[0] -= theta_limit
    roots = np.polynomial.polynomial.polyroots(limit_polynomial)
    crossings = sorted(
        float(root.real) for root in roots if abs(root.imag) <= 1e-8 and 0.0 < root.real < nominal_radius
    )
    radial_limit = crossings[0] if crossings else nominal_radius

    radii = np.linspace(0.0, radial_limit, 512, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        theta = np.polynomial.polynomial.polyval(radii, polynomial)
    valid = np.isfinite(theta) & (theta >= 0.0) & (theta <= theta_limit * (1.0 + 1e-12))
    radii = radii[valid]
    theta = theta[valid]
    if theta.size < 5 or float(np.ptp(theta)) <= np.finfo(np.float64).eps:
        raise ValueError(f"Projection '{projection_type}' has no usable radial f-theta calibration.")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError(f"Projection '{projection_type}' has a non-monotonic radial f-theta calibration.")

    # Solve in normalized theta to avoid conditioning the ninth-order column on the FOV magnitude.
    theta_scale = float(np.max(np.abs(theta)))
    normalized_theta = theta / theta_scale
    powers = np.arange(1, 10, 2, dtype=np.float64)
    basis = normalized_theta[:, None] ** powers
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(basis, radii, rcond=None)
    coefficients = cast("npt.NDArray[np.float64]", scaled_coefficients / (theta_scale**powers))
    if rank != len(powers) or not np.all(np.isfinite(coefficients)) or coefficients[0] <= 0.0:
        raise ValueError(f"Projection '{projection_type}' cannot be approximated by ROS equidistant.")
    return coefficients


def camera_info_from_intrinsics(intr: CameraIntrinsics) -> CameraInfoData:
    """Build exact pinhole or best-effort equidistant fisheye CameraInfo data."""
    isaacsim = intr.backend_config if isinstance(intr.backend_config, IsaacSimCameraConfig) else None
    if isaacsim is None or isaacsim.projection_type == "pinhole":
        f = focal_length_px(intr.height, intr.vertical_fov)  # fx == fy (square pixels)
        return _camera_info_data(
            width=intr.width,
            height=intr.height,
            fx=f,
            fy=f,
            cx=intr.width / 2.0,
            cy=intr.height / 2.0,
            distortion_model="plumb_bob",
            distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    fisheye = isaacsim.fisheye
    inverse = _fit_equidistant_inverse(fisheye, isaacsim.projection_type)
    scale_x = intr.width / fisheye.nominal_width
    scale_y = intr.height / fisheye.nominal_height
    focal_nominal = float(inverse[0])
    return _camera_info_data(
        width=intr.width,
        height=intr.height,
        fx=focal_nominal * scale_x,
        fy=focal_nominal * scale_y,
        cx=fisheye.optical_centre_x * scale_x,
        cy=fisheye.optical_centre_y * scale_y,
        distortion_model="equidistant",
        distortion=[float(value / focal_nominal) for value in inverse[1:]],
    )
