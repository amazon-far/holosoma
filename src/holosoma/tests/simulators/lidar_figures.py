"""3D point-cloud renderer used by the live LiDAR harness."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PointCloudFigureSummary:
    point_count: int
    xyz_sha256: str


def _finite_xyz(points):
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected point cloud shaped [N, 3], got {xyz.shape}.")
    xyz = np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1)])
    if not len(xyz):
        raise RuntimeError("Cannot render an empty point cloud.")
    return xyz


def _summarize_xyz(xyz: np.ndarray) -> PointCloudFigureSummary:
    return PointCloudFigureSummary(
        point_count=len(xyz),
        xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
    )


def summarize_pointcloud(points) -> PointCloudFigureSummary:
    """Summarize the finite XYZ input consumed by the renderer."""
    return _summarize_xyz(_finite_xyz(points))


def save_pointcloud_figure(points, path: Path, *, title: str) -> PointCloudFigureSummary:
    """Save finite optical-frame XYZ points as a range-colored 3D PNG."""
    xyz = _finite_xyz(points)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7, 6))
    axes = figure.add_subplot(111, projection="3d")
    color = np.linalg.norm(xyz, axis=1)
    axes.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, cmap="viridis", s=7, linewidths=0)
    axes.set(xlabel="sensor X, right (m)", ylabel="sensor Y, up (m)", zlabel="sensor Z, back (m)", title=title)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    half_extent = max(float(np.ptp(xyz, axis=0).max()) / 2.0, 0.1)
    axes.set_xlim(center[0] - half_extent, center[0] + half_extent)
    axes.set_ylim(center[1] - half_extent, center[1] + half_extent)
    axes.set_zlim(center[2] - half_extent, center[2] + half_extent)
    axes.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return _summarize_xyz(xyz)


def save_pointcloud_topdown_x_neg_z_projection(points, path: Path, *, title: str) -> PointCloudFigureSummary:
    """Save finite optical-frame points as an equal-scale top-down X/-Z projection PNG."""
    xyz = _finite_xyz(points)
    topdown = np.column_stack((xyz[:, 0], -xyz[:, 2]))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 6))
    color = np.linalg.norm(xyz, axis=1)
    axes.scatter(topdown[:, 0], topdown[:, 1], c=color, cmap="viridis", s=7, linewidths=0)
    center = (topdown.min(axis=0) + topdown.max(axis=0)) / 2.0
    half_extent = max(float(np.ptp(topdown, axis=0).max()) / 2.0, 0.1)
    axes.set(
        xlim=(center[0] - half_extent, center[0] + half_extent),
        ylim=(center[1] - half_extent, center[1] + half_extent),
        xlabel="sensor X, right",
        ylabel="sensor -Z, forward",
        title=title,
    )
    axes.set_aspect("equal", adjustable="box")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return _summarize_xyz(xyz)
