"""CPU-only verification for the optional LiDAR point-cloud figure renderer."""

from __future__ import annotations

import hashlib

import matplotlib.image as mpimg
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tests.simulators.lidar_figures import (
    PointCloudFigureSummary,
    save_pointcloud_figure,
    save_pointcloud_topdown_x_neg_z_projection,
    summarize_pointcloud,
)

pytestmark = pytest.mark.no_sim


def test_save_pointcloud_figure_renders_finite_xyz_on_a_3d_axes(tmp_path, monkeypatch) -> None:
    source_points = np.array(
        [
            [-1.0, -2.0, -3.0],
            [4.0, 5.0, 6.0],
            [0.5, -0.25, 1.5],
            [np.nan, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    expected_points = source_points[:3]
    captured = {}
    original_scatter = Axes3D.scatter

    def capture_scatter(self, xs, ys, zs=0, *args, **kwargs):
        captured["projection"] = self.name
        captured["points"] = np.column_stack((xs, ys, zs))
        return original_scatter(self, xs, ys, zs, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "scatter", capture_scatter)
    path = tmp_path / "cloud.png"
    summary = save_pointcloud_figure(source_points, path, title="known cloud")

    assert captured["projection"] == "3d"
    np.testing.assert_allclose(captured["points"], expected_points)
    assert summary.point_count == len(expected_points)
    assert summary.xyz_sha256 == hashlib.sha256(expected_points.tobytes()).hexdigest()
    assert summarize_pointcloud(source_points) == summary
    image = mpimg.imread(path)
    assert image.ndim == 3 and image.shape[0] >= 900 and image.shape[1] >= 1_000
    assert np.isfinite(image).all() and float(image.max()) > float(image.min())


def test_save_pointcloud_figure_rejects_non_xyz_input(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"Expected point cloud shaped \[N, 3\]"):
        save_pointcloud_figure(np.zeros((4, 2), dtype=np.float32), tmp_path / "invalid.png", title="invalid")


def test_save_pointcloud_figure_rejects_all_invalid_points(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Cannot render an empty point cloud"):
        save_pointcloud_figure(np.full((4, 3), np.nan, dtype=np.float32), tmp_path / "empty.png", title="empty")


def test_save_pointcloud_topdown_x_neg_z_projection_renders_at_equal_scale(tmp_path, monkeypatch) -> None:
    source_points = np.array(
        [
            [-1.0, -2.0, -3.0],
            [4.0, 5.0, 6.0],
            [0.5, -0.25, 1.5],
            [np.nan, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    expected_points = source_points[:3]
    captured = {}
    original_scatter = Axes.scatter
    original_savefig = Figure.savefig

    def capture_scatter(self, xs, ys, *args, **kwargs):
        captured["points"] = np.column_stack((xs, ys))
        return original_scatter(self, xs, ys, *args, **kwargs)

    def capture_savefig(self, *args, **kwargs):
        axes = self.axes[0]
        captured["aspect"] = axes.get_aspect()
        captured["labels"] = (axes.get_xlabel(), axes.get_ylabel())
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    path = tmp_path / "cloud_topdown_x_neg_z.png"
    summary = save_pointcloud_topdown_x_neg_z_projection(
        source_points,
        path,
        title="known cloud top-down X/-Z projection",
    )

    np.testing.assert_allclose(captured["points"], np.column_stack((expected_points[:, 0], -expected_points[:, 2])))
    assert float(captured["aspect"]) == 1.0
    assert captured["labels"] == ("sensor X, right", "sensor -Z, forward")
    assert summary == summarize_pointcloud(source_points)
    image = mpimg.imread(path)
    assert image.ndim == 3 and image.shape[0] >= 900 and image.shape[1] >= 1_000
    assert np.isfinite(image).all() and float(image.max()) > float(image.min())


def test_snapshot_figure_manifest_uses_topdown_x_neg_z_renderer(tmp_path, monkeypatch) -> None:
    from tests.simulators import lidar_assert

    points = np.asarray([[[1.0, 2.0, -3.0]]], dtype=np.float32)
    summary = PointCloudFigureSummary(point_count=1, xyz_sha256="known-buffer")
    captured = {}

    def return_summary(*_args, **_kwargs):
        return summary

    monkeypatch.setattr(lidar_assert, "save_pointcloud_figure", return_summary)
    monkeypatch.setattr(lidar_assert, "summarize_pointcloud", return_summary)

    def capture_topdown(rendered_points, path, *, title):
        captured["points"] = rendered_points
        captured["path"] = path
        captured["title"] = title
        return summary

    monkeypatch.setattr(lidar_assert, "save_pointcloud_topdown_x_neg_z_projection", capture_topdown)

    manifest = lidar_assert._save_snapshot_figures(
        points,
        output_dir=tmp_path,
        simulator="mujoco",
        step_index=2,
        num_envs=1,
    )

    np.testing.assert_array_equal(captured["points"], points[0])
    assert captured["path"].name == "mujoco_lidar_topdown_x_neg_z_step_002.png"
    assert captured["title"] == "mujoco LiDAR snapshot 2 top-down X/-Z projection"
    assert manifest["topdown_x_neg_z_path"] == captured["path"].name
    assert manifest["topdown_x_neg_z_xyz_sha256"] == summary.xyz_sha256
