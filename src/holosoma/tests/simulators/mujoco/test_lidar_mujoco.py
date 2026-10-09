"""Live MuJoCo LiDAR tests for classic and Warp backends."""

import json
import os
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytest.importorskip("mujoco")

_HARNESS = Path(__file__).resolve().parents[1] / "lidar_assert.py"


@pytest.mark.mujoco_classic
def test_lidar_classic(tmp_path):
    result_file = tmp_path / "lidar_mujoco.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/lidar",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_lidar_warp(tmp_path, monkeypatch):
    result_file = tmp_path / "lidar_mjwarp.txt"
    figure_dir = tmp_path / "pointcloud-figures"
    expected_figures = [figure_dir / f"mjwarp_lidar_step_{step:03d}.png" for step in range(2)]
    expected_topdown_figures = [figure_dir / f"mjwarp_lidar_topdown_x_neg_z_step_{step:03d}.png" for step in range(2)]
    expected_manifest = figure_dir / "mjwarp_lidar_figures.json"
    if "HOLOSOMA_LIDAR_FIGURE_DIR" not in os.environ:
        monkeypatch.setenv("HOLOSOMA_LIDAR_FIGURE_DIR", str(tmp_path / "requested-pointcloud-figures"))
    output_dir = Path(os.environ["HOLOSOMA_LIDAR_FIGURE_DIR"])
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        "--pointcloud-figure-dir",
        str(figure_dir),
        label="mjwarp/lidar",
        timeout=400,
        result_file=result_file,
    )
    for path in [*expected_figures, *expected_topdown_figures]:
        assert path.is_file() and path.stat().st_size > 0, f"missing or empty point-cloud figure: {path}"
        image = mpimg.imread(path)
        assert image.ndim == 3 and image.shape[0] >= 900 and image.shape[1] >= 1_000
        assert np.isfinite(image).all() and float(image.max()) > float(image.min())
    manifest = json.loads(expected_manifest.read_text())
    snapshots = manifest["snapshots"]
    assert [snapshot["path"] for snapshot in snapshots] == [path.name for path in expected_figures]
    assert [snapshot["topdown_x_neg_z_path"] for snapshot in snapshots] == [
        path.name for path in expected_topdown_figures
    ]
    assert all(snapshot["point_count"] > 0 for snapshot in snapshots)
    assert len({snapshot["xyz_sha256"] for snapshot in snapshots}) == len(snapshots)
    for snapshot in snapshots:
        environment_summaries = snapshot["environment_summaries"]
        assert snapshot["xyz_sha256"] == environment_summaries[0]["xyz_sha256"]
        assert snapshot["topdown_x_neg_z_xyz_sha256"] == environment_summaries[0]["xyz_sha256"]
        assert len({summary["xyz_sha256"] for summary in environment_summaries}) == 3

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [*expected_figures, *expected_topdown_figures, expected_manifest]:
        destination = output_dir / path.name
        shutil.copy2(path, destination)
        assert destination.is_file() and destination.stat().st_size == path.stat().st_size
