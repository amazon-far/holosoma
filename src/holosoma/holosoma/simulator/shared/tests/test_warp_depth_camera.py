"""Warp depth camera geometry, self-occlusion, partial rendering, and the optical-frame convention.

Runs on the CPU Warp device against a fake simulator exposing only the state the camera reads.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from holosoma.config_types.sensor import (
    SensorMountConfig,
    WarpDepthCameraConfig,
    WarpDepthMountRandomizationConfig,
)
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.simulator.shared.warp_depth_camera import register_warp_depth_cameras
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

wp = pytest.importorskip("warp")
trimesh = pytest.importorskip("trimesh")
from holosoma.utils.warp_utils import convert_to_wp_mesh  # noqa: E402

NUM_ENVS = 3
LINKS = ["base", "occluder"]


def _ground(size: float = 50.0) -> Any:
    vertices = np.array([[-size, -size, 0.0], [size, -size, 0.0], [size, size, 0.0], [-size, size, 0.0]])
    return convert_to_wp_mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]), "cpu")


class _FakeSim:
    """Robot links ``base`` (camera mount) and ``occluder``, over a ground plane at z = 0."""

    def __init__(self, config: WarpDepthCameraConfig, base_height: float = 2.0) -> None:
        self.num_envs = NUM_ENVS
        self.sim_device = "cpu"
        self.sensor_config = {"depth": config}
        self.rigid_body_pos_w = torch.zeros(NUM_ENVS, len(LINKS), 3)
        self.rigid_body_pos_w[:, 0, 2] = base_height
        self.rigid_body_pos_w[:, 1] = torch.tensor([0.0, 0.0, -100.0])  # out of view by default
        self.rigid_body_quat_w = torch.zeros(NUM_ENVS, len(LINKS), 4)
        self.rigid_body_quat_w[..., 3] = 1.0
        ground = _ground()
        self.terrain_manager = SimpleNamespace(get_state=lambda _name: SimpleNamespace(warp_mesh=ground))

    def find_rigid_body_indice(self, name: str) -> int:
        return LINKS.index(name)


def _camera(tmp_path: Path, **overrides: Any) -> WarpDepthCameraConfig:
    box = tmp_path / "box.stl"
    if not box.exists():
        trimesh.creation.box(extents=[0.4, 0.4, 0.4]).export(box)
    fields: dict[str, Any] = {
        # Identity mount on an identity link: optical -Z looks down world -Z, +Y is world +Y.
        "mount": SensorMountConfig(target_kind="robot_link", target="base"),
        "width": 16,
        "height": 12,
        "vertical_fov": 60.0,
        "far": 10.0,
        "self_occlusion": {"occluder": str(box)},
    }
    fields.update(overrides)
    return WarpDepthCameraConfig(**fields)


def _render(sim: _FakeSim) -> tuple[torch.Tensor, Any, Any]:
    """Register the fake sim's camera, render one control step, and return (depth, record, camera)."""
    manager = SensorManager(device="cpu", control_hz=50.0)
    cameras = register_warp_depth_cameras(sim, manager)  # type: ignore[arg-type]
    record = manager.get("depth")
    manager.render_shared()
    return record.buffers["depth"][..., 0], record, cameras["depth"]


def test_ground_plane_parallel_to_the_image_plane_has_constant_depth(tmp_path: Path) -> None:
    depth, _, _ = _render(_FakeSim(_camera(tmp_path), base_height=2.0))

    assert depth.shape == (NUM_ENVS, 12, 16)
    torch.testing.assert_close(depth, torch.full_like(depth, 2.0), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(("width", "height"), [(15, 11), (16, 11)])
def test_odd_image_sizes_measure_depth_to_the_image_plane(tmp_path: Path, width: int, height: int) -> None:
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, width=width, height=height), base_height=2.0))

    torch.testing.assert_close(depth, torch.full_like(depth, 2.0), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(("behavior", "expected"), [("max", 3.0), ("none", math.inf), ("zero", 0.0)])
def test_surfaces_beyond_far_are_no_return(tmp_path: Path, behavior: str, expected: float) -> None:
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, far=3.0, depth_clipping_behavior=behavior), base_height=5.0))

    assert torch.all(depth == expected)


def test_self_occlusion_mesh_is_cast_at_its_link_pose(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    sim.rigid_body_pos_w[0, 1] = torch.tensor([0.0, 0.0, 1.0])  # box spans z in [0.8, 1.2], centered below

    depth, _, _ = _render(sim)

    assert depth[0, 6, 8].item() == pytest.approx(2.0 - 1.2, abs=1e-4)  # top face of the box
    assert depth[0, 0, 0].item() == pytest.approx(2.0, abs=1e-4)  # corner still sees the ground
    torch.testing.assert_close(depth[1:], torch.full_like(depth[1:], 2.0), atol=1e-4, rtol=0.0)


def test_image_rows_run_top_down_along_optical_plus_y_and_columns_along_plus_x(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    sim.rigid_body_pos_w[0, 1] = torch.tensor([0.0, 0.5, 1.0])  # +Y of the camera
    sim.rigid_body_pos_w[1, 1] = torch.tensor([0.5, 0.0, 1.0])  # +X of the camera

    depth, _, _ = _render(sim)
    occluded = depth < 1.9

    rows = occluded[0].nonzero()[:, 0]
    cols = occluded[1].nonzero()[:, 1]
    assert rows.numel() > 0 and rows.float().mean() < 6  # upper half of the image
    assert cols.numel() > 0 and cols.float().mean() > 8  # right half of the image


def test_depth_is_measured_to_the_image_plane_for_a_tilted_camera(tmp_path: Path) -> None:
    pitch = math.radians(30.0)
    # Tilt the optical axis away from straight down by rotating the mount about +X.
    mount = SensorMountConfig(
        target_kind="robot_link",
        target="base",
        orientation=[math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0],
    )
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, mount=mount), base_height=2.0))

    # The principal ray hits the ground at 2 / cos(30deg) along the optical axis.
    assert depth[0, 6, 8].item() == pytest.approx(2.0 / math.cos(pitch), abs=1e-3)


def test_partial_rerender_updates_only_the_requested_envs(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    _, record, _ = _render(sim)
    sim.rigid_body_pos_w[:, 0, 2] = 1.5  # every env moved, but only env 2 is re-rendered

    record.rerender(torch.tensor([2]))
    depth = record.buffers["depth"][..., 0]

    torch.testing.assert_close(depth[2], torch.full_like(depth[2], 1.5), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(depth[:2], torch.full_like(depth[:2], 2.0), atol=1e-4, rtol=0.0)


def test_shared_cameras_are_not_returned_to_backend_render_loops(tmp_path: Path) -> None:
    manager = SensorManager(device="cpu", control_hz=50.0)
    register_warp_depth_cameras(_FakeSim(_camera(tmp_path)), manager)  # type: ignore[arg-type]

    assert manager.collect_due() == []
    assert manager.has_camera("depth")


def test_mount_randomization_stays_within_bounds(tmp_path: Path) -> None:
    randomization = WarpDepthMountRandomizationConfig(
        position_min=[-0.1, -0.1, -0.1],
        position_max=[0.1, 0.1, 0.1],
        euler_deg_min=[-5.0, -5.0, -5.0],
        euler_deg_max=[5.0, 5.0, 5.0],
    )
    torch.manual_seed(0)
    sim = _FakeSim(_camera(tmp_path, mount_randomization=randomization), base_height=2.0)
    _, record, camera = _render(sim)

    offsets = camera._cam_pos - sim.rigid_body_pos_w[:, 0]
    assert torch.all(offsets.abs() <= 0.1 + 1e-6)
    assert not torch.allclose(offsets[0], offsets[1])
    # The principal ray reaches the ground at the perturbed height over the cosine of the tilt,
    # which two 5 deg rotations bound by about 7.1 deg.
    center = record.buffers["depth"][:, 6, 8, 0]
    assert torch.all((center >= 1.9 - 1e-4) & (center <= 2.1 / math.cos(math.radians(7.1))))
