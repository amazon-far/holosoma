"""Ray-cast depth camera geometry, self-occlusion, partial rendering, and the optical-frame convention.

Runs on the CPU Warp device against a fake simulator exposing only the state the camera reads.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimCameraConfig, SensorMountConfig
from holosoma.simulator.shared.raycast_depth_camera import RayCastDepthCamera
from holosoma.simulator.shared.raycast_scene import RaycastSceneMeshes
from holosoma.simulator.shared.sensor_manager import SensorManager
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


class _FakeScene:
    """Already loaded geometry, independent of a terrain manager or asset file inventory."""

    def __init__(self, sim: _FakeSim, occluders: list[Any] | None = None) -> None:
        self.sim = sim
        box = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
        self.meshes = [_ground()] + (
            [convert_to_wp_mesh(box.vertices, box.faces, "cpu")] if occluders is None else occluders
        )
        self.positions = torch.zeros(NUM_ENVS, len(self.meshes), 3)
        self.orientations = torch.zeros(NUM_ENVS, len(self.meshes), 4)
        self.orientations[..., 3] = 1.0
        self.data = RaycastSceneMeshes(
            mesh_ids=wp.array([[mesh.id for mesh in self.meshes]] * NUM_ENVS, dtype=wp.uint64, device="cpu"),
            positions=wp.from_torch(self.positions, dtype=wp.vec3),
            orientations=wp.from_torch(self.orientations, dtype=wp.quat),
            num_meshes=len(self.meshes),
        )
        self.initializations = 0

    def prepare_meshes(self) -> RaycastSceneMeshes:
        self.initializations += 1
        return self.data

    def update_mesh_poses(self) -> None:
        self.positions[:, 1:] = self.sim.rigid_body_pos_w[:, 1:2]
        self.orientations[:, 1:] = self.sim.rigid_body_quat_w[:, 1:2]


class _FakeSim:
    """Robot links ``base`` (camera mount) and ``occluder``, over a ground plane at z = 0."""

    def __init__(self, config: CameraSensorConfig, base_height: float = 2.0) -> None:
        self.num_envs = NUM_ENVS
        self.sim_device = "cpu"
        self.sensor_config = {"depth": config}
        self.rigid_body_pos_w = torch.zeros(NUM_ENVS, len(LINKS), 3)
        self.rigid_body_pos_w[:, 0, 2] = base_height
        self.rigid_body_pos_w[:, 1] = torch.tensor([0.0, 0.0, -100.0])  # out of view by default
        self.rigid_body_quat_w = torch.zeros(NUM_ENVS, len(LINKS), 4)
        self.rigid_body_quat_w[..., 3] = 1.0
        self.scene = _FakeScene(self)

    def find_rigid_body_indice(self, name: str) -> int:
        return LINKS.index(name)


def _camera(tmp_path: Path, **overrides: Any) -> CameraSensorConfig:
    fields: dict[str, Any] = {
        # Identity mount on an identity link: optical -Z looks down world -Z, +Y is world +Y.
        "mount": SensorMountConfig(target_kind="robot_link", target="base"),
        "width": 16,
        "height": 12,
        "vertical_fov": 60.0,
        "far": 10.0,
        "data_types": ["depth"],
    }
    fields.update(overrides)
    return CameraSensorConfig(isaacsim=IsaacSimCameraConfig(renderer="raycast"), **fields)


def _camera_and_record(sim: _FakeSim) -> tuple[RayCastDepthCamera, Any]:
    config = sim.sensor_config["depth"]
    record = SensorManager(device="cpu", control_hz=50.0).register_camera("depth", config)
    camera = RayCastDepthCamera(sim, config, sim.scene)  # type: ignore[arg-type]
    camera.initialize()
    return camera, record


def _render(sim: _FakeSim) -> tuple[torch.Tensor, Any, Any]:
    """Render the fake sim's camera once into its record and return (depth, record, camera)."""
    camera, record = _camera_and_record(sim)
    camera.render(record)
    return record.buffers["depth"][..., 0], record, camera


@pytest.mark.parametrize(("width", "height"), [(16, 12), (15, 11), (16, 11)])
def test_ground_plane_parallel_to_the_image_plane_has_constant_depth(tmp_path: Path, width: int, height: int) -> None:
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, width=width, height=height), base_height=2.0))

    assert depth.shape == (NUM_ENVS, height, width)
    torch.testing.assert_close(depth, torch.full_like(depth, 2.0), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(("behavior", "expected"), [("max", 3.0), ("none", math.inf), ("zero", 0.0)])
def test_surfaces_beyond_far_are_no_return(tmp_path: Path, behavior: str, expected: float) -> None:
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, far=3.0, depth_clipping_behavior=behavior), base_height=5.0))

    assert torch.all(depth == expected)


def test_link_mesh_is_cast_at_its_link_pose(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    sim.rigid_body_pos_w[0, 1] = torch.tensor([0.0, 0.0, 1.0])  # box spans z in [0.8, 1.2], centered below

    depth, _, _ = _render(sim)

    assert depth[0, 6, 8].item() == pytest.approx(2.0 - 1.2, abs=1e-4)  # top face of the box
    assert depth[0, 0, 0].item() == pytest.approx(2.0, abs=1e-4)  # corner still sees the ground
    torch.testing.assert_close(depth[1:], torch.full_like(depth[1:], 2.0), atol=1e-4, rtol=0.0)


def test_every_scene_mesh_owned_by_a_body_is_cast_at_its_pose(tmp_path: Path) -> None:
    # Two halves of one link, side by side along the link's x axis.
    left = trimesh.creation.box(
        extents=[0.2, 0.4, 0.4], transform=trimesh.transformations.translation_matrix([-0.1, 0, 0])
    )
    right = trimesh.creation.box(
        extents=[0.2, 0.4, 0.2], transform=trimesh.transformations.translation_matrix([0.1, 0, 0])
    )
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    sim.scene = _FakeScene(sim, [convert_to_wp_mesh(mesh.vertices, mesh.faces, "cpu") for mesh in (left, right)])
    sim.rigid_body_pos_w[0, 1] = torch.tensor([0.0, 0.0, 1.0])

    depth, _, _ = _render(sim)

    assert depth[0, 6, 7].item() == pytest.approx(2.0 - 1.2, abs=1e-4)  # left half, top at z = 1.2
    assert depth[0, 6, 9].item() == pytest.approx(2.0 - 1.1, abs=1e-4)  # right half, top at z = 1.1


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
    depth, _, _ = _render(_FakeSim(_camera(tmp_path, mount=mount, width=15, height=11), base_height=2.0))

    # Pixel (5, 7) is centred on the principal ray, which hits the ground at 2 / cos(30deg).
    assert depth[0, 5, 7].item() == pytest.approx(2.0 / math.cos(pitch), abs=1e-3)


def test_partial_rerender_updates_only_the_requested_envs(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    _, record, camera = _render(sim)
    sim.rigid_body_pos_w[:, 0, 2] = 1.5  # every env moved, but only env 2 is re-rendered

    camera.render(record, torch.tensor([2]))
    depth = record.buffers["depth"][..., 0]

    torch.testing.assert_close(depth[2], torch.full_like(depth[2], 1.5), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(depth[:2], torch.full_like(depth[:2], 2.0), atol=1e-4, rtol=0.0)


def test_a_partial_first_render_still_renders_every_env(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path), base_height=2.0)
    camera, record = _camera_and_record(sim)

    camera.render(record, torch.tensor([2]))
    depth = record.buffers["depth"][..., 0]

    torch.testing.assert_close(depth, torch.full_like(depth, 2.0), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("before_first_render", [True, False])
def test_mount_offsets_move_and_tilt_the_camera_per_env(tmp_path: Path, before_first_render: bool) -> None:
    sim = _FakeSim(_camera(tmp_path, width=15, height=11), base_height=2.0)
    camera, record = _camera_and_record(sim)
    position = torch.zeros(NUM_ENVS, 3)
    euler = torch.zeros(NUM_ENVS, 3)
    position[0, 2] = 0.1  # env 0 raised 10 cm
    euler[1, 0] = math.radians(20.0)  # env 1 tilted about the link's +X
    if not before_first_render:
        camera.render(record)

    camera.set_mount_offset(position, euler)
    camera.render(record)
    center = record.buffers["depth"][:, 5, 7, 0]

    assert center[0].item() == pytest.approx(2.1, abs=1e-4)
    assert center[1].item() == pytest.approx(2.0 / math.cos(math.radians(20.0)), abs=1e-3)
    assert center[2].item() == pytest.approx(2.0, abs=1e-4)


def test_mount_offsets_must_cover_every_env(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path))
    camera, _ = _camera_and_record(sim)

    with pytest.raises(ValueError, match="Mount offsets"):
        camera.set_mount_offset(torch.zeros(1, 3), torch.zeros(1, 3))


def test_initialization_binds_scene_geometry_once(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path))
    camera, record = _camera_and_record(sim)
    camera.initialize()
    camera.render(record)
    camera.render(record)
    assert sim.scene.initializations == 1


def test_render_requires_scene_initialization(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path))
    config = sim.sensor_config["depth"]
    record = SensorManager(device="cpu", control_hz=50.0).register_camera("depth", config)
    camera = RayCastDepthCamera(sim, config, sim.scene)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="initialized after scene loading"):
        camera.render(record)


def test_scene_mesh_pose_changes_are_used_without_reloading_geometry(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path))
    sim.rigid_body_pos_w[:, 1, 2] = 1.0
    _, record, camera = _render(sim)
    sim.rigid_body_pos_w[1, 1, 2] = 0.5
    camera.render(record, torch.tensor([1]))
    assert record.buffers["depth"][1, 6, 8, 0].item() == pytest.approx(1.3, abs=1e-4)
    assert record.buffers["depth"][0, 6, 8, 0].item() == pytest.approx(0.8, abs=1e-4)
    assert sim.scene.initializations == 1


def test_scene_mesh_ids_can_differ_between_environments(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path))
    tall = trimesh.creation.box(extents=[0.4, 0.4, 0.8])
    tall_mesh = convert_to_wp_mesh(tall.vertices, tall.faces, "cpu")
    # Keep the mesh alive and replace only env 1's occluder ID.
    ids = [[sim.scene.meshes[0].id, mesh.id] for mesh in (sim.scene.meshes[1], tall_mesh, sim.scene.meshes[1])]
    sim.scene.data = RaycastSceneMeshes(
        mesh_ids=wp.array(ids, dtype=wp.uint64, device="cpu"),
        positions=sim.scene.data.positions,
        orientations=sim.scene.data.orientations,
        num_meshes=2,
    )
    sim.rigid_body_pos_w[:, 1, 2] = 1.0
    depth, _, _ = _render(sim)
    assert depth[0, 6, 8].item() == pytest.approx(0.8, abs=1e-4)
    assert depth[1, 6, 8].item() == pytest.approx(0.6, abs=1e-4)


def test_scene_mesh_rotation_is_used_in_its_local_frame(tmp_path: Path) -> None:
    sim = _FakeSim(_camera(tmp_path, width=15, height=11))
    box = trimesh.creation.box(extents=[0.8, 0.4, 0.2])
    sim.scene = _FakeScene(sim, [convert_to_wp_mesh(box.vertices, box.faces, "cpu")])
    sim.rigid_body_pos_w[:, 1, 2] = 1.0
    # Rotating the long X axis onto Z raises the top surface from z=1.1 to z=1.4.
    sim.rigid_body_quat_w[0, 1] = torch.tensor([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    depth, _, _ = _render(sim)
    assert depth[0, 5, 7].item() == pytest.approx(0.6, abs=1e-4)
    assert depth[1, 5, 7].item() == pytest.approx(0.9, abs=1e-4)
