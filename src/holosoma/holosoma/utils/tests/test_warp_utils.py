from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from holosoma.utils import warp_utils
from holosoma.utils.warp_interop import from_torch_vec3

pytestmark = [
    pytest.mark.mujoco_warp,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp Torch interop tests require CUDA"),
]

_DEVICE = "cuda:0"


@wp.kernel
def _translate_points(points: wp.array(dtype=wp.vec3, ndim=2)) -> None:  # type: ignore[valid-type]
    env_id, point_id = wp.tid()
    points[env_id, point_id] += wp.vec3(1.0, 2.0, 3.0)


@pytest.fixture
def square_mesh() -> wp.Mesh:
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return warp_utils.convert_to_wp_mesh(vertices, triangles, _DEVICE)


def test_torch_vec3_array_aliases_cuda_storage() -> None:
    points = torch.zeros((2, 2, 3), dtype=torch.float32, device=_DEVICE)

    points_wp = from_torch_vec3(points, wp.get_device(_DEVICE))

    assert points_wp.ptr == points.data_ptr()
    assert points_wp.shape == (2, 2)
    assert points_wp.dtype == wp.vec3
    wp.launch(_translate_points, dim=(2, 2), inputs=[points_wp], device=_DEVICE)
    wp.synchronize()
    torch.testing.assert_close(points, torch.tensor([[[1.0, 2.0, 3.0]] * 2] * 2, device=_DEVICE))


def test_ray_cast_restores_shape_and_preserves_hits_and_misses(square_mesh: wp.Mesh) -> None:
    ray_starts = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [0.5, -0.5, 3.0]],
            [[2.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
        ],
        dtype=torch.float32,
        device=_DEVICE,
    )
    ray_directions = torch.tensor(
        [
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        ],
        dtype=torch.float32,
        device=_DEVICE,
    )

    hits = warp_utils.ray_cast(ray_starts, ray_directions, square_mesh)

    assert hits.shape == ray_starts.shape
    torch.testing.assert_close(hits[0, 0], torch.tensor([0.0, 0.0, 0.0], device=_DEVICE))
    torch.testing.assert_close(hits[0, 1], torch.tensor([0.5, -0.5, 0.0], device=_DEVICE))
    assert torch.isinf(hits[1]).all()


def test_nearest_point_restores_shape_and_preserves_hits_and_misses(square_mesh: wp.Mesh) -> None:
    points = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [0.25, -0.5, -3.0]],
            [[2.0, 0.0, 1.0], [0.0, 0.0, 2_000_000.0]],
        ],
        dtype=torch.float32,
        device=_DEVICE,
    )

    nearest = warp_utils.nearest_point(points, square_mesh)

    assert nearest.shape == points.shape
    finite = torch.isfinite(nearest).all(dim=-1)
    assert finite.tolist() == [[True, True], [True, False]]
    expected_hits = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, -0.5, 0.0], [1.0, 0.0, 0.0]],
        device=_DEVICE,
    )
    torch.testing.assert_close(nearest[finite], expected_hits)
    assert torch.isinf(nearest[1, 1]).all()
