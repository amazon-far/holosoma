"""Contact-group forces in Classic and the production Warp kernel on CPU."""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.backends.classic_backend import ContactPairForces  # noqa: E402
from holosoma.simulator.mujoco.contact_warp import _group_contact_forces  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("normal_only", [True, False])
def test_group_kernel_preserves_pair_direction_world_ids_and_active_capacity(normal_only: bool) -> None:
    # Two active rigid contacts, one flex contact, and one inactive pool entry.
    geoms = np.array([[0, 1], [1, 2], [-1, 1], [0, 1]], dtype=np.int32)
    frames = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]] * 4, dtype=np.float32)
    wrenches = np.array([[2, 0, 10, 0, 0, 0], [0, 3, 20, 0, 0, 0], [0, 0, 999, 0, 0, 0]] * 2, dtype=np.float32)[:4]
    sources = np.eye(3, dtype=np.int32)
    counterparts = np.eye(3, dtype=np.int32)
    with wp.ScopedDevice("cpu"):
        output = wp.zeros((2, 3, 3, 3), dtype=wp.float32)
        wp.launch(
            _group_contact_forces,
            dim=(4, 3, 3),
            inputs=[
                wp.array([3], dtype=wp.int32),
                wp.array([0, 1, 0, 0], dtype=wp.int32),
                wp.array(geoms, dtype=wp.vec2i),
                wp.array(frames, dtype=wp.mat33),
                wp.array(wrenches, dtype=wp.spatial_vector),
                wp.array(sources, dtype=wp.int32),
                wp.array(counterparts, dtype=wp.int32),
                normal_only,
            ],
            outputs=[output],
        )
        actual = output.numpy()
    expected = np.zeros((2, 3, 3, 3), dtype=np.float32)
    first = np.array([0 if normal_only else 2, 0, 10])
    second = np.array([0, 0 if normal_only else 3, 20])
    expected[0, 0, 1] = -first
    expected[0, 1, 0] = first
    expected[1, 1, 2] = -second
    expected[1, 2, 1] = second
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reader", ["classic", "warp"])
@pytest.mark.parametrize("normal_only", [True, False])
@pytest.mark.parametrize("filtered", [True, False])
def test_grouped_forces_preserve_rotated_normals_overlap_and_internal_cancellation(
    reader: str, normal_only: bool, filtered: bool
) -> None:
    # Geoms 0 and 1 share a source group. Geom 3 is outside every explicit group.
    geoms = np.array([[0, 2], [1, 2], [0, 1], [2, 3]], dtype=np.int32)
    frames = np.array(
        [
            [[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ],
        dtype=np.float32,
    )
    total = np.array([[6, 8, 4], [-4, 3, 2], [1, 2, 3], [0, 0, 99]], dtype=np.float32)
    normal = np.array([[6, 8, 0], [0, 0, 2], [1, 0, 0], [0, 0, 99]], dtype=np.float32)
    sources = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]], dtype=np.int32)
    counterparts = np.array([[0, 0, 1, 0], [1, 1, 0, 0]], dtype=np.int32) if filtered else np.ones((1, 4), np.int32)
    first, second, internal, outside = normal if normal_only else total
    expected = np.zeros((1, 4, len(counterparts), 3), dtype=np.float32)
    if filtered:
        expected[0, 0, 0] = -(first + second)
        expected[0, 1, 0] = -first
        expected[0, 1, 1] = -internal
        expected[0, 2, 1] = first + second
        expected[0, 3, 0] = -(first + second)
        expected[0, 3, 1] = first + second
    else:
        expected[0, 0, 0] = -(first + second)
        expected[0, 1, 0] = -(first + internal)
        expected[0, 2, 0] = first + second - outside
        expected[0, 3, 0] = -outside

    if reader == "classic":
        pairs = ContactPairForces(geoms, normal, total, torch.zeros(1, 1, 3))
        actual = pairs.matrix(sources, counterparts, force_mode="normal" if normal_only else "total").numpy()
    else:
        wrenches = np.concatenate((total, np.zeros_like(total)), axis=1)
        with wp.ScopedDevice("cpu"):
            output = wp.zeros(expected.shape, dtype=wp.float32)
            wp.launch(
                _group_contact_forces,
                dim=(len(geoms), len(sources), len(counterparts)),
                inputs=[
                    wp.array([len(geoms)], dtype=wp.int32),
                    wp.array(np.zeros(len(geoms), dtype=np.int32), dtype=wp.int32),
                    wp.array(geoms, dtype=wp.vec2i),
                    wp.array(frames, dtype=wp.mat33),
                    wp.array(wrenches, dtype=wp.spatial_vector),
                    wp.array(sources, dtype=wp.int32),
                    wp.array(counterparts, dtype=wp.int32),
                    normal_only,
                ],
                outputs=[output],
            )
            actual = output.numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
