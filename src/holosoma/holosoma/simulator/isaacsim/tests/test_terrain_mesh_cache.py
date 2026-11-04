"""Managed terrain shares its existing Warp allocation across native cache lookup paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest

from holosoma.simulator.isaacsim.terrain_mesh_cache import (
    TERRAIN_MESH_PRIM_PATH,
    TERRAIN_PRIM_PATH,
    register_managed_terrain_mesh,
)

pytestmark = pytest.mark.no_sim

if TYPE_CHECKING:
    import warp as wp
else:
    wp = pytest.importorskip("warp")
from holosoma.utils.warp_utils import convert_to_wp_mesh  # noqa: E402


@pytest.fixture
def terrain_mesh() -> wp.Mesh:
    return convert_to_wp_mesh(
        np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
        np.array([[0, 1, 2]]),
        "cpu",
    )


def test_cache_borrows_one_mesh_without_building_another(
    terrain_mesh: wp.Mesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_new_mesh(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Registering managed terrain must not allocate another Warp mesh/BVH.")

    monkeypatch.setattr(wp, "Mesh", reject_new_mesh)
    cache = {"/World/unrelated": terrain_mesh}
    register_managed_terrain_mesh(cache, terrain_mesh, wp.get_device("cpu"))
    register_managed_terrain_mesh(cache, terrain_mesh, wp.get_device("cpu"))

    assert cache[TERRAIN_PRIM_PATH] is terrain_mesh
    assert cache[TERRAIN_MESH_PRIM_PATH] is terrain_mesh
    assert cache["/World/unrelated"] is terrain_mesh


@pytest.mark.parametrize("conflict_path", [TERRAIN_PRIM_PATH, TERRAIN_MESH_PRIM_PATH])
def test_conflicting_cache_entry_fails_without_partial_registration(terrain_mesh: wp.Mesh, conflict_path: str) -> None:
    cache = {conflict_path: Mock(spec=wp.Mesh)}
    before = cache.copy()
    with pytest.raises(RuntimeError, match="different Warp mesh"):
        register_managed_terrain_mesh(cache, terrain_mesh, wp.get_device("cpu"))
    assert cache == before


def test_device_mismatch_fails_before_cache_registration(terrain_mesh: wp.Mesh) -> None:
    other_device = Mock(spec=wp.Device)
    cache: dict[str, wp.Mesh] = {}
    with pytest.raises(ValueError, match="Warp mesh is on"):
        register_managed_terrain_mesh(cache, terrain_mesh, other_device)
    assert not cache
