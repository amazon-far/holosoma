"""Isaac-SDK-free ray-cast target helpers: target assembly, selection, robot mesh poses, terrain cache."""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from holosoma.config_types.sensor import IsaacSimRaycastMeshConfig
from holosoma.simulator.isaacsim.raycast_targets import (
    TERRAIN_MESH_PRIM_PATH,
    TERRAIN_PRIM_PATH,
    RobotMeshPoses,
    _is_path_at_or_below,
    assemble_default_mesh_prim_paths,
    register_managed_terrain_mesh,
    select_mesh_prim_paths,
)

pytestmark = pytest.mark.no_sim

if TYPE_CHECKING:
    import warp as wp
else:
    wp = pytest.importorskip("warp")
from holosoma.utils.warp_utils import convert_to_wp_mesh  # noqa: E402


# --- Target-list assembly


def test_path_boundary_does_not_match_prefix_siblings() -> None:
    root = "/World/envs/env_0/Robot"

    assert _is_path_at_or_below(root, root)
    assert _is_path_at_or_below(f"{root}/torso/visual", root)
    assert not _is_path_at_or_below("/World/envs/env_0/RobotArm/visual", root)


def test_robot_filter_omits_robot_discovery_but_retains_ground_and_scene_actors() -> None:
    def unexpected_robot_discovery(_excluded_link: str | None) -> list[str]:
        raise AssertionError("whole-robot filter still enumerated robot mesh targets")

    paths = assemble_default_mesh_prim_paths(
        excluded_target=("robot", ""),
        robot_mesh_prim_paths=unexpected_robot_discovery,
        actor_names=["box", "robot_decoy"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
    )

    assert paths == [
        "/World/ground/visual",
        "/World/envs/env_.*/box/visual",
        "/World/envs/env_.*/robot_decoy/visual",
    ]


def test_robot_link_filter_keeps_existing_single_owner_discovery() -> None:
    exclusions = []

    def robot_discovery(excluded_link: str | None) -> list[str]:
        exclusions.append(excluded_link)
        return ["/World/envs/env_.*/Robot/torso/visual"]

    paths = assemble_default_mesh_prim_paths(
        excluded_target=("robot_link", "head_link"),
        robot_mesh_prim_paths=robot_discovery,
        actor_names=["box"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
    )

    assert exclusions == ["head_link"]
    assert paths == [
        "/World/envs/env_.*/Robot/torso/visual",
        "/World/ground/visual",
        "/World/envs/env_.*/box/visual",
    ]


def test_actor_filter_excludes_only_the_selected_scene_actor() -> None:
    paths = assemble_default_mesh_prim_paths(
        excluded_target=("actor", "shell"),
        robot_mesh_prim_paths=lambda _excluded_link: ["/World/envs/env_.*/Robot/visual"],
        actor_names=["shell", "target"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
    )

    assert paths == [
        "/World/envs/env_.*/Robot/visual",
        "/World/ground/visual",
        "/World/envs/env_.*/target/visual",
    ]


def test_robot_filter_empty_scene_names_the_filter_in_the_error() -> None:
    with pytest.raises(RuntimeError, match=r"body filter 'robot' excluded the robot"):
        assemble_default_mesh_prim_paths(
            excluded_target=("robot", ""),
            robot_mesh_prim_paths=lambda _excluded_link: [],
            actor_names=[],
            discover_mesh_prim_paths=lambda _source, _expression: [],
        )


# --- Include/exclude selection


_ROBOT = "/World/envs/env_.*/Robot"
_PATHS = [
    "/World/ground/mesh",
    f"{_ROBOT}/dummy/torso_link/visuals/housing",
    f"{_ROBOT}/dummy/left_wrist/visuals/hand",
    "/World/envs/env_.*/box/visuals/mesh",
    "/World/envs/env_.*/box_extra/visuals/mesh",
]
_SOURCE_ROBOT = "/World/envs/env_0/Robot"
_ROOTS = {
    "/World": ["/World"],
    "/World/envs/env_0/Robot": [_SOURCE_ROBOT],
    "/World/envs/env_0/Robot/.*/torso_link": [f"{_SOURCE_ROBOT}/dummy/torso_link"],
    "/World/envs/env_0/box": ["/World/envs/env_0/box"],
    "/World/envs/env_0/box/visuals/mesh": ["/World/envs/env_0/box/visuals/mesh"],
    "/World/envs/env_0/missing_optional_object": [],
    "/World/envs/env_0/missing": [],
    "/World/ground": ["/World/ground"],
}


def _select(config: IsaacSimRaycastMeshConfig, paths: list[str] = _PATHS) -> list[str]:
    return select_mesh_prim_paths(paths, config, resolve_matching_paths=_ROOTS.__getitem__)


def test_default_keeps_all_discovered_geometry_and_deduplicates() -> None:
    resolver = Mock(side_effect=AssertionError("default discovery must not query extra roots"))
    assert select_mesh_prim_paths(
        _PATHS + _PATHS, IsaacSimRaycastMeshConfig(), resolve_matching_paths=resolver
    ) == sorted(_PATHS)
    resolver.assert_not_called()


def test_include_object_root_expands_to_exact_mesh_paths() -> None:
    selection = IsaacSimRaycastMeshConfig(include_paths=[_ROBOT])
    resolver = Mock(return_value=[_SOURCE_ROBOT])
    assert select_mesh_prim_paths(_PATHS, selection, resolve_matching_paths=resolver) == sorted(_PATHS[1:3])
    resolver.assert_called_once_with(_SOURCE_ROBOT)


def test_exclusion_overrides_inclusion_and_omits_descendants() -> None:
    selection = IsaacSimRaycastMeshConfig(
        include_paths=["/World"],
        exclude_paths=[f"{_ROBOT}/.*/torso_link", "/World/envs/env_.*/box"],
    )
    assert _select(selection) == sorted([_PATHS[0], _PATHS[2], _PATHS[4]])


def test_exact_mesh_include_and_optional_exclude() -> None:
    selection = IsaacSimRaycastMeshConfig(
        include_paths=["/World/envs/env_.*/box/visuals/mesh"],
        exclude_paths=["/World/envs/env_.*/missing_optional_object"],
    )
    assert _select(selection) == [_PATHS[3]]


@pytest.mark.parametrize(
    ("include", "match"),
    [
        ("/World/envs/env_.*/missing", "missing.*matched no"),  # no such prim
        ("/World/lights", "lights.*matched no"),  # prim exists, but has no supported geometry
    ],
)
def test_each_include_must_match_geometry(include: str, match: str) -> None:
    roots = {**_ROOTS, "/World/lights": ["/World/lights"]}
    selection = IsaacSimRaycastMeshConfig(include_paths=["/World/ground", include])
    with pytest.raises(ValueError, match=match):
        select_mesh_prim_paths(_PATHS, selection, resolve_matching_paths=roots.__getitem__)


def test_empty_selection_fails() -> None:
    with pytest.raises(ValueError, match="selected no"):
        _select(IsaacSimRaycastMeshConfig(exclude_paths=["/World"]))


def test_native_component_expression_can_resolve_multiple_roots() -> None:
    selection = IsaacSimRaycastMeshConfig(include_paths=["/World/envs/env_.*/box(_extra)?"])
    resolver = Mock(return_value=["/World/envs/env_0/box", "/World/envs/env_0/box_extra"])
    assert select_mesh_prim_paths(_PATHS, selection, resolve_matching_paths=resolver) == _PATHS[3:]
    resolver.assert_called_once_with("/World/envs/env_0/box(_extra)?")


# --- Robot mesh poses


def test_offsets_duplicate_body_bindings_and_unbound_slots() -> None:
    # Body 1 rotates +90 degrees around Z; two different meshes belong to it.
    sim = SimpleNamespace(
        rigid_body_pos_w=torch.tensor([[[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]]),
        rigid_body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, sqrt(0.5), sqrt(0.5)]]]),
    )
    binding = RobotMeshPoses(
        target_paths=frozenset({"mesh_a", "mesh_b", "mesh_c"}),
        body_ids=torch.tensor([1, 0, 1]),
        mesh_slots=torch.tensor([1, 3, 4]),
        position_offsets=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]]),
        orientation_offsets=torch.tensor(
            [[[sqrt(0.5), 0.0, 0.0, sqrt(0.5)], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
    )
    positions = torch.full((1, 5, 3), -7.0)
    orientations = torch.full((1, 5, 4), -9.0)
    position_storage, orientation_storage = positions.data_ptr(), orientations.data_ptr()

    binding.update(sim, positions, orientations)  # type: ignore[arg-type]

    torch.testing.assert_close(
        positions[:, [1, 3, 4]], torch.tensor([[[0.0, 21.0, 0.0], [10.0, 2.0, 0.0], [-1.0, 20.0, 0.0]]])
    )
    # Rz(90) * Rx(90), returned in native wxyz order.
    torch.testing.assert_close(orientations[:, 1], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    torch.testing.assert_close(orientations[:, 3], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(orientations[:, 4], torch.tensor([[sqrt(0.5), 0.0, 0.0, sqrt(0.5)]]))
    assert torch.all(positions[:, [0, 2]] == -7)
    assert torch.all(orientations[:, [0, 2]] == -9)
    assert positions.data_ptr() == position_storage
    assert orientations.data_ptr() == orientation_storage

    # refresh_sim_tensors replaces body tensors; cached bindings must reread them.
    sim.rigid_body_pos_w = sim.rigid_body_pos_w + 100.0
    binding.update(sim, positions, orientations)  # type: ignore[arg-type]
    torch.testing.assert_close(positions[:, 1], torch.tensor([[100.0, 121.0, 100.0]]))


def test_each_environment_uses_its_own_offset_and_body_state() -> None:
    sim = SimpleNamespace(
        rigid_body_pos_w=torch.tensor([[[1.0, 2.0, 3.0]], [[11.0, 12.0, 13.0]]]),
        rigid_body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 0.0]]]),
    )
    binding = RobotMeshPoses(
        target_paths=frozenset({"mesh"}),
        body_ids=torch.tensor([0]),
        mesh_slots=torch.tensor([0]),
        position_offsets=torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]),
        orientation_offsets=torch.tensor([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]]),
    )
    positions, orientations = torch.zeros(2, 1, 3), torch.zeros(2, 1, 4)

    binding.update(sim, positions, orientations)  # type: ignore[arg-type]

    torch.testing.assert_close(positions, torch.tensor([[[2.0, 2.0, 3.0]], [[9.0, 12.0, 13.0]]]))
    torch.testing.assert_close(orientations, torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]]))


# --- Terrain mesh cache


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
    cache: dict[str, wp.Mesh] = {}
    register_managed_terrain_mesh(cache, terrain_mesh, wp.get_device("cpu"))
    register_managed_terrain_mesh(cache, terrain_mesh, wp.get_device("cpu"))

    assert cache[TERRAIN_PRIM_PATH] is terrain_mesh
    assert cache[TERRAIN_MESH_PRIM_PATH] is terrain_mesh


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
