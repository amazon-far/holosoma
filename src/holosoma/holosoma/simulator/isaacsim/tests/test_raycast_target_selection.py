"""Scene-object and mesh path inclusion/exclusion without an Isaac Sim dependency."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from holosoma.config_types.sensor import IsaacSimRaycastMeshConfig
from holosoma.simulator.isaacsim.raycast_target_discovery import select_mesh_prim_paths

pytestmark = pytest.mark.no_sim

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


def test_each_include_must_match_geometry() -> None:
    selection = IsaacSimRaycastMeshConfig(include_paths=["/World/ground", "/World/envs/env_.*/missing"])
    with pytest.raises(ValueError, match="missing.*matched no"):
        _select(selection)


def test_empty_selection_fails() -> None:
    with pytest.raises(ValueError, match="selected no"):
        _select(IsaacSimRaycastMeshConfig(exclude_paths=["/World"]))


def test_native_component_expression_can_resolve_multiple_roots() -> None:
    selection = IsaacSimRaycastMeshConfig(include_paths=["/World/envs/env_.*/box(_extra)?"])
    resolver = Mock(return_value=["/World/envs/env_0/box", "/World/envs/env_0/box_extra"])
    assert select_mesh_prim_paths(_PATHS, selection, resolve_matching_paths=resolver) == _PATHS[3:]
    resolver.assert_called_once_with("/World/envs/env_0/box(_extra)?")


def test_existing_root_without_supported_geometry_is_not_a_valid_include() -> None:
    selection = IsaacSimRaycastMeshConfig(include_paths=["/World/lights"])
    with pytest.raises(ValueError, match="lights.*matched no"):
        select_mesh_prim_paths(_PATHS, selection, resolve_matching_paths=lambda _: ["/World/lights"])


@pytest.mark.parametrize(
    "path", ["", "box", "/World/[", " /World/ground", "/World/ground ", "{ENV_REGEX_NS}/Robot"]
)
def test_invalid_path_rules_fail_at_config_construction(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        IsaacSimRaycastMeshConfig(include_paths=[path])
    with pytest.raises(ValueError, match="path"):
        IsaacSimRaycastMeshConfig(exclude_paths=[path])


def test_empty_include_list_fails_at_config_construction() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        IsaacSimRaycastMeshConfig(include_paths=[])
