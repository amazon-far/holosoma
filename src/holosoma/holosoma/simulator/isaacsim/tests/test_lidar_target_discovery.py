"""Isaac Sim LiDAR target-list assembly tests."""

from __future__ import annotations

import re

import pytest

from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend
from holosoma.simulator.isaacsim.lidar_target_discovery import (
    _is_path_at_or_below,
    assemble_default_mesh_prim_paths,
)

pytestmark = pytest.mark.no_sim


def test_path_boundary_does_not_match_prefix_siblings() -> None:
    root = "/World/envs/env_0/Robot"

    assert _is_path_at_or_below(root, root)
    assert _is_path_at_or_below(f"{root}/torso/visual", root)
    assert not _is_path_at_or_below("/World/envs/env_0/RobotArm/visual", root)


@pytest.mark.parametrize("env_regex_ns", ["/World/envs/env_.*", "/World/envs/env_[^/]+"])
def test_robot_filter_omits_robot_discovery_but_retains_ground_and_scene_actors(env_regex_ns: str) -> None:
    def unexpected_robot_discovery(_excluded_link: str | None) -> list[str]:
        raise AssertionError("whole-robot filter still enumerated robot mesh targets")

    paths = assemble_default_mesh_prim_paths(
        excluded_target=("robot", ""),
        robot_mesh_prim_paths=unexpected_robot_discovery,
        actor_names=["box", "robot_decoy"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
        env_regex_ns=env_regex_ns,
    )

    assert paths == [
        "/World/ground/visual",
        f"{env_regex_ns}/box/visual",
        f"{env_regex_ns}/robot_decoy/visual",
    ]


@pytest.mark.parametrize("env_regex_ns", ["/World/envs/env_.*", "/World/envs/env_[^/]+"])
def test_robot_link_filter_keeps_existing_single_owner_discovery(env_regex_ns: str) -> None:
    exclusions = []

    def robot_discovery(excluded_link: str | None) -> list[str]:
        exclusions.append(excluded_link)
        return [f"{env_regex_ns}/Robot/torso/visual"]

    paths = assemble_default_mesh_prim_paths(
        excluded_target=("robot_link", "head_link"),
        robot_mesh_prim_paths=robot_discovery,
        actor_names=["box"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
        env_regex_ns=env_regex_ns,
    )

    assert exclusions == ["head_link"]
    assert paths == [
        f"{env_regex_ns}/Robot/torso/visual",
        "/World/ground/visual",
        f"{env_regex_ns}/box/visual",
    ]


@pytest.mark.parametrize("env_regex_ns", ["/World/envs/env_.*", "/World/envs/env_[^/]+"])
def test_actor_filter_excludes_only_the_selected_scene_actor(env_regex_ns: str) -> None:
    paths = assemble_default_mesh_prim_paths(
        excluded_target=("actor", "shell"),
        robot_mesh_prim_paths=lambda _excluded_link: [f"{env_regex_ns}/Robot/visual"],
        actor_names=["shell", "target"],
        discover_mesh_prim_paths=lambda _source, expression: [f"{expression}/visual"],
        env_regex_ns=env_regex_ns,
    )

    assert paths == [
        f"{env_regex_ns}/Robot/visual",
        "/World/ground/visual",
        f"{env_regex_ns}/target/visual",
    ]


def test_robot_filter_empty_scene_names_the_filter_in_the_error() -> None:
    with pytest.raises(RuntimeError, match=r"body filter 'robot' excluded the robot"):
        assemble_default_mesh_prim_paths(
            excluded_target=("robot", ""),
            robot_mesh_prim_paths=lambda _excluded_link: [],
            actor_names=[],
            discover_mesh_prim_paths=lambda _source, _expression: [],
        )


def test_native_actor_expression_excludes_same_named_descendants_of_other_actors() -> None:
    """Match external stage paths with the SDK's whole-path regex semantics."""
    stage_meshes = [
        "/World/envs/env_0/box/visual",
        "/World/envs/env_1/box/visual",
        "/World/envs/env_0/freebox/Geometry/box/visual",
    ]

    def discover(_source: str, expression: str) -> list[str]:
        return [path for path in stage_meshes if re.fullmatch(f"{expression}/visual", path)]

    paths = assemble_default_mesh_prim_paths(
        excluded_target=("robot", ""),
        robot_mesh_prim_paths=lambda _excluded_link: [],
        actor_names=["box"],
        discover_mesh_prim_paths=discover,
        env_regex_ns=IsaacLab3Backend().env_regex_ns,
    )
    assert paths == ["/World/envs/env_0/box/visual", "/World/envs/env_1/box/visual"]
