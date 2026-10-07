"""Isaac Sim LiDAR target-list assembly tests."""

from __future__ import annotations

import pytest

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
