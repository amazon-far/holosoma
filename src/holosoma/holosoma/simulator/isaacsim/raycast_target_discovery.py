"""Pure scene target-list assembly and path selection for Isaac Sim ray-cast sensors."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from loguru import logger

from holosoma.config_types.sensor import IsaacSimRaycastMeshConfig


def _is_path_at_or_below(path: str, root: str) -> bool:
    """Return whether ``path`` is ``root`` or one of its descendants."""
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def assemble_default_mesh_prim_paths(
    *,
    excluded_target: tuple[str, str] | None,
    robot_mesh_prim_paths: Callable[[str | None], list[str]],
    actor_names: Iterable[str],
    discover_mesh_prim_paths: Callable[[str, str], Iterable[str]],
) -> list[str]:
    """Combine visual robot, ground, and scene-object targets after applying the body filter."""
    exclude_robot = excluded_target is not None and excluded_target[0] == "robot"
    excluded_robot_link = (
        excluded_target[1] if excluded_target is not None and excluded_target[0] == "robot_link" else None
    )
    paths = [] if exclude_robot else robot_mesh_prim_paths(excluded_robot_link)
    candidates = [("/World/ground", "/World/ground")]
    candidates.extend(
        (
            f"/World/envs/env_0/{actor_name}",
            f"/World/envs/env_.*/{actor_name}",
        )
        for actor_name in actor_names
        if excluded_target != ("actor", actor_name)
    )
    for source, expression in candidates:
        paths.extend(discover_mesh_prim_paths(source, expression))
    if not paths:
        if exclude_robot:
            raise RuntimeError(
                "Isaac Sim ray-cast auto-discovery found no supported visual scene meshes "
                "(body filter 'robot' excluded the robot; ground and scene actors contributed none)."
            )
        raise RuntimeError("Isaac Sim ray-cast auto-discovery found no supported visual scene meshes.")
    return paths


def select_mesh_prim_paths(
    mesh_paths: Iterable[str],
    config: IsaacSimRaycastMeshConfig,
    *,
    resolve_matching_paths: Callable[[str], Iterable[str]],
) -> list[str]:
    """Filter discovered meshes below roots resolved with IsaacLab's native path matcher.

    Discovery and configuration use LiDAR's cloned ``env_.*`` namespace. Resolve rules against
    env_0 once, retaining the discovered expressions for native per-environment mesh tracking.
    """
    paths = sorted(set(mesh_paths))

    def source_path(path: str) -> str:
        return path.replace("/World/envs/env_.*", "/World/envs/env_0", 1)

    def matches(expression: str) -> set[str]:
        roots = tuple(resolve_matching_paths(source_path(expression)))
        return {path for path in paths if any(_is_path_at_or_below(source_path(path), root) for root in roots)}

    included = set(paths)
    if config.include_paths is not None:
        included = set()
        for expression in config.include_paths:
            matched = matches(expression)
            if not matched:
                raise ValueError(f"Raycast mesh include {expression!r} matched no supported visual scene geometry.")
            included.update(matched)
    for expression in config.exclude_paths:
        matched = matches(expression)
        if not matched:
            # Allowed, so one config can exclude objects some scenes lack; a typo surfaces here.
            logger.warning(f"Raycast mesh exclude {expression!r} matched no supported visual scene geometry.")
        included.difference_update(matched)
    selected = sorted(included)
    if not selected:
        raise ValueError("Raycast mesh inclusion/exclusion selected no supported visual scene geometry.")
    return selected
