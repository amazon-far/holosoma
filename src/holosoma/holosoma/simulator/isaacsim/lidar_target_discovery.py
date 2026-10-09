"""Pure target-list assembly for Isaac Sim LiDAR auto-discovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable


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
                "Isaac Sim LiDAR auto-discovery found no supported visual scene meshes "
                "(body filter 'robot' excluded the robot; ground and scene actors contributed none)."
            )
        raise RuntimeError("Isaac Sim LiDAR auto-discovery found no supported visual scene meshes.")
    return paths
