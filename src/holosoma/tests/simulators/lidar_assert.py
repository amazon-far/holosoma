# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Cross-backend live LiDAR assertion harness.

The harness checks the public output contract against a flat terrain and a moving cube. This
exercises native ray queries in MuJoCo, the shared terrain caster in Isaac Gym, and mutable
MultiMeshRayCaster directions in Isaac Sim.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from importlib.util import find_spec
from pathlib import Path

# Avoid shadowing the installed isaacsim package with tests/simulators/isaacsim.
if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from holosoma.simulator.base_simulator.hooks import Phase  # noqa: E402
from holosoma.utils.sim_utils import setup_simulation_environment  # noqa: E402
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step  # noqa: E402
from tests.simulators.lidar_figures import (  # noqa: E402
    save_pointcloud_figure,
    save_pointcloud_topdown_x_neg_z_projection,
    summarize_pointcloud,
)


def _expected_cyclic_directions(sim_time: float, ray_count: int, device):
    """Return the test fixture's expected 10 Hz window without using its pattern provider."""
    import torch

    azimuth = torch.linspace(-math.pi / 18.0, math.pi / 18.0, 128, device=device)
    sequence = torch.stack(
        [
            torch.sin(azimuth),
            torch.zeros_like(azimuth),
            -torch.cos(azimuth),
        ],
        dim=1,
    )
    snapshot_index = math.floor(sim_time * 10.0 + 1e-5)
    indices = torch.arange(ray_count, device=device)
    indices = torch.remainder(indices + snapshot_index * ray_count, len(sequence))
    return sequence.index_select(0, indices)


def _quat_mul_xyzw(left, right):
    """Compose XYZW quaternions with test-local Hamilton-product math."""
    import torch

    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _check_robot_link_cloud(sim, name: str, sensor_positions, sensor_orientations) -> list[str]:
    """Compare the mounted scan with the exact world-ground intersection."""
    import torch

    local_direction = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=sensor_positions.device)
    local_directions = local_direction.expand(sim.num_envs, -1)
    world_directions = _quat_apply_xyzw(sensor_orientations, local_directions)
    if torch.any(world_directions[:, 2] >= 0.0):
        return [f"{name}: mounted ray does not point toward the world ground"]
    expected_ranges = (sensor_positions[:, 2] / -world_directions[:, 2]).unsqueeze(1)
    expected_points = expected_ranges.unsqueeze(-1) * local_direction.view(1, 1, 3)
    ranges = sim.get_lidar_data(name, "ranges")
    points = sim.get_lidar_data(name, "points")
    failures = []
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3):
        failures.append(
            f"{name}: ranges do not match the current mounted-link pose and orientation "
            f"(actual={ranges[:, 0].tolist()}, expected={expected_ranges[:, 0].tolist()})"
        )
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3):
        failures.append(f"{name}: sensor-frame XYZ does not match the exact ground intersection")
    return failures


def _isaacgym_robot_link_sensor_pose(sim, manager, name: str, device):
    """Resolve one robot-link test sensor's current world pose through Isaac Gym's body cache."""
    import torch

    failures = []
    record = manager.get_lidar(name)
    cached_body_index = record.backend_cache["isaacgym_body_index"]
    body_index = sim._body_list.index(record.config.mount.target)
    if cached_body_index != body_index:
        failures.append(
            f"{name}: cached mount-body index does not identify the configured Isaac Gym link "
            f"(cached={cached_body_index}, expected={body_index})"
        )
    mount_offset = torch.tensor(
        record.config.mount.position,
        dtype=sim.rigid_body_pos_w.dtype,
        device=device,
    ).expand(sim.num_envs, -1)
    parent_orientations = sim.rigid_body_quat_w[:, body_index]
    sensor_positions = sim.rigid_body_pos_w[:, body_index] + _quat_apply_xyzw(
        parent_orientations,
        mount_offset,
    )
    mount_orientation_wxyz = torch.tensor(
        record.config.mount.orientation,
        dtype=parent_orientations.dtype,
        device=device,
    )
    mount_orientation_xyzw = mount_orientation_wxyz[[1, 2, 3, 0]].expand(sim.num_envs, -1)
    sensor_orientations = _quat_mul_xyzw(parent_orientations, mount_orientation_xyzw)
    return sensor_positions, sensor_orientations, failures


def _check_cloud(sim, name: str, expected_rays: int, sensor_heights, plane_normal) -> list[str]:
    import torch

    failures = []
    points = sim.get_lidar_data(name, "points")
    ranges = sim.get_lidar_data(name, "ranges")
    expected_points = (sim.num_envs, expected_rays, 3)
    expected_ranges = (sim.num_envs, expected_rays)
    if tuple(points.shape) != expected_points:
        failures.append(f"{name}: points shape {tuple(points.shape)} != {expected_points}")
    if tuple(ranges.shape) != expected_ranges:
        failures.append(f"{name}: ranges shape {tuple(ranges.shape)} != {expected_ranges}")
    if points.dtype != torch.float32 or ranges.dtype != torch.float32:
        failures.append(f"{name}: expected float32 outputs, got points={points.dtype}, ranges={ranges.dtype}")
    if failures:
        return failures

    finite = torch.isfinite(ranges)
    finite_count = int(finite.sum())
    if finite_count == 0:
        return [f"{name}: all {ranges.numel()} rays missed the flat terrain"]

    point_valid = torch.isfinite(points).all(dim=-1)
    if not torch.equal(point_valid, finite):
        failures.append(f"{name}: finite points do not match finite ranges")
    if finite.any():
        norms = torch.linalg.vector_norm(points[finite], dim=-1)
        if not torch.allclose(norms, ranges[finite], atol=2e-3, rtol=2e-3):
            failures.append(f"{name}: point norms do not match radial ranges")
        plane_coordinate = torch.einsum("nrc,c->nr", points, plane_normal)
        plane_error = torch.abs(plane_coordinate[finite] + sensor_heights[:, None].expand_as(finite)[finite])
        if float(plane_error.max()) > 0.05:
            failures.append(
                f"{name}: points do not terminate on the flat terrain in the rolled sensor frame "
                f"(max plane error {float(plane_error.max()):.4f} m)"
            )
    print(
        f"{name}: points={tuple(points.shape)} finite={finite_count}/{ranges.numel()} "
        f"range=[{float(ranges[finite].min()):.3f}, {float(ranges[finite].max()):.3f}]"
    )
    return failures


def _expected_static_returns(sensor_heights, plane_normal, directions, *, behavior: str):
    import torch

    unit_directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    world_down_component = -(unit_directions @ plane_normal)
    physical_ranges = sensor_heights[:, None] / world_down_component[None, :]
    physical = (world_down_component > 0.0)[None, :] & (physical_ranges >= 0.1) & (physical_ranges <= 10.0)
    physical = physical.expand(sensor_heights.shape[0], -1)
    sentinel_ranges = {"none": float("inf"), "max": 10.0, "zero": 0.0}
    expected_ranges = torch.where(
        physical,
        physical_ranges,
        torch.full_like(physical_ranges, sentinel_ranges[behavior]),
    )
    expected_points = expected_ranges.unsqueeze(-1) * unit_directions.unsqueeze(0)
    if behavior == "none":
        expected_points = torch.where(
            physical.unsqueeze(-1),
            expected_points,
            torch.full_like(expected_points, float("nan")),
        )
    return expected_ranges, expected_points, physical


def _check_static_cloud(sim, sensor_heights, plane_normal, directions) -> list[str]:
    return _check_known_plane_cloud(sim, "downward", sensor_heights, plane_normal, directions)


def _check_isaacsim_prim_resolution_helpers() -> list[str]:
    """Exercise link and native-owner resolution against independent USD fixtures."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from holosoma.simulator.isaacsim.prim_utils import (
        resolve_robot_link_prim_expression,
        resolve_robot_link_rigid_body_prim_expression,
        set_instanceable,
    )

    failures = []
    stage = Usd.Stage.CreateInMemory()
    robot_path = "/World/envs/env_0/Robot"
    UsdGeom.Xform.Define(stage, robot_path)

    owner = UsdGeom.Xform.Define(stage, f"{robot_path}/owner").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(owner).CreateRigidBodyEnabledAttr(True)
    fixed_link = UsdGeom.Xform.Define(stage, f"{robot_path}/owner/fixed_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(fixed_link).CreateRigidBodyEnabledAttr(False)
    expected_owner = "/World/envs/env_.*/Robot/owner"
    if resolve_robot_link_rigid_body_prim_expression("fixed_link", stage) != expected_owner:
        failures.append("disabled RigidBodyAPI did not resolve to its nearest enabled owner")

    geometry_link = UsdGeom.Cube.Define(stage, f"{robot_path}/geometry_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(geometry_link).CreateRigidBodyEnabledAttr(True)
    expected_geometry = "/World/envs/env_.*/Robot/geometry_link"
    if resolve_robot_link_prim_expression("geometry_link", stage) != expected_geometry:
        failures.append("a rigid body authored directly on geometry was not accepted as a robot link")

    UsdGeom.Xform.Define(stage, f"{robot_path}/owner/a/duplicate")
    UsdGeom.Xform.Define(stage, f"{robot_path}/owner/b/duplicate")
    try:
        resolve_robot_link_prim_expression("duplicate", stage)
    except ValueError:
        pass
    else:
        failures.append("ambiguous transformable robot-link names did not fail")

    UsdGeom.Scope.Define(stage, f"{robot_path}/scope_only")
    try:
        resolve_robot_link_prim_expression("scope_only", stage)
    except ValueError:
        pass
    else:
        failures.append("a non-transformable scope was accepted as a robot link")

    template = UsdGeom.Xform.Define(stage, "/Template").GetPrim()
    UsdGeom.Xform.Define(stage, "/Template/proxy_link")
    instance = UsdGeom.Xform.Define(stage, f"{robot_path}/instance").GetPrim()
    instance.GetReferences().AddInternalReference(template.GetPath())
    instance.SetInstanceable(True)
    proxy_path = f"{robot_path}/instance/proxy_link"
    if not stage.GetPrimAtPath(proxy_path).IsInstanceProxy():
        failures.append("USD instance fixture did not create an instance proxy")
    elif not set_instanceable(stage, robot_path, False):
        failures.append("robot deinstancing rejected a valid robot root")
    elif stage.GetPrimAtPath(proxy_path).IsInstanceProxy():
        failures.append("robot deinstancing left a nested instance proxy unresolved")
    elif resolve_robot_link_prim_expression("proxy_link", stage) != "/World/envs/env_.*/Robot/instance/proxy_link":
        failures.append("a deinstanced nested robot link did not resolve to its environment expression")

    return failures


def _check_isaacsim_native_configs(sim, manager) -> list[str]:
    """Verify Isaac Sim preserves shared ray shapes, cadence, and scene target coverage."""
    import isaaclab.sim as sim_utils

    from holosoma.simulator.isaacsim.prim_utils import (
        get_current_stage,
        is_rigid_body_enabled,
        resolve_robot_link_rigid_body_prim_expression,
    )
    from holosoma.simulator.isaacsim.sensor_setup import _is_visual_raycast_prim

    failures = []
    for lidar_record in manager.lidars:
        native = sim.scene.sensors.get(lidar_record.name)
        if native is None:
            failures.append(f"{lidar_record.name}: no native Isaac Sim ray caster was registered")
            continue
        if lidar_record.pattern_provider is None:
            failures.append(f"{lidar_record.name}: no shared LiDAR pattern provider was registered")
            continue
        if native.num_rays != lidar_record.pattern_provider.ray_count:
            failures.append(
                f"{lidar_record.name}: Isaac Sim ray caster has {native.num_rays} rays, "
                f"but its shared pattern provider has {lidar_record.pattern_provider.ray_count}"
            )
        if native.cfg.update_period != 0.0:
            failures.append(
                f"{lidar_record.name}: native update_period={native.cfg.update_period}, "
                "but the shared sensor manager must own capture cadence"
            )
        target_specs = [(target.prim_expr, target.track_mesh_transforms) for target in native.cfg.mesh_prim_paths]
        target_paths = {path for path, _track in target_specs}
        if lidar_record.config.isaacsim is None or lidar_record.config.isaacsim.mesh_prim_paths is None:
            ground_specs = [
                (path, track)
                for path, track in target_specs
                if path == "/World/ground" or path.startswith("/World/ground/")
            ]
            if not ground_specs:
                failures.append(f"{lidar_record.name}: auto-discovered targets omit the static ground")
            elif any(track for _path, track in ground_specs):
                failures.append(f"{lidar_record.name}: auto-discovered static ground tracks transforms")
            if "/World/envs/env_.*/.*" in target_paths:
                failures.append(f"{lidar_record.name}: auto-discovery retained the unsafe environment catch-all")
            body_filter = lidar_record.config.body_filter
            exclude_robot = body_filter.target_kind == "robot"
            excluded_target = None
            if body_filter.target_kind == "mount":
                mount = lidar_record.config.mount
                if mount.target_kind != "world":
                    excluded_target = (mount.target_kind, mount.target)
            elif body_filter.target_kind not in ("robot", "none"):
                excluded_target = (body_filter.target_kind, body_filter.target)
            excluded_path = None
            if excluded_target is not None:
                kind, target = excluded_target
                excluded_path = (
                    resolve_robot_link_rigid_body_prim_expression(target)
                    if kind == "robot_link"
                    else f"/World/envs/env_.*/{target}"
                )
            if excluded_path is not None:
                if excluded_target is not None and excluded_target[0] == "actor":
                    if excluded_path in target_paths:
                        failures.append(f"{lidar_record.name}: auto-discovery includes filtered actor {excluded_path}")
                else:
                    stage = get_current_stage()
                    excluded_source = excluded_path.replace("env_.*", "env_0")
                    for path in target_paths:
                        if not path.startswith("/World/envs/env_.*/Robot/"):
                            continue
                        owner = stage.GetPrimAtPath(path.replace("env_.*", "env_0"))
                        while owner.IsValid() and str(owner.GetPath()).startswith("/World/envs/env_0/Robot"):
                            if is_rigid_body_enabled(owner):
                                break
                            owner = owner.GetParent()
                        if owner.IsValid() and str(owner.GetPath()) == excluded_source:
                            failures.append(
                                f"{lidar_record.name}: auto-discovery includes geometry owned by filtered body "
                                f"{excluded_path}"
                            )
                            break
            stage = get_current_stage()
            source_root = "/World/envs/env_0/Robot"
            expression_root = "/World/envs/env_.*/Robot"
            expected_robot_targets = set()
            if not exclude_robot:
                for mesh_prim in sim_utils.get_all_matching_child_prims(
                    source_root,
                    _is_visual_raycast_prim,
                ):
                    owner = mesh_prim
                    while owner.IsValid() and str(owner.GetPath()).startswith(source_root):
                        if is_rigid_body_enabled(owner):
                            break
                        owner = owner.GetParent()
                    if excluded_path is not None and str(owner.GetPath()) == excluded_path.replace("env_.*", "env_0"):
                        continue
                    mesh_path = str(mesh_prim.GetPath())
                    expected_robot_targets.add(f"{expression_root}{mesh_path[len(source_root) :]}")
            actual_robot_targets = {path for path in target_paths if path.startswith("/World/envs/env_.*/Robot/")}
            if actual_robot_targets != expected_robot_targets:
                failures.append(
                    f"{lidar_record.name}: robot visual discovery differs from the eligible geometry "
                    f"(missing={sorted(expected_robot_targets - actual_robot_targets)}, "
                    f"extra={sorted(actual_robot_targets - expected_robot_targets)})"
                )
            for path, track_transforms in target_specs:
                if not (path == "/World/ground" or path.startswith("/World/ground/")) and not track_transforms:
                    failures.append(f"{lidar_record.name}: dynamic target {path} has transform tracking disabled")
            if not exclude_robot and not actual_robot_targets:
                failures.append(f"{lidar_record.name}: auto-discovery found no robot-link visual targets")
            expected_actors = {
                f"/World/envs/env_.*/{actor_name}"
                for actor_name in sim.scene.rigid_objects
                if excluded_target != ("actor", actor_name)
            }
            missing_actors = {
                actor_path
                for actor_path in expected_actors
                if not any(path == actor_path or path.startswith(f"{actor_path}/") for path in target_paths)
            }
            if missing_actors:
                failures.append(f"{lidar_record.name}: auto-discovery omits scene actors {sorted(missing_actors)}")
        else:
            expected_targets = set(lidar_record.config.isaacsim.mesh_prim_paths)
            if target_paths != expected_targets:
                failures.append(
                    f"{lidar_record.name}: explicit Isaac targets changed "
                    f"(actual={sorted(target_paths)}, expected={sorted(expected_targets)})"
                )
    return failures


def _check_known_plane_cloud(sim, name: str, sensor_heights, plane_normal, directions) -> list[str]:
    """Compare one ordered no-clipping scan against exact intersections with the test plane."""
    import torch

    points = sim.get_lidar_data(name, "points")
    ranges = sim.get_lidar_data(name, "ranges")
    expected_ranges, expected_points, physical = _expected_static_returns(
        sensor_heights,
        plane_normal,
        directions,
        behavior="none",
    )

    failures = _check_cloud(sim, name, directions.shape[0], sensor_heights, plane_normal)
    if not torch.equal(torch.isfinite(ranges), physical):
        failures.append(f"{name}: finite-return mask does not match exact terrain/range-cutoff expectation")
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3):
        range_errors = torch.abs(ranges - expected_ranges)
        flat_index = int(torch.nan_to_num(range_errors, nan=-1.0).argmax())
        env_index, ray_index = divmod(flat_index, ranges.shape[1])
        failures.append(
            f"{name}: ranges do not match exact intersections for all rays and environments "
            f"(env {env_index}, ray {ray_index}, actual={float(ranges[env_index, ray_index]):.6f}, "
            f"expected={float(expected_ranges[env_index, ray_index]):.6f})"
        )
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3, equal_nan=True):
        point_errors = torch.abs(points - expected_points)
        flat_index = int(torch.nan_to_num(point_errors, nan=-1.0).argmax())
        axis_index = flat_index % points.shape[2]
        env_ray_index = flat_index // points.shape[2]
        env_index, ray_index = divmod(env_ray_index, points.shape[1])
        failures.append(
            f"{name}: XYZ points do not match exact ordered sensor-frame intersections "
            f"(env {env_index}, ray {ray_index}, axis {axis_index}, "
            f"actual={float(points[env_index, ray_index, axis_index]):.6f}, "
            f"expected={float(expected_points[env_index, ray_index, axis_index]):.6f})"
        )
    return failures


def _check_static_range_clipping(
    sim,
    name: str,
    *,
    behavior: str,
    sensor_heights,
    plane_normal,
    directions,
) -> list[str]:
    import torch

    ranges = sim.get_lidar_data(name, "ranges")
    points = sim.get_lidar_data(name, "points")
    expected_ranges, expected_points, physical = _expected_static_returns(
        sensor_heights,
        plane_normal,
        directions,
        behavior=behavior,
    )
    failures = []
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3, equal_nan=True):
        failures.append(f"{name}: ranges do not match the configured {behavior!r} clipping convention")
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3, equal_nan=True):
        failures.append(f"{name}: points do not match the configured {behavior!r} clipping convention")
    synthetic = ~physical
    if behavior == "max" and not torch.all(ranges[synthetic] == 10.0):
        failures.append(f"{name}: synthetic endpoint ranges are not far")
    if behavior == "zero" and not torch.all(points[synthetic] == 0.0):
        failures.append(f"{name}: synthetic zero points are not at the sensor origin")
    return failures


def _check_mujoco_geometry_ids(sim, sensor_heights, plane_normal, directions) -> list[str]:
    """Verify that MuJoCo geometry IDs describe only physical returns."""
    import torch

    geom_ids = sim.get_lidar_data("downward", "geom_ids")
    _, _, physical = _expected_static_returns(
        sensor_heights,
        plane_normal,
        directions,
        behavior="none",
    )
    if geom_ids.shape != physical.shape:
        return [f"downward: geom_ids shape {tuple(geom_ids.shape)} does not match ranges {tuple(physical.shape)}"]
    failures = []
    if not torch.all(geom_ids[physical] >= 0):
        failures.append("downward: physical terrain returns are missing MuJoCo geometry IDs")
    if not torch.all(geom_ids[~physical] == -1):
        failures.append("downward: clipped or missed rays retain a MuJoCo geometry ID")
    return failures


def _save_snapshot_figures(points, *, output_dir: Path, simulator: str, step_index: int, num_envs: int) -> dict:
    step = f"{step_index:03d}"
    cloud_path = output_dir / f"{simulator}_lidar_step_{step}.png"
    topdown_path = output_dir / f"{simulator}_lidar_topdown_x_neg_z_step_{step}.png"
    title = f"{simulator} LiDAR snapshot {step_index}"
    summary = save_pointcloud_figure(points[0], cloud_path, title=title)
    topdown_summary = save_pointcloud_topdown_x_neg_z_projection(
        points[0],
        topdown_path,
        title=f"{title} top-down X/-Z projection",
    )
    if topdown_summary != summary:
        raise RuntimeError("3D and top-down X/-Z point-cloud figures consumed different point buffers.")
    return {
        "environment_summaries": [
            dataclasses.asdict(summarize_pointcloud(points[env_index])) for env_index in range(num_envs)
        ],
        "path": cloud_path.name,
        "topdown_x_neg_z_path": topdown_path.name,
        "topdown_x_neg_z_xyz_sha256": topdown_summary.xyz_sha256,
        **dataclasses.asdict(summary),
    }


def _check_tracked_target(sim, env_ids, actor_states, physics_per_control: int) -> list[str]:
    """Verify a known target-cube return before and after its deterministic lateral motion."""
    import torch

    before = sim.get_lidar_data("tracked_target", "ranges").clone()
    if before.shape != (sim.num_envs, 1):
        return [f"tracked_target: unexpected range shape {tuple(before.shape)}"]
    # camera-target's cube is 0.1 m tall with its origin at actor_states[:, 2]. World-mounted
    # sensors are local to each environment, so their z=2 m mount also includes env_origins.z.
    sensor_z = 2.0 + sim.env_origins[:, 2].to(actor_states.device, actor_states.dtype)
    expected_before = sensor_z - (actor_states[:, 2] + 0.05)
    if not torch.allclose(before[:, 0], expected_before, atol=2e-3, rtol=2e-3):
        return [
            "tracked_target: initial cube-top ranges do not match the known scene solution "
            f"(actual={before[:, 0].tolist()}, expected={expected_before.tolist()})"
        ]
    expected_ground = 2.0 + sim.env_origins[:, 2].to(before.device, before.dtype)
    filtered_before = sim.get_lidar_data("tracked_target_actor_filtered", "ranges")
    if not torch.allclose(filtered_before[:, 0], expected_ground, atol=2e-3, rtol=2e-3):
        return [
            "tracked_target_actor_filtered: filtering the near cube did not reveal the known ground "
            f"(actual={filtered_before[:, 0].tolist()}, expected={expected_ground.tolist()})"
        ]
    if sim.sensor_manager is not None and sim.sensor_manager.has_lidar("tracked_target_group_one"):
        grouped_before = sim.get_lidar_data("tracked_target_group_one", "ranges")
        if not torch.allclose(grouped_before[:, 0], expected_before, atol=2e-3, rtol=2e-3):
            return [
                "tracked_target_group_one: group-1 target range does not match the known scene solution "
                f"(actual={grouped_before[:, 0].tolist()}, expected={expected_before.tolist()})"
            ]

    moved = actor_states.clone()
    moved[:, 0] += 1.0
    moved[:, 7:] = 0.0
    sim.set_actor_states(["target"], env_ids, moved)
    sim.write_state_updates()
    for _ in range(physics_per_control):
        step(sim, 1)
    sim.hooks.emit(Phase.FRAME_END)

    after = sim.get_lidar_data("tracked_target", "ranges")
    expected_after = expected_ground.to(after.device, after.dtype)
    failures = []
    if not torch.allclose(after[:, 0], expected_after, atol=2e-3, rtol=2e-3):
        failures.append(
            "tracked_target: post-motion ground ranges do not match the known scene solution "
            f"(actual={after[:, 0].tolist()}, expected={expected_after.tolist()})"
        )
    if not torch.all(after[:, 0] > before[:, 0] + 0.2):
        failures.append("tracked_target: moving the cube did not replace its near return with the ground")
    filtered_after = sim.get_lidar_data("tracked_target_actor_filtered", "ranges")
    if not torch.allclose(filtered_after[:, 0], expected_after, atol=2e-3, rtol=2e-3):
        failures.append(
            "tracked_target_actor_filtered: filtered ground return changed after the excluded cube moved "
            f"(actual={filtered_after[:, 0].tolist()}, expected={expected_after.tolist()})"
        )
    if sim.sensor_manager is not None and sim.sensor_manager.has_lidar("tracked_target_group_one"):
        grouped_after = sim.get_lidar_data("tracked_target_group_one", "ranges")
        grouped_points = sim.get_lidar_data("tracked_target_group_one", "points")
        if not torch.isinf(grouped_after).all():
            failures.append("tracked_target_group_one: group-0 ground was not excluded after target motion")
        if not torch.isnan(grouped_points).all():
            failures.append("tracked_target_group_one: excluded group-0 ground produced finite XYZ")
    return failures


def _check_mjwarp_clock(sim) -> list[str]:
    """Verify that the public simulation clock tracks the native Warp clock and lifecycle resets."""
    import torch

    start = sim.time()
    step(sim, 1)
    advanced = sim.time()
    expected_advanced = start + sim.sim_dt
    failures = []
    if abs(advanced - expected_advanced) > 2e-6:
        failures.append(
            f"mjwarp: simulation clock advanced by {advanced - start:.8f} s, expected one sim_dt={sim.sim_dt:.8f} s"
        )

    # ``prepare_sim`` is intentionally a one-time scene-registration lifecycle. Reinitializing the
    # Warp backend is the supported path that refreshes its batched state from the current CPU data.
    assert sim.root_data is not None and sim.root_model is not None
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    actor_names = list(sim.object_registry.name_to_index)
    actor_states = sim.get_actor_states(actor_names, env_ids).clone()
    sim.root_data.time = 0.0
    sim.backend.initialize_state(sim.root_model, sim.root_data)
    sim.set_actor_states(actor_names, env_ids, actor_states)
    sim.write_state_updates()
    restored_states = sim.get_actor_states(actor_names, env_ids)
    if not torch.allclose(restored_states, actor_states, atol=2e-5, rtol=2e-5):
        failures.append("mjwarp: backend reinitialization did not restore every actor state")
    reset = sim.time()
    if abs(reset) > 2e-6:
        failures.append(f"mjwarp: simulation clock did not reset with backend reinitialization (time={reset:.8f} s)")
    return failures


def _check_actor_mount_origin_refresh(
    sim,
    env_ids,
    physics_per_control: int,
    *,
    actor_mount_height: float,
    plane_normal,
    directions,
) -> list[str]:
    """Move an actor-mounted scanner vertically and verify its new exact terrain intersections."""
    import torch

    before = sim.get_lidar_data("downward", "ranges").clone()
    moved = sim.get_actor_states(["target"], env_ids).clone()
    moved[:, 2] += 0.25
    moved[:, 7:] = 0.0
    sim.set_actor_states(["target"], env_ids, moved)
    sim.write_state_updates()
    for _ in range(physics_per_control):
        step(sim, 1)
    sim.hooks.emit(Phase.FRAME_END)

    ranges = sim.get_lidar_data("downward", "ranges")
    points = sim.get_lidar_data("downward", "points")
    sensor_heights = moved[:, 2] + actor_mount_height
    expected_ranges, expected_points, physical = _expected_static_returns(
        sensor_heights,
        plane_normal,
        directions,
        behavior="none",
    )
    failures = []
    if torch.allclose(ranges, before, atol=2e-3, rtol=2e-3, equal_nan=True):
        failures.append("downward: actor-mounted LiDAR ranges did not change after a vertical mount move")
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3):
        failures.append("downward: ranges did not refresh to exact intersections after a vertical mount move")
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3, equal_nan=True):
        failures.append("downward: XYZ points did not refresh after a vertical mount move")
    if not torch.equal(torch.isfinite(ranges), physical):
        failures.append("downward: finite-return mask did not refresh after a vertical mount move")
    return failures


def _quat_apply_xyzw(quaternions, vectors):
    """Rotate vector rows with test-local XYZW quaternion math."""
    import torch

    xyz = quaternions[:, :3]
    w = quaternions[:, 3:4]
    twice_cross = 2.0 * torch.cross(xyz, vectors, dim=-1)
    return vectors + w * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def _place_cyclic_target(sim, env_ids, direction, *, distance: float) -> float:
    """Center the test cube on an offset actor-mounted cyclic ray."""
    import torch

    from tests.simulators import _lidar_presets

    mount_states = sim.get_actor_states(["target"], env_ids)
    parent_quat = mount_states[:, 3:7]
    mount_position = torch.tensor(
        _lidar_presets.CYCLIC_TARGET_MOUNT_POSITION,
        dtype=direction.dtype,
        device=direction.device,
    ).expand_as(mount_states[:, :3])
    offset_wxyz = torch.tensor(
        _lidar_presets.CYCLIC_TARGET_MOUNT_ORIENTATION_WXYZ,
        dtype=direction.dtype,
        device=direction.device,
    )
    offset_quat = offset_wxyz[[1, 2, 3, 0]].expand_as(parent_quat)
    sensor_origin = mount_states[:, :3] + _quat_apply_xyzw(parent_quat, mount_position)
    sensor_directions = _quat_apply_xyzw(offset_quat, direction.unsqueeze(0).expand_as(sensor_origin))
    world_directions = _quat_apply_xyzw(parent_quat, sensor_directions)
    target_states = sim.get_actor_states(["scan_target"], env_ids).clone()
    target_states[:, :3] = sensor_origin + world_directions * distance
    target_states[:, 7:] = 0.0
    sim.set_actor_states(["scan_target"], env_ids, target_states)
    sim.write_state_updates()
    return distance - 0.05 / float(world_directions.abs().max())


def _check_cyclic_target_return(sim, direction, *, expected_range: float, snapshot: str) -> list[str]:
    """Assert the one-ray cyclic probe intersects the moved cube at the known range."""
    import torch

    points = sim.get_lidar_data("cyclic_target", "points")
    ranges = sim.get_lidar_data("cyclic_target", "ranges")
    expected_ranges = torch.full_like(ranges, expected_range)
    expected_points = direction.unsqueeze(0).unsqueeze(0) * expected_ranges.unsqueeze(-1)
    failures = []
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3):
        failures.append(f"cyclic_target {snapshot}: range does not match the analytic first-face intersection")
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3):
        max_error = float(torch.abs(points - expected_points).max())
        failures.append(
            f"cyclic_target {snapshot}: XYZ does not match the current ray and analytic cube intersection "
            f"(max error {max_error:.6f} m, actual={points[0, 0].tolist()}, "
            f"expected={expected_points[0, 0].tolist()})"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None)
    parser.add_argument(
        "--pointcloud-figure-dir",
        default=None,
        help="optionally save successive environment-0 LiDAR snapshots as 3D and top-down X/-Z PNG figures",
    )
    args = parser.parse_args()

    from tests.simulators import _camera_presets, _lidar_presets

    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    sensors = dict(_lidar_presets.LIDAR_RIG)
    if args.simulator in {"mujoco", "mjwarp"}:
        sensors["downward_filtered"] = _lidar_presets.downward_filtered_lidar
        sensors["tracked_target_group_one"] = _lidar_presets.tracked_target_group_one_lidar
    if args.simulator != "isaacgym":
        sensors["tracked_target"] = _lidar_presets.tracked_target_lidar
        sensors["tracked_target_actor_filtered"] = _lidar_presets.tracked_target_actor_filtered_lidar
    config = build_run_sim_config(
        sim_arg,
        "camera-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=sensors,
    )
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)
    device = "cpu" if args.simulator == "mujoco" else "cuda:0"
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    if args.simulator == "isaacsim":
        isaaclab_spec = find_spec("isaaclab")
        isaaclab_roots = () if isaaclab_spec is None else (isaaclab_spec.submodule_search_locations or ())
        has_multi_mesh_ray_caster = any(
            (Path(root) / "sensors" / "ray_caster" / "multi_mesh_ray_caster.py").is_file() for root in isaaclab_roots
        )
        if not has_multi_mesh_ray_caster:
            print("SKIP: Isaac Sim LiDAR requires IsaacLab 2.3.2 or newer")
            return 77

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    env_origins = torch.zeros(args.num_envs, 3, device=device)
    if args.num_envs > 1:
        env_origins[:, 0] = torch.arange(args.num_envs, device=device, dtype=torch.float32) * 5.0
        env_origins[:, 2] = torch.arange(args.num_envs, device=device, dtype=torch.float32) * 0.25
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(args.num_envs, env_origins, base_init)
    sim.prepare_sim()
    failures = _check_mjwarp_clock(sim) if args.simulator == "mjwarp" else []

    env_ids = torch.arange(args.num_envs, device=device)
    actor_states = sim.get_actor_states(["target"], env_ids).clone()
    actor_states[:, 7:] = 0.0
    sim.set_actor_states(["target"], env_ids, actor_states)
    sim.write_state_updates()

    manager = sim.sensor_manager
    if manager is None:
        failures.append("sensor manager was not created")
    else:
        if args.simulator == "isaacsim":
            failures += _check_isaacsim_prim_resolution_helpers()
            failures += _check_isaacsim_native_configs(sim, manager)
        target_record = None
        first_target_direction = None
        first_target_range = None
        if args.simulator != "isaacgym":
            target_record = manager.get_lidar("cyclic_target")
            first_target_direction = _expected_cyclic_directions(sim.time(), 1, device)[0]
            first_target_range = _place_cyclic_target(
                sim,
                env_ids,
                first_target_direction,
                distance=2.0,
            )
        sim.hooks.emit(Phase.FRAME_END)
        figure_summaries = []
        plane_normal = torch.tensor(_lidar_presets.PLANE_NORMAL_SENSOR, dtype=torch.float32, device=device)
        cyclic_plane_normal = torch.tensor(
            _lidar_presets.CYCLIC_PLANE_NORMAL_SENSOR,
            dtype=torch.float32,
            device=device,
        )
        sensor_heights = actor_states[:, 2] + _lidar_presets.ACTOR_MOUNT_HEIGHT
        if args.simulator == "isaacgym":
            robot_link_pose = _isaacgym_robot_link_sensor_pose(
                sim,
                manager,
                "robot_link",
                device,
            )
            robot_filtered_pose = _isaacgym_robot_link_sensor_pose(
                sim,
                manager,
                "robot_filtered",
                device=device,
            )
            failures.extend(robot_link_pose[2])
            failures.extend(robot_filtered_pose[2])
        else:
            robot_link_sensor_heights = sim.robot_root_states[:, 2] + _lidar_presets.ROBOT_LINK_MOUNT_HEIGHT
            robot_filtered_sensor_heights = sim.robot_root_states[:, 2] + _lidar_presets.ROBOT_FILTERED_MOUNT_HEIGHT
        static_directions = torch.tensor(
            _lidar_presets.DOWNWARD_DIRECTIONS,
            dtype=torch.float32,
            device=device,
        )
        cyclic_record = manager.get_lidar("cyclic")
        first_directions = cyclic_record.pattern_at(sim.time()).directions.clone()
        expected_first_directions = _expected_cyclic_directions(sim.time(), len(first_directions), device)
        if not torch.allclose(first_directions, expected_first_directions, atol=1e-6, rtol=1e-6):
            failures.append("cyclic: initial directions do not match the configured 10 Hz sequence window")
        first_cyclic_points = sim.get_lidar_data("cyclic", "points").clone()
        first_cyclic_ranges = sim.get_lidar_data("cyclic", "ranges").clone()
        first_cyclic_graph = None
        if args.simulator == "mjwarp":
            workspace = cyclic_record.backend_cache.get("mujoco_warp_rays")
            first_cyclic_graph = None if workspace is None else workspace.graph
            if first_cyclic_graph is None:
                failures.append("cyclic: MJWarp did not capture a reusable ray graph for the first snapshot")
        if args.pointcloud_figure_dir:
            figure_summaries.append(
                _save_snapshot_figures(
                    first_cyclic_points,
                    output_dir=Path(args.pointcloud_figure_dir),
                    simulator=args.simulator,
                    step_index=0,
                    num_envs=sim.num_envs,
                )
            )
        failures += _check_static_cloud(sim, sensor_heights, plane_normal, static_directions)
        if args.simulator == "isaacgym":
            failures += _check_robot_link_cloud(sim, "robot_link", robot_link_pose[0], robot_link_pose[1])
            failures += _check_robot_link_cloud(
                sim,
                "robot_filtered",
                robot_filtered_pose[0],
                robot_filtered_pose[1],
            )
        else:
            failures += _check_known_plane_cloud(
                sim,
                "robot_link",
                robot_link_sensor_heights,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
                torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
            )
            failures += _check_known_plane_cloud(
                sim,
                "robot_filtered",
                robot_filtered_sensor_heights,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
                torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
            )
        failures += _check_known_plane_cloud(
            sim,
            "cyclic",
            sensor_heights,
            cyclic_plane_normal,
            first_directions,
        )
        if first_target_direction is not None and first_target_range is not None:
            failures += _check_cyclic_target_return(
                sim,
                first_target_direction,
                expected_range=first_target_range,
                snapshot="initial",
            )
        if args.simulator in {"mujoco", "mjwarp"}:
            failures += _check_mujoco_geometry_ids(sim, sensor_heights, plane_normal, static_directions)
        physics_per_control = config.simulator.config.sim.control_decimation_steps
        if args.simulator != "isaacgym":
            failures += _check_tracked_target(sim, env_ids, actor_states, physics_per_control)
        failures += _check_static_range_clipping(
            sim,
            "downward_max",
            behavior="max",
            sensor_heights=sensor_heights,
            plane_normal=plane_normal,
            directions=static_directions,
        )
        failures += _check_static_range_clipping(
            sim,
            "downward_zero",
            behavior="zero",
            sensor_heights=sensor_heights,
            plane_normal=plane_normal,
            directions=static_directions,
        )
        if args.simulator in {"mujoco", "mjwarp"}:
            filtered_ranges = sim.get_lidar_data("downward_filtered", "ranges")
            filtered_points = sim.get_lidar_data("downward_filtered", "points")
            if not torch.isinf(filtered_ranges).all():
                failures.append("downward_filtered: disabled MuJoCo geom groups still returned geometry")
            if not torch.isnan(filtered_points).all():
                failures.append("downward_filtered: disabled MuJoCo geom groups produced finite XYZ")

        second_target_direction = None
        second_target_range = None
        captured_next_cyclic_window = False
        for _ in range(cyclic_record.effective_decimation):
            step(sim, physics_per_control)
            if target_record is not None:
                second_target_direction = _expected_cyclic_directions(sim.time(), 1, device)[0]
                second_target_range = _place_cyclic_target(
                    sim,
                    env_ids,
                    second_target_direction,
                    distance=2.0,
                )
            sim.hooks.emit(Phase.FRAME_END)
            if cyclic_record.name in manager.last_lidar_due:
                captured_next_cyclic_window = True
                break
        if not captured_next_cyclic_window:
            failures.append("cyclic: no due snapshot was captured within one configured measurement period")

        second_directions = cyclic_record.pattern_at(sim.time()).directions
        expected_second_directions = _expected_cyclic_directions(sim.time(), len(second_directions), device)
        if not torch.allclose(second_directions, expected_second_directions, atol=1e-6, rtol=1e-6):
            failures.append("cyclic: advanced directions do not match the configured 10 Hz sequence window")
        second_cyclic_points = sim.get_lidar_data("cyclic", "points")
        second_cyclic_ranges = sim.get_lidar_data("cyclic", "ranges")
        if args.simulator == "mjwarp":
            workspace = cyclic_record.backend_cache.get("mujoco_warp_rays")
            second_cyclic_graph = None if workspace is None else workspace.graph
            if second_cyclic_graph is None:
                failures.append("cyclic: MJWarp did not retain a reusable ray graph for the second snapshot")
            elif first_cyclic_graph is not None and second_cyclic_graph is not first_cyclic_graph:
                failures.append("cyclic: MJWarp replaced its ray graph instead of replaying it")
        if torch.equal(first_directions, second_directions):
            failures.append("cyclic: directions did not advance between simulation-time snapshots")
        if torch.allclose(first_cyclic_points, second_cyclic_points, equal_nan=True):
            failures.append("cyclic: point output buffer did not change after the next due cast")
        if torch.allclose(first_cyclic_ranges, second_cyclic_ranges, equal_nan=True):
            failures.append("cyclic: range output buffer did not change after the next due cast")
        failures += _check_known_plane_cloud(
            sim,
            "cyclic",
            sensor_heights,
            cyclic_plane_normal,
            second_directions,
        )
        if second_target_direction is not None and second_target_range is not None:
            failures += _check_cyclic_target_return(
                sim,
                second_target_direction,
                expected_range=second_target_range,
                snapshot="advanced",
            )
        if args.pointcloud_figure_dir:
            figure_summaries.append(
                _save_snapshot_figures(
                    second_cyclic_points,
                    output_dir=Path(args.pointcloud_figure_dir),
                    simulator=args.simulator,
                    step_index=1,
                    num_envs=sim.num_envs,
                )
            )
            manifest_path = Path(args.pointcloud_figure_dir) / f"{args.simulator}_lidar_figures.json"
            with manifest_path.open("w") as manifest:
                json.dump({"snapshots": figure_summaries}, manifest, indent=2, sort_keys=True)
        failures += _check_actor_mount_origin_refresh(
            sim,
            env_ids,
            physics_per_control,
            actor_mount_height=_lidar_presets.ACTOR_MOUNT_HEIGHT,
            plane_normal=plane_normal,
            directions=static_directions,
        )
    if args.result_file:
        with open(args.result_file, "w") as result:
            result.write("OK" if not failures else "FAIL\n" + "\n".join(failures))
    if failures:
        for failure in failures:
            print(f"[{args.simulator}] FAIL: {failure}")
        return 1
    print(f"[{args.simulator}] PASS: static and moving-object LiDAR snapshots satisfy the live contract")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
