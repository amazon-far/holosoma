"""Standalone paced timing/quality controls; imports no simulator SDK.

All primary snapshot controls retain the SDK-like 1/60 native step. Elapsed
physics deltas are diagnostics. Native configured 10 Hz uses a stable 50 Hz
sensor clock. Wall deadlines pace the measured common timeline at 20 ms.
Only due owners receive current endpoint writes. Quality metrics and hashes
run after the paced timeline; copied observations retain independent shadows.
"""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ovrtx import Renderer, RendererConfig, Semantic, TextureStreamingMode

if TYPE_CHECKING:
    from .ovrtx_instances_repro import memory_sample as resource_snapshot
    from .ovrtx_instances_repro import pixels as read
    from .ovrtx_instances_repro import reuse_shader_cache as shader_cache
else:
    # Direct execution keeps this fixture usable outside the simulator package.
    from ovrtx_instances_repro import memory_sample as resource_snapshot
    from ovrtx_instances_repro import pixels as read
    from ovrtx_instances_repro import reuse_shader_cache as shader_cache


def pose(group, env):
    angle = group * math.pi / 2
    cosine, sine = round(math.cos(angle)), round(math.sin(angle))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = ((cosine, sine), (-sine, cosine))
    matrix[3, :3] = (env * 10, 0, 2)
    return matrix


def matrix_text(matrix):
    return "(" + ", ".join("(" + ", ".join(str(float(x)) for x in row) + ")" for row in matrix) + ")"


CONTROL_NAMES = ("native10", "manual10", "uniform10", "mixed", "continuous50")


def scene(resolution, instances, groups, native_schema=False):
    cameras = []
    products = []
    objects = [
        f"""
        def Cube "Panel{env}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
            double size = 0.12
            matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),({env * 10},0.08,0,1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            rel material:binding = </World/Material{env}>
        }}
        def Material "Material{env}" {{
            token outputs:surface.connect = </World/Material{env}/Surface.outputs:surface>
            def Shader "Surface" {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1,0,0)
                float inputs:roughness = 1
                token outputs:surface
            }}
        }}"""
        for env in range(instances)
    ]
    cols = math.ceil(math.sqrt(instances))
    rows = math.ceil(instances / cols)
    for group in groups:
        for env in range(instances):
            api = '(prepend apiSchemas = ["OmniSensorAPI"])' if native_schema else ""
            timing = (
                "double2 omni:sensor:frameRate = (50, 5)\ndouble2 omni:sensor:startTime = (0, 50)"
                if native_schema
                else ""
            )
            cameras.append(f"""
        def Camera "Camera{group}_{env}" {api} {{
            float focalLength = 24
            float horizontalAperture = 24
            float verticalAperture = 24
            float2 clippingRange = (0.01, 100)
            matrix4d xformOp:transform = {matrix_text(pose(group, env))}
            uniform token[] xformOpOrder = ["xformOp:transform"]
            {timing}
        }}""")
        camera_paths = ", ".join(f"</World/Camera{group}_{env}>" for env in range(instances))
        products.append(f"""
    def Scope "Render{group}" {{
        def RenderProduct "Product" (prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]) {{
            rel camera = [{camera_paths}]
            uint[] deviceIds = [0]
            uniform int2 resolution = ({cols * resolution}, {rows * resolution})
            token omni:rtx:rendermode = "RealTimePathTracing"
            bool omni:rtx:rt:ecoMode:enabled = false
            token omni:rtx:background:source:type = "color"
            color3f omni:rtx:background:source:color = (0,0,0)
            float omni:rtx:rt:ambientLight:intensity = 1
            token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
            rel orderedVars = [</Render{group}/Color>, </Render{group}/Depth>]
        }}
        def RenderVar "Color" {{
            uniform string sourceName = "LdrColor"
        }}
        def RenderVar "Depth" {{
            uniform string sourceName = "DistanceToImagePlaneSD"
        }}
    }}""")
    return (
        '#usda 1.0\n(defaultPrim="World"\nmetersPerUnit=1\nupAxis="Z")\n'
        'def Xform "World" {\n' + "\n".join(objects + cameras) + "\n}\n" + "\n".join(products)
    )


def target(cycle, env, warmup):
    if warmup:
        return (-0.12 if env % 2 == 0 else 0.12), 0.08 + env * 0.035, env * 0.02
    # Smooth movement with independent per-tile phases and varying depth.
    return (
        0.18 * math.sin(0.31 * cycle + env * 1.7),
        0.08 + env * 0.035,
        env * 0.02 + 0.012 * ((cycle + 3 * env) % 11),
    )


def colors(cycle, instances, warmup):
    channels = [0 if warmup else (2 + (cycle + 5 * env) // 20) % 3 for env in range(instances)]
    return np.eye(3, dtype=np.float32)[channels]


def dilation(mask, padding=1):
    padded = np.pad(mask, padding)
    return np.logical_or.reduce(
        [
            padded[row : row + mask.shape[0], col : col + mask.shape[1]]
            for row in range(2 * padding + 1)
            for col in range(2 * padding + 1)
        ]
    )


def metrics(rgb, depth, cycle, group, env, resolution, warmup, requested_color, previous=None):
    x, y, z = target(cycle, env, warmup)
    distance = 2 - z - 0.06
    angle = group * math.pi / 2
    local_x = math.cos(angle) * x + math.sin(angle) * y
    local_y = -math.sin(angle) * x + math.cos(angle) * y
    center = (resolution - 1) / 2
    expected = np.array([center - local_y * resolution / distance, center + local_x * resolution / distance])
    front = np.isfinite(depth) & (np.abs(depth - distance) < 0.006)
    surface = np.isfinite(depth) & (depth > 0) & (depth < 3)
    # Edge expansion excludes ordinary reconstruction along the current silhouette.
    outside = ~dilation(surface)
    interior = front & ~dilation(~front)
    if not interior.any():
        interior = front
    coordinates = np.argwhere(front)
    depth_error = float(np.max(np.abs(coordinates.mean(axis=0) - expected))) if coordinates.size else math.inf
    front_rgb = rgb[interior].astype(np.float64)
    mean_rgb = front_rgb.mean(axis=0) if front_rgb.size else np.full(3, np.nan)
    median_rgb = np.median(front_rgb, axis=0) if front_rgb.size else np.full(3, np.nan)
    requested_channel = int(np.argmax(requested_color))
    brightness = np.max(rgb, axis=-1).astype(np.float64)
    colored = brightness > 15
    ghost = outside & colored
    whole_energy = float(brightness.sum())
    ghost_energy = float(brightness[ghost].sum())
    record = {
        "cycle": cycle,
        "group": group,
        "env": env,
        "expected_center": expected.tolist(),
        "front_depth": distance,
        "depth_centroid_error": depth_error,
        "front_pixels": int(front.sum()),
        "current_surface_pixels": int(surface.sum()),
        "mean_front_rgb": mean_rgb.tolist(),
        "median_front_rgb": median_rgb.tolist(),
        "requested_color": requested_color.tolist(),
        "requested_channel": requested_channel,
        "desired_channel_fraction": float(mean_rgb[requested_channel] / max(float(mean_rgb.sum()), 1.0)),
        "other_channel_level": float((mean_rgb.sum() - mean_rgb[requested_channel]) / 2),
        "ghost_pixels_outside_current_surface": int(ghost.sum()),
        "ghost_energy_outside_current_surface": ghost_energy,
        "ghost_energy_fraction": ghost_energy / max(whole_energy, 1.0),
        "max_outside_signal": float(brightness[outside].max()) if outside.any() else 0.0,
        "rgb_current_surface_signal": float(np.mean(brightness[interior])) if interior.any() else 0.0,
    }
    if previous is not None:
        old_rgb, old_depth, old_mean, old_color = previous
        old_surface = np.isfinite(old_depth) & (old_depth > 0) & (old_depth < 3)
        new_surface = interior & ~dilation(old_surface)
        record.update(
            {
                "rgb_changed_from_previous_capture": not np.array_equal(old_rgb, rgb),
                "depth_changed_from_previous_capture": not np.array_equal(old_depth, depth),
                "new_surface_pixels": int(new_surface.sum()),
                "new_surface_mean_signal": float(brightness[new_surface].mean()) if new_surface.any() else None,
                "previous_rgb_signal_in_new_surface": float(np.max(old_rgb, axis=-1)[new_surface].mean())
                if new_surface.any()
                else None,
                "material_changed_from_previous_capture": not np.array_equal(requested_color, old_color),
                "desired_color_direction_progress": float(np.dot(mean_rgb - old_mean, requested_color - old_color)),
                "previous_capture_mean_rgb": old_mean.tolist(),
            }
        )
    return record, mean_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=CONTROL_NAMES, required=True)
    parser.add_argument("--clock-policy", choices=("fixed", "elapsed"), default="fixed")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-captures", type=int, default=10)
    parser.add_argument("--shader-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multi = args.control in ("uniform10", "mixed")
    owner_groups = [[0], [1]] if args.control == "mixed" else [[0, 1] if multi else [0]]
    cols = math.ceil(math.sqrt(args.instances))
    rows = math.ceil(args.instances / cols)
    renderers, submissions, quality, failures = [], [], [], []
    held, previous, observed_cycles, sampled_rgb, sampled_depth = {}, {}, [], [], []
    captured_arrays, held_events = [], []
    warmup_sampled_rgb, warmup_sampled_depth, warmup_quality = [], [], []
    counts = {str(group): 0 for group in range(2 if multi else 1)}
    warmup_counts = dict.fromkeys(counts, 0)
    metadata = {
        "control": args.control,
        "clock_policy": args.clock_policy,
        "clock_label": "native_sensor_50hz_elapsed"
        if args.control == "native10"
        else ("primary_sdk_fixed_1_60" if args.clock_policy == "fixed" else "diagnostic_elapsed_physics_time"),
        "ovrtx": importlib.metadata.version("ovrtx"),
        "numpy": np.__version__,
        "python": sys.version,
        "instances": args.instances,
        "resolution": args.resolution,
        "frames": args.frames,
        "warmup_slow_captures": args.warmup_captures,
        "physics_control_period_seconds": 0.02,
        "slow_decimation": 5,
        "slow_group_and_owner": 0,
        "render_mode": "RealTimePathTracing",
        "aa_override": None,
        "history_resets": 0,
        "prototype_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_material_api": "Renderer.write_attribute(inputs:diffuseColor,float32[N,3])",
        "current_time_api": "RenderProductSetOutputs.simulation_end_time",
        "pinned_temporal_clock_caveat": (
            "0.6 changed exposure/TAA/DLSS to simulation-step time; do not assume this in 0.5"
        ),
        "timeline_policy": (
            "Only due owners receive latest endpoint writes; metrics, hashes and held-copy validation run after pacing."
        ),
    }
    native_end = [0.0] * len(owner_groups)
    last_global_cycle = [None] * len(owner_groups)
    last_material_colors = [
        np.repeat(np.array([[1, 0, 0]], dtype=np.float32), args.instances, axis=0) for _ in owner_groups
    ]
    last_matrices = [None] * len(owner_groups)
    resource_samples = [resource_snapshot("before")]
    begin_startup = time.perf_counter()
    try:
        for owner, groups in enumerate(owner_groups):
            config = RendererConfig(
                active_cuda_gpus="0",
                log_file_path=str(args.output_dir / f"native-{owner}.log"),
                log_level="warn",
                read_gpu_transforms=True,
                keep_system_alive=True,
                suppress_deprecation_warnings=True,
                texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
            )
            if owner == 0:
                shader_cache(config, args.shader_cache)
            renderer = Renderer(config)
            renderers.append(renderer)
            text = scene(args.resolution, args.instances, groups, args.control == "native10")
            (args.output_dir / f"scene-{owner}.usda").write_text(text)
            renderer.open_usd_from_string(text)
        native_init_seconds = time.perf_counter() - begin_startup
        warmup_cycles = args.warmup_captures * 5
        test_wall_start = None
        material_history = []
        for global_cycle in range(warmup_cycles + args.frames):
            warmup = global_cycle < warmup_cycles
            cycle = global_cycle if warmup else global_cycle - warmup_cycles
            if not warmup:
                if test_wall_start is None:
                    test_wall_start = time.monotonic()
                    resource_samples.append(resource_snapshot("after_warmup"))
                    # Snapshot resource calls finish before defining the paced timeline.
                    test_wall_start = time.monotonic()
                deadline = test_wall_start + cycle * 0.02
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            requested_colors = colors(cycle, args.instances, warmup)
            matrices = np.repeat(np.eye(4, dtype=np.float64)[None], args.instances, axis=0)
            positions = []
            for env in range(args.instances):
                xyz = target(cycle, env, warmup)
                positions.append(list(xyz))
                matrices[env, 3, :3] = (env * 10 + xyz[0], xyz[1], xyz[2])
            if not warmup:
                material_history.append(
                    {
                        "cycle": cycle,
                        "physical_time": cycle * 0.02,
                        "positions": positions,
                        "colors": requested_colors.tolist(),
                    }
                )
            observed = {}
            for owner, (renderer, groups) in enumerate(zip(renderers, owner_groups, strict=False)):
                due_slow = cycle % 5 == 0
                if warmup:
                    selected = groups if due_slow or args.control == "native10" else []
                elif args.control in ("native10", "continuous50"):
                    selected = groups
                elif args.control == "mixed":
                    selected = [group for group in groups if group == 1 or due_slow]
                else:
                    selected = groups if due_slow else []
                if not selected:
                    continue
                write_start = time.monotonic()
                transform_write_seconds = 0.0
                material_write_seconds = 0.0
                if last_matrices[owner] is None or not np.array_equal(matrices, last_matrices[owner]):
                    attribute_start = time.monotonic()
                    renderer.write_attribute(
                        [f"/World/Panel{env}" for env in range(args.instances)],
                        "omni:xform",
                        matrices,
                        semantic=Semantic.XFORM_MAT4x4,
                    )
                    transform_write_seconds = time.monotonic() - attribute_start
                    last_matrices[owner] = matrices.copy()
                changed = np.flatnonzero(np.any(requested_colors != last_material_colors[owner], axis=1))
                if changed.size:
                    attribute_start = time.monotonic()
                    renderer.write_attribute(
                        [f"/World/Material{env}/Surface" for env in changed],
                        "inputs:diffuseColor",
                        requested_colors[changed],
                    )
                    material_write_seconds = time.monotonic() - attribute_start
                    last_material_colors[owner][changed] = requested_colors[changed]
                write_seconds = time.monotonic() - write_start
                if args.control == "native10":
                    delta = 0.1 + 2e-9 if last_global_cycle[owner] is None else 0.02
                elif args.clock_policy == "fixed":
                    delta = 1 / 60
                else:
                    delta = (
                        0.1 if last_global_cycle[owner] is None else (global_cycle - last_global_cycle[owner]) * 0.02
                    )
                last_global_cycle[owner] = global_cycle
                wall = time.monotonic()
                products = renderer.step({f"/Render{group}/Product" for group in selected}, delta_time=delta)
                step_end = time.monotonic()
                output_record = []
                for group in selected:
                    path = f"/Render{group}/Product"
                    frames = products[path].frames if path in products else []
                    expected = 1 if args.control != "native10" or due_slow else 0
                    if len(frames) != expected:
                        failures.append(
                            f"{'warmup' if warmup else 'test'} cycle{cycle} group{group}: "
                            f"frames{len(frames)} expected{expected}"
                        )
                    for frame in frames:
                        rgb = read(frame, f"/Render{group}/Color").reshape(
                            rows * args.resolution, cols * args.resolution, -1
                        )[..., :3]
                        depth = read(frame, f"/Render{group}/Depth").reshape(
                            rows * args.resolution, cols * args.resolution, -1
                        )[..., 0]
                        output_record.append(
                            {
                                "group": group,
                                "start_time": frame.start_time,
                                "end_time": frame.end_time,
                            }
                        )
                        if warmup:
                            warmup_counts[str(group)] += 1
                        else:
                            counts[str(group)] += 1
                        # Continuous reference renders intervening frames, but observations
                        # and held copies follow exactly the same slow10Hz schedule.
                        observation_due = group != 0 or due_slow
                        # CPU map read already copies native storage before the next
                        # submission. Independent shadows preserve an initial
                        # observation value for validation after the wall timeline.
                        capture_index = len(captured_arrays)
                        captured_arrays.append(
                            {
                                "phase": "warmup" if warmup else "test",
                                "cycle": cycle,
                                "group": group,
                                "observation_due": observation_due,
                                "rgb": rgb,
                                "depth": depth,
                                "rgb_shadow": rgb.copy() if observation_due else None,
                                "depth_shadow": depth.copy() if observation_due else None,
                                "colors": requested_colors.copy(),
                                "output_record": output_record[-1],
                            }
                        )
                        if observation_due:
                            observed[group] = (rgb, depth, capture_index)
                read_end = time.monotonic()
                native_start = float(products.simulation_start_time)
                native_stop = float(products.simulation_end_time)
                if abs(native_start - native_end[owner]) > 1e-8 or abs(native_stop - native_start - delta) > 1e-8:
                    failures.append(f"cycle{cycle} owner{owner}: unexpected native clock window")
                native_end[owner] = native_stop
                submissions.append(
                    {
                        "phase": "warmup" if warmup else "test",
                        "cycle": cycle,
                        "global_cycle": global_cycle,
                        "physical_time": cycle * 0.02,
                        "physical_global_time": global_cycle * 0.02,
                        "owner": owner,
                        "selected_groups": selected,
                        "delta_time": delta,
                        "simulation_start_time": native_start,
                        "current_time_after": native_stop,
                        "submit_monotonic": wall,
                        "submit_wall_seconds_from_test_start": None if warmup else wall - test_wall_start,
                        "deadline_lateness_seconds": None if warmup else wall - (test_wall_start + cycle * 0.02),
                        "scene_write_start_monotonic": write_start,
                        "scene_write_seconds": write_seconds,
                        "transform_write_seconds": transform_write_seconds,
                        "material_write_seconds": material_write_seconds,
                        "step_seconds": step_end - wall,
                        "step_plus_read_seconds": read_end - wall,
                        "read_copy_seconds": read_end - step_end,
                        "outputs": output_record,
                    }
                )
                del products
            held.update(observed)
            for group, captures in held.items():
                held_events.append(
                    {
                        "phase": "warmup" if warmup else "test",
                        "cycle": cycle,
                        "group": group,
                        "capture_index": captures[2],
                    }
                )
            if not warmup and cycle % 5 == 0 and 0 in observed:
                observed_cycles.append(cycle)
                sampled_rgb.append(observed[0][0])
                sampled_depth.append(observed[0][1])
            if warmup and cycle % 5 == 0 and 0 in observed:
                warmup_sampled_rgb.append(observed[0][0])
                warmup_sampled_depth.append(observed[0][1])
            if not warmup and cycle in (0, args.frames - 1):
                print("PROGRESS", cycle, counts, "hard_failures", len(failures), flush=True)
        test_seconds = time.monotonic() - test_wall_start
        resource_samples.append(resource_snapshot("end"))
        begin_analysis = time.monotonic()
        for capture in captured_arrays:
            rgb, depth = capture["rgb"], capture["depth"]
            capture["output_record"].update(
                {
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                }
            )
            if not capture["observation_due"]:
                continue
            cycle, group = capture["cycle"], capture["group"]
            warmup = capture["phase"] == "warmup"
            if not np.array_equal(rgb, capture["rgb_shadow"]) or not np.array_equal(depth, capture["depth_shadow"]):
                failures.append(f"{capture['phase']} cycle{cycle} group{group}: held copied observation changed")
            for env in range(args.instances):
                row, col = divmod(env, cols)
                region = np.s_[
                    row * args.resolution : (row + 1) * args.resolution,
                    col * args.resolution : (col + 1) * args.resolution,
                ]
                requested_color = capture["colors"][env]
                record, mean_rgb = metrics(
                    rgb[region],
                    depth[region],
                    cycle,
                    group,
                    env,
                    args.resolution,
                    warmup,
                    requested_color,
                    previous.get((group, env)),
                )
                record["phase"] = capture["phase"]
                (warmup_quality if warmup else quality).append(record)
                if not math.isfinite(record["depth_centroid_error"]) or record["depth_centroid_error"] > max(
                    1.6, args.resolution * 0.005
                ):
                    failures.append(
                        f"cycle{cycle} group{group} env{env}: "
                        f"depth geometry/orientation error{record['depth_centroid_error']}"
                    )
                if record["rgb_current_surface_signal"] < 15:
                    failures.append(f"cycle{cycle} group{group} env{env}: no RGB signal on current depth surface")
                new_signal = record.get("new_surface_mean_signal")
                prior_signal = record.get("previous_rgb_signal_in_new_surface")
                if record.get("new_surface_pixels", 0) > 5 and new_signal is not None and prior_signal is not None:
                    if new_signal <= 30 or new_signal <= prior_signal + 15:
                        failures.append(
                            f"cycle{cycle} group{group} env{env}: current RGB fails stale-frame negative control"
                        )
                previous[(group, env)] = (rgb[region], depth[region], mean_rgb.copy(), requested_color)
        for event in held_events:
            capture = captured_arrays[event["capture_index"]]
            if capture["group"] != event["group"] or capture["cycle"] > event["cycle"]:
                failures.append(
                    f"{event['phase']} cycle{event['cycle']} group{event['group']}: invalid held observation event"
                )
            if event["phase"] == capture["phase"] == "test" and event["group"] == 0:
                if capture["cycle"] != 5 * (event["cycle"] // 5):
                    failures.append(f"test cycle{event['cycle']} group0: held observation schedule changed")
        deferred_analysis_seconds = time.monotonic() - begin_analysis
        for group, count in warmup_counts.items():
            if count != args.warmup_captures:
                failures.append(f"group{group}: warmup captures{count} expected{args.warmup_captures}")
        np.savez_compressed(
            args.output_dir / "slow-observations.npz",
            cycles=np.array(observed_cycles),
            rgb=np.stack(sampled_rgb),
            depth=np.stack(sampled_depth),
            warmup_rgb=np.stack(warmup_sampled_rgb),
            warmup_depth=np.stack(warmup_sampled_depth),
        )
    finally:
        for renderer in reversed(renderers):
            renderer.destroy()
    bad_modules = [name for name in sys.modules if name.startswith(("holosoma", "isaaclab", "omni.kit", "omni.physx"))]
    assert not bad_modules, bad_modules
    slow_submits = [
        item for item in submissions if item["phase"] == "test" and any(out["group"] == 0 for out in item["outputs"])
    ]
    observation_submits = [item for item in slow_submits if item["cycle"] % 5 == 0]
    intervals = [b["submit_monotonic"] - a["submit_monotonic"] for a, b in itertools.pairwise(observation_submits)]
    result = {
        **metadata,
        "native_init_seconds": native_init_seconds,
        "paced_test_seconds": test_seconds,
        "native_capture_counts_by_group": counts,
        "warmup_capture_counts_by_group": warmup_counts,
        "slow_observation_count": len(observed_cycles),
        "observation_submit_intervals_seconds": intervals,
        "submissions": submissions,
        "quality": quality,
        "warmup_quality": warmup_quality,
        "failures": failures,
        "material_history": material_history,
        "resource_samples": resource_samples,
        "deferred_analysis_seconds": deferred_analysis_seconds,
        "held_observation_events": held_events,
        "quality_policy": (
            "Record desired-channel convergence and outside-depth ghost metrics without overriding defaults; "
            "hard failures cover counts, clock, geometry, current-surface RGB signal and held copies."
        ),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(
        "SUMMARY",
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("submissions", "quality", "warmup_quality", "material_history", "resource_samples")
            }
        ),
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    faulthandler.dump_traceback_later(120, repeat=True)
    raise SystemExit(main())
