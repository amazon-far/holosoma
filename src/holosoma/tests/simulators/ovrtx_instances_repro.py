"""Isolated OVRTX multi-instance probe; only OVRTX, NumPy and standard library."""

import argparse
import ctypes
import faulthandler
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ovrtx import Device, Renderer, RendererConfig, Semantic, TextureStreamingMode


def reuse_shader_cache(config, cache_path):
    """Apply the OVRTX native settings extension; no SDK runtime is involved."""
    from ovrtx._src import bindings

    class Settings(ctypes.Structure):
        _fields_ = [("apply_settings", ctypes.CFUNCTYPE(bindings.ovrtx_result_t, bindings.ovx_string_t))]

    loader = bindings._ovrtx_loader
    if loader._lib is None:
        loader.create_bindings(Renderer._to_c_config(config))
    lib = loader._lib
    lib.ovrtx_query_extension.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ovrtx_query_extension.restype = bindings.ovrtx_result_t
    pointer = ctypes.c_void_p()
    result = lib.ovrtx_query_extension(b"ovrtx.settings.apply_settings", ctypes.byref(pointer))
    assert result.status == bindings.OVRTX_API_SUCCESS
    settings = ctypes.cast(pointer, ctypes.POINTER(Settings)).contents
    for setting in ("driverShaderCachePath", "driverAppShaderCachePath"):
        result = settings.apply_settings(bindings.ovx_string_t(f"--/rtx/shaderDb/{setting}={cache_path}"))
        assert result.status == bindings.OVRTX_API_SUCCESS


def scene(decimations, mode, resolution, instances, owner=None):
    cameras = []
    products = []
    for index, decimation in enumerate(decimations):
        if owner is not None and index != owner:
            continue
        # Odd cameras rotate 90 degrees about their optical axis, providing
        # visibly different expected images at the same world position.
        angle = index * math.pi / 2
        c, s = round(math.cos(angle)), round(math.sin(angle))
        rotation = f"(({c}, {s}, 0, 0), ({-s}, {c}, 0, 0), (0, 0, 1, 0), (0, 0, 2, 1))"
        sensor_api = '(prepend apiSchemas = ["OmniSensorAPI"])' if mode not in ("auto", "manual", "serial") else ""
        timing = (
            f"double2 omni:sensor:frameRate = (50, {decimation})\n"
            f"double2 omni:sensor:startTime = ({max(decimations) - decimation}, 50)"
            if mode not in ("auto", "manual", "serial")
            else ""
        )
        camera_template = f"""
    def Camera "Camera{index}" {sensor_api} {{
        float focalLength = 24
        float horizontalAperture = 24
        float verticalAperture = 24
        float2 clippingRange = (0.01, 100)
        matrix4d xformOp:transform = {rotation}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        {timing}
    }}
"""
        cameras.extend(
            camera_template.replace(f'"Camera{index}"', f'"Camera{index}_{env}"').replace(
                "(0, 0, 2, 1)", f"({env * 10}, 0, 2, 1)"
            )
            for env in range(instances)
        )
        targets = ", ".join(f"</World/Camera{index}_{env}>" for env in range(instances))
        cols = math.ceil(math.sqrt(instances))
        rows = math.ceil(instances / cols)
        products.append(
            f"""
def Scope "Render{index}" {{
    def RenderProduct "Product" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{targets}]
        uint[] deviceIds = [0]
        uniform int2 resolution = ({resolution * cols}, {resolution * rows})
        token omni:rtx:rendermode = "RealTimePathTracing"
        bool omni:rtx:rt:ecoMode:enabled = false
        token omni:rtx:background:source:type = "color"
        color3f omni:rtx:background:source:color = (0, 0, 0)
        float omni:rtx:rt:ambientLight:intensity = 1
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render{index}/Color>, </Render{index}/Depth>]
    }}
    def RenderVar "Color" {{
        uniform string sourceName = "LdrColor"
    }}
    def RenderVar "Depth" {{
        uniform string sourceName = "DistanceToImagePlaneSD"
    }}
}}
"""
        )
    text = (
        """#usda 1.0
(defaultPrim = "World"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "World" {
PANEL_INSTANCES
    def Material "Red" {
        token outputs:surface.connect = </World/Red/Surface.outputs:surface>
        def Shader "Surface" {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1,0,0)
            float inputs:roughness = 1
            token outputs:surface
        }
    }
"""
        + "".join(cameras)
        + "}\n"
        + "".join(products)
    )
    panels = "\n".join(
        f"""    def Cube "Panel{env}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
        double size = 0.12
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),({env * 10 + 0.15},0.1,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </World/Red>
    }}"""
        for env in range(instances)
    )
    return text.replace("PANEL_INSTANCES", panels)


def pixels(frame, path):
    with frame.render_vars[path].map(device=Device.CPU) as mapping:
        mapping.wait()
        return np.from_dlpack(mapping).copy()


def memory_sample(phase):
    fields = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True
        )
        .strip()
        .split(",")
    )
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    status = Path("/proc/self/status").read_text()
    rss = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
    return {
        "phase": phase,
        "device_used_mib": int(fields[0]),
        "device_total_mib": int(fields[1]),
        "compute_processes": process,
        "rss_kib": int(rss.split()[1]),
    }


def check_tiles(rgb, depth, cycle, camera, xs, resolution, instances, camera_y=0.0):
    cols = math.ceil(math.sqrt(instances))
    rows = math.ceil(instances / cols)
    rgb = rgb.reshape(resolution * rows, resolution * cols, -1)[..., :3].astype(np.int16)
    depth = depth.reshape(resolution * rows, resolution * cols, -1)[..., 0]
    angle = camera * math.pi / 2
    c, s = round(math.cos(angle)), round(math.sin(angle))
    origin = (resolution - 1) / 2
    scale = resolution / 1.94
    tolerance = max(1.6, resolution * 0.005)
    failures, measurements = [], []
    for env in range(instances):
        row, col = divmod(env, cols)
        r = rgb[row * resolution : (row + 1) * resolution, col * resolution : (col + 1) * resolution]
        d = depth[row * resolution : (row + 1) * resolution, col * resolution : (col + 1) * resolution]
        panel = np.isfinite(d) & (np.abs(d - 1.94) < 0.03)
        red = (r[..., 0] > r[..., 1] + 30) & (r[..., 0] > r[..., 2] + 30) & panel
        dy = 0.1 - camera_y if camera == 0 and env == 0 else 0.1
        expected = [origin + (s * xs[env] - c * dy) * scale, origin + (c * xs[env] + s * dy) * scale]
        geometry_center = np.argwhere(panel).mean(axis=0) if panel.any() else np.full(2, np.nan)
        rgb_center = np.argwhere(red).mean(axis=0) if red.any() else np.full(2, np.nan)
        geometry_error = float(np.max(np.abs(geometry_center - expected)))
        rgb_error = float(np.max(np.abs(rgb_center - expected)))
        median_depth = float(np.median(d[red])) if red.any() else float("nan")
        measurement = {
            "cycle": cycle,
            "camera": camera,
            "env": env,
            "camera_y": camera_y if camera == 0 and env == 0 else 0.0,
            "rgb_center": rgb_center.tolist(),
            "depth_center": geometry_center.tolist(),
            "expected": expected,
            "rgb_error": rgb_error,
            "depth_error": geometry_error,
            "median_depth": median_depth,
        }
        measurements.append(measurement)
        if (
            not math.isfinite(rgb_error)
            or rgb_error > tolerance
            or not math.isfinite(geometry_error)
            or geometry_error > tolerance
            or abs(median_depth - 1.94) > 0.03
        ):
            failures.append(measurement)
    return failures, measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["normal", "serial", "instances"], required=True)
    parser.add_argument("--decimations", default="1,5")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--identical-instance-paths",
        action="store_true",
        help="Give independent renderer scenes identical camera/product prim paths, as SDK children do.",
    )
    parser.add_argument(
        "--defer-prime",
        action="store_true",
        help="Create/open every renderer before priming, as normal SDK camera initialization does.",
    )
    parser.add_argument(
        "--serial-single-prime",
        action="store_true",
        help="Prime serialized products individually, matching production membership history.",
    )
    parser.add_argument("--shader-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.frames:
        parser.error("--warmup must be nonnegative and less than --frames.")
    if args.instances <= 0 or args.resolution <= 0:
        parser.error("--instances and --resolution must be positive.")
    try:
        rates = [int(value) for value in args.decimations.split(",")]
    except ValueError:
        parser.error("--decimations must be comma-separated positive integers.")
    if any(rate <= 0 for rate in rates):
        parser.error("--decimations must be comma-separated positive integers.")
    assert not args.identical_instance_paths or args.strategy == "instances"
    assert not args.serial_single_prime or args.strategy == "serial"

    def namespace(camera):
        return 0 if args.identical_instance_paths else camera

    def prime_batches(products):
        return [{product} for product in sorted(products)] if args.serial_single_prime else [products]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ovrtx": importlib.metadata.version("ovrtx"),
        "numpy": np.__version__,
        "strategy": args.strategy,
        "decimations": rates,
        "instances": args.instances,
        "resolution": args.resolution,
        "frames": args.frames,
        "warmup": args.warmup,
        "identical_instance_paths": args.identical_instance_paths,
        "defer_prime": args.defer_prime,
        "serial_single_prime": args.serial_single_prime,
    }
    print("START", json.dumps(metadata), flush=True)
    renderers, creation_times, memories = [], [], [memory_sample("before")]
    begin = time.perf_counter()
    failures, measurements, step_seconds, copy_seconds, update_seconds = [], [], [], [], []
    counts = [0] * len(rates)
    measured_counts = [0] * len(rates)
    held, held_hashes, startup_native_steps = {}, {}, 0
    exit_code = 0
    try:
        for owner in range(len(rates) if args.strategy == "instances" else 1):
            config = RendererConfig(
                active_cuda_gpus="0",
                log_file_path=str(args.output_dir / f"native-{owner}.log"),
                log_level="warn",
                read_gpu_transforms=True,
                keep_system_alive=True,
                suppress_deprecation_warnings=True,
                texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
            )
            if not renderers:
                reuse_shader_cache(config, args.shader_cache)
            started = time.perf_counter()
            print("CREATE_BEGIN", owner, flush=True)
            renderer = Renderer(config)
            renderers.append(renderer)
            print("CREATE_END", owner, flush=True)
            usd = scene(
                rates, "serial", args.resolution, args.instances, owner=owner if args.strategy == "instances" else None
            )
            if args.identical_instance_paths:
                usd = usd.replace(f"Camera{owner}", "Camera0").replace(f"Render{owner}", "Render0")
            (args.output_dir / f"scene-{owner}.usda").write_text(usd)
            renderer.open_usd_from_string(usd)
            print("OPEN_END", owner, flush=True)
            if not args.defer_prime:
                prime_products = (
                    {f"/Render{namespace(owner)}/Product"}
                    if args.strategy == "instances"
                    else {f"/Render{i}/Product" for i in range(len(rates))}
                )
                for products in prime_batches(prime_products):
                    prime = renderer.step(render_products=products, delta_time=1)
                    for path in prime:
                        for frame in prime[path].frames:
                            startup_native_steps += 1
                            camera = int(path.split("/")[1].removeprefix("Render"))
                            pixels(frame, f"/Render{camera}/Color")
                            pixels(frame, f"/Render{camera}/Depth")
                    del prime
            creation_times.append(time.perf_counter() - started)
            memories.append(memory_sample(f"initialized-{owner}"))
            print("INITIALIZED", owner, "PRIMED", not args.defer_prime, flush=True)
        if args.defer_prime:
            for owner, renderer in enumerate(renderers):
                prime_products = (
                    {f"/Render{namespace(owner)}/Product"}
                    if args.strategy == "instances"
                    else {f"/Render{i}/Product" for i in range(len(rates))}
                )
                for products in prime_batches(prime_products):
                    prime = renderer.step(render_products=products, delta_time=1)
                    for path in prime:
                        for frame in prime[path].frames:
                            startup_native_steps += 1
                            camera = int(path.split("/")[1].removeprefix("Render"))
                            pixels(frame, f"/Render{camera}/Color")
                            pixels(frame, f"/Render{camera}/Depth")
                    del prime
                print("DELAYED_PRIME", owner, flush=True)
        startup_seconds = time.perf_counter() - begin
        print("READY", startup_seconds, flush=True)
        trajectory = np.array([-0.18, 0.18, -0.06, 0.12, -0.12, 0.06, 0.0])
        for cycle in range(args.frames):
            # Each tile has its own phase, exposing tile/instance substitutions.
            phase = np.arange(args.instances) * 3
            if cycle >= args.frames // 2:
                # Partial environment reset changes only tile 0's moving-target phase.
                phase[0] += 2
            xs = trajectory[(cycle + phase) % len(trajectory)]
            matrix = np.repeat(np.eye(4, dtype=np.float64)[None], args.instances, axis=0)
            matrix[:, 3, :3] = np.column_stack(
                (xs + np.arange(args.instances) * 10, np.full(args.instances, 0.1), np.zeros(args.instances))
            )
            started = time.perf_counter()
            for renderer in renderers:
                renderer.write_attribute(
                    [f"/World/Panel{env}" for env in range(args.instances)],
                    "omni:xform",
                    matrix,
                    semantic=Semantic.XFORM_MAT4x4,
                )
            # Reset only one camera/environment pose, then restore it. Other tiles,
            # camera clocks and retained observations continue without resets.
            camera_y = 0.04 if args.frames // 4 <= cycle < args.frames // 2 else 0.0
            camera_matrix = np.eye(4, dtype=np.float64)[None]
            camera_matrix[0, 3, :3] = [0.0, camera_y, 2.0]
            renderers[0].write_attribute(
                ["/World/Camera0_0"], "omni:xform", camera_matrix, semantic=Semantic.XFORM_MAT4x4
            )
            if cycle >= args.warmup:
                update_seconds.append(time.perf_counter() - started)
            due = [i for i, rate in enumerate(rates) if cycle % rate == 0]
            requests = (
                [(None, renderers[0], {f"/Render{i}/Product" for i in range(len(rates))})]
                if args.strategy == "normal"
                else [
                    (
                        i,
                        renderers[i] if args.strategy == "instances" else renderers[0],
                        {f"/Render{namespace(i)}/Product"},
                    )
                    for i in due
                ]
            )
            for owner, renderer, paths in requests:
                started = time.perf_counter()
                outputs = renderer.step(render_products=paths, delta_time=1)
                if cycle >= args.warmup:
                    step_seconds.append(time.perf_counter() - started)
                for path in paths:
                    camera = owner if args.strategy == "instances" else int(path.split("/")[1].removeprefix("Render"))
                    frames = outputs[path].frames if path in outputs else []
                    counts[camera] += len(frames)
                    if cycle >= args.warmup:
                        measured_counts[camera] += len(frames)
                    if len(frames) != 1:
                        failures.append({"cycle": cycle, "camera": camera, "frames": len(frames), "expected_frames": 1})
                    for frame in frames:
                        started = time.perf_counter()
                        rgb = pixels(frame, f"/Render{namespace(camera)}/Color")
                        depth = pixels(frame, f"/Render{namespace(camera)}/Depth")
                        if cycle >= args.warmup:
                            copy_seconds.append(time.perf_counter() - started)
                        errors, details = check_tiles(
                            rgb, depth, cycle, camera, xs, args.resolution, args.instances, camera_y
                        )
                        failures.extend(errors)
                        measurements.extend(details)
                        # Native handles are consumed before another step; retained observations own copies.
                        if camera in due:
                            held[camera] = (rgb.copy(), depth.copy())
                            held_hashes[camera] = (hash(rgb.tobytes()), hash(depth.tobytes()))
                del outputs
            for camera, buffers in held.items():
                if camera not in due:
                    assert (hash(buffers[0].tobytes()), hash(buffers[1].tobytes())) == held_hashes[camera]
            if cycle == args.warmup - 1:
                memories.append(memory_sample("after-warmup"))
            if failures:
                print("FAILURE_COUNT", cycle, len(failures), flush=True)
            if cycle % 10 == 0:
                print("CYCLE", cycle, "COUNTS", counts, flush=True)
        memories.append(memory_sample("finished"))
        report = {
            **metadata,
            "counts": counts,
            "measured_counts": measured_counts,
            "partial_camera_pose_reset_cycles": [args.frames // 4, args.frames // 2],
            "partial_target_reset_cycle": args.frames // 2,
            "measurements": measurements,
            "failures": failures,
            "startup_seconds": startup_seconds,
            "per_renderer_startup_seconds": creation_times,
            "startup_native_frames": startup_native_steps,
            "measured_cycles": max(0, args.frames - args.warmup),
            "native_capture_seconds": sum(step_seconds),
            "cpu_readback_seconds": sum(copy_seconds),
            "transform_update_seconds": sum(update_seconds),
            "memory": memories,
        }
        report["native_capture_ms_per_cycle"] = 1000 * report["native_capture_seconds"] / report["measured_cycles"]
        report["capture_plus_readback_ms_per_cycle"] = (
            1000 * (report["native_capture_seconds"] + report["cpu_readback_seconds"]) / report["measured_cycles"]
        )
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            "SUMMARY",
            json.dumps({k: v for k, v in report.items() if k not in ("measurements", "failures")}),
            flush=True,
        )
        exit_code = int(bool(failures))
    finally:
        for renderer in reversed(renderers):
            renderer.destroy()
    imports = [name for name in sys.modules if name.startswith(("holosoma", "isaaclab", "omni.kit"))]
    assert not imports, imports
    return exit_code


if __name__ == "__main__":
    faulthandler.dump_traceback_later(30, repeat=True)
    raise SystemExit(main())
