"""Standalone OVRTX camera-cadence probe: no Holosoma, Isaac Lab, Kit or physics.

Run in the pinned OVRTX environment:
  python ovrtx_cadence_repro.py --mode periodic --decimations 1,5,2
  python ovrtx_cadence_repro.py --mode subset --decimations 1,5,2
"""

import argparse
import ctypes
import faulthandler
import importlib.metadata
import json
import math
import sys
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


def scene(decimations, mode, resolution, instances):
    cameras = []
    products = []
    for index, decimation in enumerate(decimations):
        # Odd cameras rotate 90 degrees about their optical axis, providing
        # visibly different expected images at the same world position.
        rotation = (
            "((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 1))"
            if index % 2
            else "((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 2, 1))"
        )
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
    # CPU mapping avoids Warp kernels, CUDA stream ownership and SDK readback.
    with frame.render_vars[path].map(device=Device.CPU) as mapping:
        mapping.wait()
        return np.from_dlpack(mapping).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "periodic", "subset", "manual", "serial"], default="periodic")
    parser.add_argument("--decimations", default="1,5,2")
    parser.add_argument("--disable-spg", action="store_true")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--shader-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decimations = [int(value) for value in args.decimations.split(",")]
    usd = scene(decimations, args.mode, args.resolution, args.instances)
    (args.output_dir / "scene.usda").write_text(usd)
    print("OVRTX_ONLY", importlib.metadata.version("ovrtx"), vars(args), flush=True)
    config = RendererConfig(
        active_cuda_gpus="0",
        enable_spg=False if args.disable_spg else None,
        log_file_path=str(args.output_dir / "native.log"),
        log_level="warn",
        read_gpu_transforms=True,
        suppress_deprecation_warnings=True,
        keep_system_alive=True,
        texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
    )
    if args.shader_cache:
        reuse_shader_cache(config, args.shader_cache)
    renderer = Renderer(config)
    print("RENDERER_CREATED", flush=True)
    all_products = {f"/Render{index}/Product" for index in range(len(decimations))}
    counts = [0] * len(decimations)
    last_time = 0.0
    failures = []
    rows = []
    try:
        renderer.open_usd_from_string(usd)
        print("SCENE_OPENED", flush=True)
        for cycle in range(args.frames):
            x = [-0.18, 0.18, -0.06, 0.12, -0.12, 0.06, 0.0][cycle % 7]
            matrix = np.repeat(np.eye(4, dtype=np.float64)[None], args.instances, axis=0)
            matrix[:, 3, :3] = np.array([[x + env * 10, 0.1, 0.0] for env in range(args.instances)])
            renderer.write_attribute(
                [f"/World/Panel{env}" for env in range(args.instances)],
                "omni:xform",
                matrix,
                semantic=Semantic.XFORM_MAT4x4,
            )
            timestamp = (max(decimations) + cycle) / 50 + 2e-9
            delta = 1.0 if args.mode in ("auto", "manual", "serial") else timestamp - last_time
            requested = (
                {f"/Render{i}/Product" for i, rate in enumerate(decimations) if cycle % rate == 0}
                if args.mode in ("subset", "manual", "serial")
                else all_products
            )

            def submissions(requested=requested, delta=delta):
                if args.mode == "serial":
                    for path in sorted(requested):
                        yield renderer.step(render_products={path}, delta_time=delta)
                else:
                    yield renderer.step(render_products=requested, delta_time=delta)

            last_time = timestamp
            for outputs in submissions():
                for index, rate in enumerate(decimations):
                    path = f"/Render{index}/Product"
                    if args.mode == "serial" and path not in outputs:
                        continue
                    frames = outputs[path].frames if path in outputs else []
                    counts[index] += len(frames)
                    expected_count = 1 if args.mode == "auto" or cycle % rate == 0 else 0
                    if len(frames) != expected_count:
                        failures.append(f"cycle {cycle} camera {index}: frames {len(frames)} expected {expected_count}")
                    for frame in frames:
                        cols = math.ceil(math.sqrt(args.instances))
                        tile_rows = math.ceil(args.instances / cols)
                        rgb = (
                            pixels(frame, f"/Render{index}/Color")
                            .reshape(args.resolution * tile_rows, args.resolution * cols, -1)[
                                : args.resolution, : args.resolution, :3
                            ]
                            .astype(np.int16)
                        )
                        red = (rgb[..., 0] > rgb[..., 1] + 30) & (rgb[..., 0] > rgb[..., 2] + 30)
                        center = np.argwhere(red).mean(axis=0)
                        # Cube front face is z=0.06, so camera distance is 1.94.
                        scale = args.resolution / 1.94
                        origin = (args.resolution - 1) / 2
                        expected = (
                            [origin + x * scale, origin + 0.1 * scale]
                            if index % 2
                            else [origin - 0.1 * scale, origin + x * scale]
                        )
                        error = float(np.max(np.abs(center - expected)))
                        row = {
                            "cycle": cycle,
                            "camera": index,
                            "center": center.tolist(),
                            "expected": expected,
                            "error": error,
                            "frame_start": frame.start_time,
                            "frame_end": frame.end_time,
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        if not math.isfinite(error) or error > 1.6:
                            failures.append(f"cycle {cycle} camera {index}: {row}")
                del outputs

    finally:
        renderer.destroy()
    imports = [name for name in sys.modules if name.startswith(("holosoma", "isaaclab", "omni.kit"))]
    assert not imports, imports
    result = {"mode": args.mode, "counts": counts, "failures": failures, "measurements": rows}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps({"mode": args.mode, "counts": counts, "failures": failures}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    faulthandler.dump_traceback_later(120, repeat=True)
    raise SystemExit(main())
