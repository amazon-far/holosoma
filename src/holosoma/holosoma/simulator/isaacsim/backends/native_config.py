"""Translate Holosoma's small selection API into public Isaac Lab 3 configs."""

from __future__ import annotations

import dataclasses
import types
from importlib import import_module
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from holosoma.config_types.isaac import IsaacCameraRendererConfig, IsaacSimRenderConfig

if TYPE_CHECKING:
    from holosoma.config_types.simulator import SimulatorInitConfig

PHYSICS = {
    "isaacsim_physx": ("isaaclab_physx.physics", "PhysxCfg"),
    "newton": ("isaaclab_newton.physics", "NewtonCfg"),
    "ovphysx": ("isaaclab_ov.physics", "OvPhysxCfg"),
}
SOLVERS = {
    "mjwarp": "MJWarpSolverCfg",
    "kamino_padmm": "KaminoPADMMSolverCfg",
    "kamino_dvi": "KaminoDVISolverCfg",
    "xpbd": "XPBDSolverCfg",
    "featherstone": "FeatherstoneSolverCfg",
    "vbd": "VBDSolverCfg",
}
RENDERERS = {
    "isaacsim_rtx": ("isaaclab_physx.renderers", "IsaacRtxRendererCfg"),
    "ovrtx": ("isaaclab_ov.renderers", "OVRTXRendererCfg"),
    "newton_renderer": ("isaaclab_newton.renderers", "NewtonWarpRendererCfg"),
}
VISUALIZERS = {
    "kit": ("isaaclab_visualizers.kit", "KitVisualizerCfg"),
    "newton_gl": ("isaaclab_visualizers.newton", "NewtonGLVisualizerCfg"),
    "newton_rtx": ("isaaclab_visualizers.newton", "NewtonRTXVisualizerCfg"),
    "viser": ("isaaclab_visualizers.viser", "ViserVisualizerCfg"),
}


def construct(target: tuple[str, str]) -> Any:
    module, name = target
    return getattr(import_module(module), name)()


def apply_options(config: Any, options: dict[str, Any], *, path: str, owned: tuple[str, ...] = ()) -> Any:
    """Set public native fields, preserving native nested config types and validation."""
    fields = {field.name for field in dataclasses.fields(config) if field.init}
    values = {name: getattr(config, name) for name in fields - {"class_type", "launcher_type"}}
    for name, value in options.items():
        location = f"{path}.{name}"
        if name.startswith("_") or name not in fields or name in {"class_type", "launcher_type"}:
            raise ValueError(f"{location}: unknown or non-configurable field on {type(config).__name__}.")
        if name in owned:
            raise ValueError(f"{location}: owned by Holosoma's explicit configuration; set it there.")
        previous = getattr(config, name)
        if isinstance(value, dict):
            if previous is None:
                annotation = get_type_hints(type(config)).get(name)
                candidates = (
                    get_args(annotation) if get_origin(annotation) in (Union, types.UnionType) else (annotation,)
                )
                cls = next((c for c in candidates if isinstance(c, type) and dataclasses.is_dataclass(c)), None)
                if cls is not None:
                    previous = cls()
            if dataclasses.is_dataclass(previous):
                value = apply_options(previous, value, path=location)
            elif not isinstance(previous, dict):
                raise ValueError(f"{location}: expected a native scalar or sequence, not a config object.")
        values[name] = value
    # Run the native constructor after overrides. In particular, Newton derives
    # its manager class from solver_cfg in __post_init__; mutating a default
    # NewtonCfg would leave every selected solver routed to MJWarp.
    return type(config)(**values)


def physics_config(config: SimulatorInitConfig) -> Any:
    selection = config.isaacsim
    physics = construct(PHYSICS[selection.physics])
    if selection.physics == "isaacsim_physx":
        source = config.sim.physx
        physics.solver_type = source.solver_type
        physics.max_position_iteration_count = source.num_position_iterations
        physics.max_velocity_iteration_count = source.num_velocity_iterations
        physics.bounce_threshold_velocity = source.bounce_threshold_velocity
        physics.gpu_max_rigid_patch_count = 10 * 2**15
        physics.enable_external_forces_every_iteration = False
    elif selection.physics == "newton":
        solver = construct(("isaaclab_newton.physics", SOLVERS[selection.solver or "mjwarp"]))
        physics.solver_cfg = apply_options(solver, selection.solver_options, path="solver_options")
    return apply_options(physics, selection.physics_options, path="physics_options", owned=("solver_cfg",))


def isaac_rtx_rendering_mode(mode: str) -> Any:
    """Translate the Lab 2.3.2 rendering presets removed by Lab 3.

    Values match ``apps/rendering_modes/{performance,balanced,quality}.kit``.
    Use native named fields where available; the remaining RTX flags stay raw.
    """
    from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

    quality = mode == "quality"
    performance = mode == "performance"
    extra: dict[str, Any] = {
        "/isaaclab/rendering/rendering_mode": mode,
        "/rtx/reflections/denoiser/enabled": not performance,
        "/rtx/indirectDiffuse/denoiser/enabled": not performance,
        "/rtx/directLighting/sampledLighting/enabled": not performance,
    }
    if performance:
        extra["/rtx/directLighting/sampledLighting/denoisingTechnique"] = 0
    return IsaacRtxRendererGlobalSettingsCfg(
        enable_translucency=quality,
        enable_reflections=quality,
        enable_global_illumination=quality,
        enable_dlssg=False,
        enable_dl_denoiser=not performance,
        dlss_mode={"performance": 0, "balanced": 1, "quality": 2}[mode],
        enable_shadows=True,
        enable_ambient_occlusion=quality,
        dome_light_upper_lower_strategy=3 if performance else None,
        ambient_light_intensity=1.0,
        ambient_occlusion_denoiser_mode=0 if quality else 1,
        subpixel_mode=1 if quality else 0,
        enable_cached_raytracing=not performance,
        max_samples_per_launch=1000000,
        view_tile_limit=1000000,
        carb_settings=extra,
    )


def renderer_config(selection: IsaacSimRenderConfig, override: IsaacCameraRendererConfig | None = None) -> Any:
    name = override.backend if override is not None and override.backend is not None else selection.renderer
    if name == "none":
        raise ValueError("Camera or video capture requires a renderer; renderer='none' disables capture.")
    # Options belong to a renderer type. A camera selecting another renderer
    # starts with that renderer's defaults instead of inheriting unrelated fields.
    options = dict(selection.renderer_options) if name == selection.renderer else {}
    if override is not None:
        options.update(override.options)
    config = construct(RENDERERS[name])
    if name == "isaacsim_rtx" and selection.rendering_mode is not None:
        config.global_settings = isaac_rtx_rendering_mode(selection.rendering_mode)
    return apply_options(config, options, path="renderer_options")


def visualizer_configs(selection: IsaacSimRenderConfig, *, headless: bool) -> list[Any]:
    result = []
    for name in selection.selected_visualizers(headless):
        cfg = construct(VISUALIZERS[name])
        if name == "viser":
            cfg.bind_address = "127.0.0.1"
        result.append(apply_options(cfg, selection.visualizer_options.get(name, {}), path=f"visualizer_options.{name}"))
    return result
