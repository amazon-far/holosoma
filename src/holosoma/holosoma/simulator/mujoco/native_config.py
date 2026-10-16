"""Apply native settings at the scene and backend stages that consume them.

``MujocoSceneManager.compile`` applies world options before compiling the spec.
``WarpBackend`` converts that model and applies uploaded options, then allocates
data and applies workspace options before CUDA graph capture. Reset preserves
those settings; changing them requires constructing a new simulator.

Generic inputs are checked against public properties, Option fields and allocation
signatures. MJWarp's own conversion checks reject unsupported solvers and flags.
Dependency versions belong to the setup requirements in ``scripts/versions.sh``.
"""

from __future__ import annotations

import math
from dataclasses import fields
from enum import Flag
from inspect import signature
from numbers import Integral, Real
from typing import Any, get_type_hints

import mujoco
import numpy as np

from holosoma.config_types.mujoco import MujocoConfig, MujocoGeomPhysicsConfig, MujocoWarpConfig

# MuJoCo exposes these properties as plain ints, without their enum type metadata.
# Keep the four type relationships explicit; choices come from the installed enums.
ENUMS = {
    "integrator": mujoco.mjtIntegrator,
    "cone": mujoco.mjtCone,
    "solver": mujoco.mjtSolver,
    "jacobian": mujoco.mjtJacobian,
}
MODEL_INTERNAL = {"id", "signature", "name", "info", "group"}


def requested(config: MujocoConfig | MujocoGeomPhysicsConfig) -> dict[str, Any]:
    """Collect configured dataclass fields for application.

    None and empty extra mappings are inactive. Zero, False and empty flag tuples
    remain explicit overrides, including tuples that clear an authored flag mask.
    """
    result = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if value is not None and (not isinstance(value, dict) or value):
            result[field.name] = value
    return result


def enum_value(enum: Any, value: Any, path: str) -> int:
    """Resolve a native enum name or integer, reporting invalid values at ``path``.

    Names accept the upstream spelling or its suffix, case-insensitively. Flag
    integers may combine defined bits; ordinary enums require an actual member.
    """
    members = enum.__members__
    if isinstance(value, str):
        matches = [
            v for k, v in members.items() if k.lower() == value.lower() or k.lower().endswith("_" + value.lower())
        ]
        if len(matches) == 1:
            return int(matches[0])
    elif isinstance(value, int):
        if issubclass(enum, Flag):
            allowed = 0
            for member in members.values():
                allowed |= int(member)
            if value >= 0 and not value & ~allowed:
                return value
        elif value in {int(v) for v in members.values()}:
            return value
    raise ValueError(f"{path}: invalid {enum.__name__} value {value!r}; upstream choices: {tuple(members)}")


def native_value(name: str, value: Any, path: str) -> Any:
    """Convert readable MuJoCo enum/flag inputs to the integers required by MjOption.

    Flag-name sequences replace the whole mask; an empty sequence clears it.
    Other scalar/vector properties retain their value for native type validation.
    """
    if name in ENUMS:
        return enum_value(ENUMS[name], value, path)
    if name in ("disableflags", "enableflags"):
        enum_name, prefix = ("mjtDisableBit", "mjDSBL_") if name == "disableflags" else ("mjtEnableBit", "mjENBL_")
        enum = getattr(mujoco, enum_name)
        names = {k[len(prefix) :].lower(): int(v) for k, v in enum.__members__.items() if k.startswith(prefix)}
        if isinstance(value, int):
            allowed = 0
            for bit in names.values():
                allowed |= bit
            if value < 0 or value & ~allowed:
                raise ValueError(f"{path}: contains unknown {enum_name} bits")
            return value
        flags = value.split("|") if isinstance(value, str) and value else () if isinstance(value, str) else value
        mask = 0
        for flag in flags:
            key = flag.strip().lower()
            if key.startswith(prefix.lower()):
                key = key[len(prefix) :]
            if key not in names:
                raise ValueError(f"{path}: unknown flag {flag!r}; upstream choices: {tuple(names)}")
            mask |= names[key]
        return mask
    return value


def resolve_public(target: Any, name: str, config_path: str) -> tuple[Any, str]:
    """Find a public writable binding on an MjOption or MjsGeom.

    Reject unknown/read-only names before assignment and include the Holosoma
    config path in the error. Generic settings do not traverse nested attributes.
    """
    descriptor = getattr(type(target), name, None)
    if name.startswith("_") or not isinstance(descriptor, property) or descriptor.fset is None:
        raise ValueError(f"{config_path}: unknown or read-only upstream setting {name!r}")
    return target, name


def validate_value(current: Any, value: Any, path: str) -> Any:
    """Check a generic input against the native scalar or vector it will replace.

    Vectors must have the native shape and finite numeric elements. Scalars retain
    native bool/integer/real/string distinctions. Structural fields are rejected;
    model authoring is outside this simulator configuration interface.
    """
    if hasattr(type(current), "__members__"):
        return type(current)(enum_value(type(current), value, path))
    if isinstance(current, np.ndarray):
        array = np.asarray(value)
        if array.shape != current.shape or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{path}: expected a finite numeric vector of shape {current.shape}")
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected a boolean")
        return value
    if isinstance(current, Integral):
        if not isinstance(value, (int, bool)):
            raise ValueError(f"{path}: expected an integer/boolean")
        return value
    if isinstance(current, Real):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path}: expected a finite number")
        return value
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string")
        return value
    raise ValueError(f"{path}: structural/generated data belongs in the asset, not simulator settings")


def effective_options(config: Any) -> dict[str, Any]:
    """Merge named world options and ``extra`` from a SimulatorInitConfig.

    Return explicit MjOption overrides, without changing the config or model.
    Extra keys must exist on the installed public API; Pydantic has already
    rejected duplicates and timestep overrides during configuration parsing.
    """
    cfg = config.sim.mujoco
    selected = requested(cfg)
    result = {
        name: value for name, value in selected.items() if isinstance(getattr(mujoco.MjOption, name, None), property)
    }
    for name, value in cfg.extra.items():
        if not isinstance(getattr(mujoco.MjOption, name, None), property):
            raise ValueError(
                f"simulator.config.sim.mujoco.extra.{name}: "
                "unknown public MjOption input; put MJWarp-only inputs in mujoco.warp.extras"
            )
        result[name] = value
    return result


def apply_world_config(spec: mujoco.MjSpec, config: Any) -> None:
    """Apply world options immediately before spec compilation.

    Unset options retain the spec values. The physics rate always supplies timestep.
    For Warp, reject requested fields that conversion would silently discard and
    validate Warp extras; native conversion owns solver/flag compatibility checks.
    """
    options = effective_options(config)
    for name, value in options.items():
        path = (
            f"simulator.config.sim.mujoco.{name}"
            if name in MujocoConfig.__dataclass_fields__
            else f"simulator.config.sim.mujoco.extra.{name}"
        )
        target, key = resolve_public(spec.option, name, path)
        setattr(target, key, validate_value(getattr(target, key), native_value(name, value, path), path))
    spec.option.timestep = 1.0 / config.sim.fps
    if config.mujoco_backend == "warp":
        validate_warp_option_coverage(options)
        warp_option_overrides(config.sim.mujoco.warp)


def apply_fields(target: Any, config: MujocoGeomPhysicsConfig, path: str) -> None:
    """Apply surface contact overrides to one MjsGeom before compilation.

    The scene manager calls this for every geom of a scene object or robot link.
    Extra inputs cannot overwrite identity/generated fields or the mass/density
    settings already owned by shared PhysicsConfig.
    """
    for name, value in requested(config).items():
        if name == "extra":
            continue
        setattr(target, name, value)
    for name, value in config.extra.items():
        if name in MODEL_INTERNAL or name in {"mass", "density"}:
            raise ValueError(f"{path}.extra.{name}: already owned by shared physics/scene configuration")
        obj, key = resolve_public(target, name, f"{path}.extra.{name}")
        setattr(obj, key, validate_value(getattr(obj, key), value, f"{path}.extra.{name}"))


def validate_warp_option_coverage(overrides: dict[str, Any]) -> None:
    """Reject explicit MuJoCo options that MJWarp conversion would silently omit.

    ``put_model`` already rejects unsupported integrators, solvers, flags and the
    no-slip solver. This check only covers valid MuJoCo properties absent from
    Warp's Option surface, with allowances for transformed or compilation inputs.
    """
    import mujoco_warp as mjw

    supported = {f.name for f in fields(mjw.Option)} | {"impratio", "jacobian", "noslip_iterations"}
    for name in overrides:
        if name not in supported:
            raise ValueError(f"simulator.config.sim.mujoco.extra.{name}: not implemented by MJWarp")


def warp_extra_inputs(cfg: MujocoWarpConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition ``warp.extras`` between execution options and allocation arguments.

    Names come from Option fields and the public make_data signature. Shared physics
    stays in MujocoConfig; derived fields and ambiguous names are errors. Return
    separate dictionaries without allocating data or modifying the uploaded model.
    """
    import mujoco_warp as mjw

    option_names = {field.name for field in fields(mjw.Option)}
    allocation_names = set(signature(mjw.make_data).parameters)
    options, allocations = {}, {}
    for name, value in cfg.extras.items():
        path = f"simulator.config.sim.mujoco.warp.extras.{name}"
        if isinstance(getattr(mujoco.MjOption, name, None), property):
            raise ValueError(f"{path}: shared physics option; configure sim.mujoco.{name} or sim.mujoco.extra.{name}")
        if name == "impratio_invsqrt":
            raise ValueError(f"{path}: derived Option field; configure impratio directly")
        if name in option_names and name in allocation_names:
            raise ValueError(
                f"{path}: ambiguous upstream option/allocation name; dependency compatibility review required"
            )
        if name in option_names:
            options[name] = value
        elif name in allocation_names:
            allocations[name] = value
        else:
            raise ValueError(f"{path}: unknown public MJWarp Option/make_data input")
    return options, allocations


def allocation_kwargs(model: mujoco.MjModel, cfg: MujocoWarpConfig, nworld: int) -> dict[str, Any]:
    """Build validated make_data arguments before backend allocation.

    Contacts use an average per-world capacity in a shared pool, unless naconmax
    sets the total explicitly. Constraint rows have a hard per-world limit.
    Preserve Holosoma's existing defaults and reject invalid CCD/velocity capacities
    with config paths before starting GPU work.
    """
    nconmax = cfg.nconmax_per_env if cfg.nconmax_per_env is not None else 96
    result = {
        "nworld": nworld,
        "nconmax": nconmax,
        "njmax": cfg.njmax_per_env if cfg.njmax_per_env is not None else max(nconmax * 6, model.nv * 4),
    }
    if cfg.naconmax is not None:
        result["naconmax"] = cfg.naconmax
    _, allocation_inputs = warp_extra_inputs(cfg)
    for name, value in allocation_inputs.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"simulator.config.sim.mujoco.warp.extras.{name}: "
                "expected a nonnegative allocation argument from make_data"
            )
        result[name] = value
    if result.get("nvmax", model.nv) > model.nv:
        raise ValueError("simulator.config.sim.mujoco.warp.extras.nvmax: exceeds model.nv")
    ncon = result["nconmax"]
    total = result.get("naconmax", ncon * nworld)
    ccd_total = result.get("naccdmax", result["nccdmax"] * nworld if "nccdmax" in result else total)
    if result.get("nccdmax", ncon) > ncon or ccd_total > total:
        raise ValueError("simulator.config.sim.mujoco.warp: CCD capacity exceeds contact allocation")
    return result


def warp_option_overrides(cfg: MujocoWarpConfig) -> dict[str, Any]:
    """Convert requested Warp-only options using the installed Option annotations.

    This validates scalar types and readable enum/flag names before graph capture.
    Structural fields are not supported as generic settings. Allocation extras are
    handled separately by allocation_kwargs.
    """
    import mujoco_warp as mjw

    option_types = get_type_hints(mjw.Option)
    option_inputs, _ = warp_extra_inputs(cfg)
    if cfg.graph_conditional is not None:
        option_inputs["graph_conditional"] = cfg.graph_conditional
    result = {}
    for name, value in option_inputs.items():
        path = (
            f"simulator.config.sim.mujoco.warp."
            f"{name if name in MujocoWarpConfig.__dataclass_fields__ else 'extras.' + name}"
        )
        native_type = option_types[name]
        if hasattr(native_type, "__members__"):
            if issubclass(native_type, Flag) and isinstance(value, str):
                applied = 0
                for flag in value.split("|"):
                    applied |= enum_value(native_type, flag, path)
                result[name] = applied
            else:
                result[name] = enum_value(native_type, value, path)
        elif native_type in (bool, int, float, str):
            result[name] = validate_value(native_type(), value, path)
            if name == "contact_sensor_maxmatch" and value < 0:
                raise ValueError(f"{path}: expected a nonnegative contact-match capacity")
        else:
            raise ValueError(f"{path}: structural/generated Option field is not a configurable scalar")
    return result


def apply_warp_model(model: Any, cfg: MujocoConfig, options: dict[str, Any]) -> None:
    """Apply effective physics and execution options to a converted Warp model.

    Call inside the backend's device scope after put_model and before graph capture.
    Restore explicit values that conversion changes, including tolerances below its
    default float32 floor. Transform impratio to its inverse square root and preserve
    Warp's required ISLAND disable bit. Assign real device arrays or scalar options;
    this function does not update an already captured graph.
    """
    import mujoco_warp as mjw

    applied: Any
    for name, value in options.items():
        if name in {"jacobian", "noslip_iterations"}:
            continue
        if name == "impratio":
            target, applied = model.opt.impratio_invsqrt, 1.0 / math.sqrt(max(value, mujoco.mjMINVAL))
        else:
            applied = native_value(name, value, f"sim.mujoco.{name}")
            target = getattr(model.opt, name)
        if hasattr(target, "assign"):
            target.assign(np.asarray([applied], dtype=np.float32))
        else:
            if name == "disableflags":
                applied |= int(mjw.DisableBit.ISLAND)  # upstream disables experimental island allocation
            setattr(model.opt, name, applied)
    for name, value in warp_option_overrides(cfg.warp).items():
        setattr(model.opt, name, value)


def apply_warp_data(data: Any, model: mujoco.MjModel, cfg: MujocoConfig) -> None:
    """Restore an explicit tolerance in the sleeping solver's allocated workspace.

    make_data clamps its compact tolerance independently of put_model. Preserve
    the native DOF normalization, then update the real array before graph capture.
    Unset tolerances and allocations without a compact workspace retain defaults.
    """
    if cfg.tolerance is not None and data.ctol.size:
        scale = model.nv / data.nvmax_pad
        data.ctol.assign(np.asarray([cfg.tolerance * scale], dtype=np.float32))
