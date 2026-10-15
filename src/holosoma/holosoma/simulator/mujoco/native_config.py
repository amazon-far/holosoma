"""Native configuration application and API-derived validation, owned by core setup."""

from __future__ import annotations

import importlib.metadata
import json
import math
from dataclasses import fields
from enum import Flag
from inspect import signature
from numbers import Integral, Real
from typing import Any, get_type_hints

import mujoco
import numpy as np

from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig, NativeConfig

MUJOCO_VERSION = "3.10.0"
MJWARP_COMMIT = "ecaef88917a3c90cd238bf76681ca770f58033df"
WARP_VERSION = "1.15.0"
# Type relationships, not accepted values: enum members come from the installed API.
ENUMS = {
    "integrator": ("mjtIntegrator", "mjINT_"),
    "cone": ("mjtCone", "mjCONE_"),
    "solver": ("mjtSolver", "mjSOL_"),
    "jacobian": ("mjtJacobian", "mjJAC_"),
}
MODEL_INTERNAL = {"id", "signature", "name", "info", "group"}


def requested(config: NativeConfig) -> dict[str, Any]:
    """Include explicit zero/false values; omit optional overrides left unset."""
    result = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if value is not None and (not isinstance(value, dict) or value):
            result[field.name] = value
    return result


def enum_value(enum: Any, value: Any, path: str) -> int:
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
    if name in ENUMS:
        enum_name, _ = ENUMS[name]
        return enum_value(getattr(mujoco, enum_name), value, path)
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


def resolve_public(target: Any, path: str, config_path: str) -> tuple[Any, str]:
    """Only public writable scalar/vector properties; never arbitrary model/data arrays."""
    parts = path.split(".")
    for name in parts:
        if not name or name.startswith("_"):
            raise ValueError(f"{config_path}: expected a public upstream setting")
    for name in parts[:-1]:
        descriptor = getattr(type(target), name, None)
        if not isinstance(descriptor, property):
            raise ValueError(f"{config_path}: unknown upstream setting {path!r}")
        target = getattr(target, name)
    name = parts[-1]
    descriptor = getattr(type(target), name, None)
    if not isinstance(descriptor, property) or descriptor.fset is None:
        raise ValueError(f"{config_path}: unknown or read-only upstream setting {path!r}")
    return target, name


def validate_value(current: Any, value: Any, path: str) -> Any:
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
    options = effective_options(config)
    if options and mujoco.__version__ != MUJOCO_VERSION:
        raise ValueError(
            f"simulator.config.sim.mujoco: native settings require MuJoCo {MUJOCO_VERSION}; "
            f"installed {mujoco.__version__}"
        )
    for name, value in options.items():
        path = (
            f"simulator.config.sim.mujoco.{name}"
            if name in MujocoConfig.__dataclass_fields__
            else f"simulator.config.sim.mujoco.extra.{name}"
        )
        if name == "timestep":
            raise ValueError(f"{path}: sim.fps owns timestep")
        target, key = resolve_public(spec.option, name, path)
        setattr(target, key, validate_value(getattr(target, key), native_value(name, value, path), path))
    spec.option.timestep = 1.0 / config.sim.fps
    if config.mujoco_backend == "warp":
        validate_warp_options(spec.option, options)
        warp_option_overrides(config.sim.mujoco.warp)


def apply_fields(target: Any, config: NativeConfig, path: str) -> None:
    for name, value in requested(config).items():
        if name == "extra":
            continue
        setattr(target, name, value)
    for name, value in config.extra.items():
        if name in MODEL_INTERNAL or name in {"mass", "density"}:
            raise ValueError(f"{path}.extra.{name}: already owned by shared physics/scene configuration")
        obj, key = resolve_public(target, name, f"{path}.extra.{name}")
        setattr(obj, key, validate_value(getattr(obj, key), value, f"{path}.extra.{name}"))


def validate_warp_options(option: Any, overrides: dict[str, Any]) -> None:
    import mujoco_warp as mjw

    for name, enum in [("integrator", mjw.IntegratorType), ("solver", mjw.SolverType), ("cone", mjw.ConeType)]:
        if getattr(option, name) not in set(enum):
            raise ValueError(f"simulator.config.sim.mujoco.{name}: unsupported by pinned MJWarp")
    for name, enum in [("disableflags", mjw.DisableBit), ("enableflags", mjw.EnableBit)]:
        allowed = 0
        for flag in enum:
            allowed |= int(flag)
        if getattr(option, name) & ~allowed:
            raise ValueError(f"simulator.config.sim.mujoco.{name}: contains flags unsupported by pinned MJWarp")
    if option.noslip_iterations:
        raise ValueError("simulator.config.sim.mujoco.noslip_iterations: pinned MJWarp has no no-slip solver")
    supported = {f.name for f in fields(mjw.Option)} | {"impratio", "jacobian", "noslip_iterations"}
    for name in overrides:
        if name not in supported:
            raise ValueError(f"simulator.config.sim.mujoco.extra.{name}: not implemented by pinned MJWarp")
    if "disableflags" in overrides and option.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD:
        raise ValueError("simulator.config.sim.mujoco.disableflags.nativeccd: ignored by pinned MJWarp")


def check_warp_version() -> None:
    import warp as wp

    distribution = importlib.metadata.distribution("mujoco-warp")
    if mujoco.__version__ != MUJOCO_VERSION or distribution.version != "3.10.0" or wp.__version__ != WARP_VERSION:
        raise ValueError(
            "simulator.config.sim.mujoco: use MuJoCo 3.10.0, pinned MJWarp ecaef889 "
            "and Warp 1.15.0 from scripts/versions.sh"
        )
    direct = distribution.read_text("direct_url.json")
    revision = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    if revision and revision != MJWARP_COMMIT:
        raise ValueError(f"simulator.config.sim.mujoco: unsupported revision {revision}")


def warp_extra_inputs(cfg: MujocoWarpConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route bare names using the public API; reject ambiguous or generated inputs."""
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
    """Validate Warp-only scalar options against their actual native field types."""
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
        else:
            raise ValueError(f"{path}: structural/generated Option field is not a configurable scalar")
    return result


def apply_warp_model(model: Any, cfg: MujocoConfig, options: dict[str, Any]) -> None:
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
