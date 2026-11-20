"""Apply native settings at the scene and backend stages that consume them.

``MujocoSceneManager.compile`` applies world options before compiling the spec.
Warp-specific validation and application belong to the backend. Reset preserves
settings; changing them requires constructing a new simulator.

Generic inputs are checked against public properties and their native values.
Dependency versions belong to the setup requirements in ``scripts/versions.sh``.
"""

from __future__ import annotations

import math
from dataclasses import fields
from enum import Flag
from numbers import Integral, Real
from typing import Any

import mujoco
import numpy as np

from holosoma.config_types.mujoco import MujocoConfig, MujocoGeomPhysicsConfig

# MuJoCo exposes these properties as plain ints, without their enum type metadata.
# Keep the four type relationships explicit; choices come from the installed enums.
ENUMS = {
    "integrator": mujoco.mjtIntegrator,
    "cone": mujoco.mjtCone,
    "solver": mujoco.mjtSolver,
    "jacobian": mujoco.mjtJacobian,
}
MODEL_AUTHORING = {
    "id",
    "signature",
    "name",
    "info",
    "group",
    "classname",
    "compiler",
    "frame",
    "parent",
    "plugin",
    "type",
    "typeinertia",
    "pos",
    "quat",
    "alt",
    "fromto",
    "size",
    "fitscale",
    "meshname",
    "hfieldname",
    "material",
    "rgba",
    "userdata",
}


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
    native bool/integer/real/string distinctions. Structured objects are rejected.
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
    Backend validation runs later, before model conversion.
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


def apply_fields(target: Any, config: MujocoGeomPhysicsConfig, path: str) -> None:
    """Apply surface contact overrides to one MjsGeom before compilation.

    The scene manager calls this for every geom of a scene object or robot link.
    Mask overrides leave authored visual-only geoms inactive. Extra inputs cannot
    replace asset authoring or mass/density settings owned by shared PhysicsConfig.
    """
    visual_only = target.contype == 0 and target.conaffinity == 0
    for name, value in requested(config).items():
        if name == "extra":
            continue
        if visual_only and name in {"contype", "conaffinity"}:
            continue
        setattr(target, name, value)
    for name, value in config.extra.items():
        if name in MODEL_AUTHORING or name in {"mass", "density"}:
            raise ValueError(f"{path}.extra.{name}: belongs to asset authoring or shared physics configuration")
        obj, key = resolve_public(target, name, f"{path}.extra.{name}")
        setattr(obj, key, validate_value(getattr(obj, key), value, f"{path}.extra.{name}"))
