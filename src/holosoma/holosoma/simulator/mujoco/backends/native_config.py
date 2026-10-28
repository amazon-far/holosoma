"""Validate native MJWarp inputs and apply execution options before allocation/capture."""

from __future__ import annotations

from dataclasses import fields
from enum import Flag
from inspect import signature
from typing import Any, get_type_hints

import mujoco
import numpy as np

from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.simulator.mujoco.native_config import enum_value, validate_value

UNSUPPORTED_ENABLE_FLAGS = {
    "sleep": "pose and DOF state writes do not wake sleeping trees",
}


def validate_warp_options(model: mujoco.MjModel, overrides: dict[str, Any]) -> None:
    """Reject known Holosoma limitations and options conversion would silently omit.

    ``put_model`` already rejects unsupported integrators, solvers, flags and the
    no-slip solver. API coverage comes from Option, with allowances for transformed
    or compilation inputs; the blacklist covers known state-handling limitations.
    """
    import mujoco_warp as mjw

    for name, reason in UNSUPPORTED_ENABLE_FLAGS.items():
        if model.opt.enableflags & int(getattr(mujoco.mjtEnableBit, "mjENBL_" + name.upper())):
            raise ValueError(
                f"simulator.config.sim.mujoco.enableflags: Holosoma's Warp backend does not support {name}: {reason}"
            )
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
            owner = name if name in MujocoConfig.__dataclass_fields__ else "extra." + name
            raise ValueError(f"{path}: physics option; configure simulator.config.sim.mujoco.{owner}")
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


def allocation_kwargs(
    model: mujoco.MjModel, cfg: MujocoWarpConfig, nworld: int, *, inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    if inputs is None:
        _, inputs = warp_extra_inputs(cfg)
    for name, value in inputs.items():
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


def warp_option_overrides(cfg: MujocoWarpConfig, *, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert requested Warp-only options using the installed Option annotations.

    This validates scalar types and readable enum/flag names before graph capture.
    Structural fields are not supported as generic settings. Allocation extras are
    handled separately by allocation_kwargs.
    """
    import mujoco_warp as mjw

    option_types = get_type_hints(mjw.Option)
    option_inputs = dict(inputs) if inputs is not None else warp_extra_inputs(cfg)[0]
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
    """Restore explicit tolerance and apply validated execution options before capture.

    put_model owns physics conversion, including derived values and island policy.
    Its float32 tolerance floor is a default; explicit values replace it.
    """
    if cfg.tolerance is not None:
        model.opt.tolerance.assign(np.asarray([cfg.tolerance], dtype=np.float32))
    for name, value in options.items():
        setattr(model.opt, name, value)
