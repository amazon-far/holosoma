"""Regenerate the versioned API inventory (run in Holosoma's pinned MuJoCo environment)."""

import dataclasses
import inspect
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.simulator.mujoco.native_config import MJWARP_COMMIT


def public_properties(obj):
    return {
        name: getattr(obj, name)
        for name in dir(type(obj))
        if isinstance(getattr(type(obj), name), property) and not name.startswith("_")
    }


def type_name(native_type):
    if hasattr(native_type, "dtype"):
        return "array[" + native_type.dtype.__name__ + "]"
    return getattr(native_type, "__name__", str(native_type))


def json_default(value):
    if hasattr(type(value), "__members__"):
        return int(value)
    return value


def setting(upstream, path, native_type, default, stage, notes, exclusion=None):
    return {
        "upstream": upstream,
        "path": path,
        "type": type_name(native_type),
        "native_default": default,
        "backends": "warp",
        "stage": stage,
        "notes": notes,
        "exclusion": exclusion,
    }


def inventory():
    shared = public_properties(mujoco.MjOption())
    warp_fields = {f.name: f for f in dataclasses.fields(mjw.Option)}
    records = []
    for name, value in shared.items():
        default = value.tolist() if isinstance(value, np.ndarray) else value
        backend = (
            "classic + warp"
            if name in warp_fields or name in {"impratio", "jacobian", "noslip_iterations", "timestep"}
            else "classic"
        )
        path = (
            "simulator.config.sim.fps"
            if name == "timestep"
            else "simulator.config.sim.mujoco."
            + (name if name in MujocoConfig.__dataclass_fields__ else "extra." + name)
        )
        notes = "Unset retains compiled asset/native values."
        if name == "timestep":
            notes = "Excluded from native input: derived as 1/fps; owns all rates and simulation time."
        if name == "noslip_iterations":
            notes = "Warp supports only zero; no no-slip solver."
        if name == "impratio":
            notes = "Uploaded as impratio_invsqrt; explicit value applies before capture."
        if name == "jacobian":
            notes = "Conversion selects layout; dense nv>60 rejected by pinned upstream."
        if name == "tolerance":
            notes = (
                "Unset Warp upload uses at least 1e-6; "
                "explicit values are restored on the uploaded array before capture."
            )
        if name in {"disableflags", "enableflags"}:
            notes = (
                "Readable names validated against upstream enums; "
                "Warp rejects unsupported bits and retains its disabled island solver."
            )
        records.append(
            {
                "upstream": "MjOption." + name,
                "path": path,
                "type": type(value).__name__,
                "native_default": default,
                "backends": backend,
                "stage": "final compilation, then upload before capture",
                "notes": notes,
            }
        )
    block_type = next(f.type for f in dataclasses.fields(mjw.Model) if f.name == "block_dim")
    prefix = "simulator.config.sim.mujoco.warp."
    warp_options = [
        setting(
            "mujoco_warp.Option." + name,
            (
                next(record["path"] for record in records if record["upstream"] == "MjOption." + name)
                if name in shared
                else prefix + (name if name in MujocoWarpConfig.__dataclass_fields__ else "extras." + name)
                if name != "impratio_invsqrt"
                else None
            ),
            field.type,
            "conversion from compiled MjOption" if name in shared else "upstream put_model default",
            "conversion, then explicit overrides before capture",
            "One shared input is applied during compilation and restored on upload as needed."
            if name in shared and name != "timestep"
            else "Upstream derived impedance representation; use mujoco.impratio."
            if name == "impratio_invsqrt"
            else "Native conversion default retained when unset.",
            "Derived from impratio; direct override would create competing sources of truth."
            if name == "impratio_invsqrt"
            else "Derived from sim.fps; not a native input."
            if name == "timestep"
            else None,
        )
        for name, field in warp_fields.items()
    ]
    allocation_notes = {
        "nconmax": "Average contacts per world, pooled across worlds; Holosoma default 96.",
        "njmax": "Hard constraint rows per world; Holosoma default max(6*nconmax_per_env, 4*nv).",
        "nccdmax": "Average CCD contacts per world; defaults to nconmax; must not exceed nconmax.",
        "naconmax": "Total pooled contacts; overrides nconmax*nworld.",
        "naccdmax": "Total CCD pool; overrides nccdmax*nworld; defaults to naconmax; cannot exceed it.",
        "njmax_nnz": "Sparse constraint Jacobian nonzeros per world; native model-dependent estimate.",
        "nvmax": "Compacted active DOFs per world; defaults to model.nv; must be in [0, model.nv].",
    }
    allocations = [
        setting(
            "mujoco_warp.make_data." + name,
            "training.num_envs"
            if name == "nworld"
            else prefix + name + "_per_env"
            if name in {"nconmax", "njmax"}
            else prefix + (name if name in MujocoWarpConfig.__dataclass_fields__ else "extras." + name)
            if name != "mjm"
            else None,
            parameter.annotation,
            None if parameter.default is inspect.Parameter.empty else parameter.default,
            "before allocation",
            allocation_notes.get(name, "Owned by shared simulator setup."),
            "Compiled model input, not a configurable resource." if name == "mjm" else None,
        )
        for name, parameter in inspect.signature(mjw.make_data).parameters.items()
    ]
    blocks = [
        setting(
            "mujoco_warp.Model.block_dim." + field.name,
            None,
            field.type,
            field.default,
            "before kernel loading and capture",
            "Native kernel block dimensions remain unchanged.",
            "Kernel tiling is outside this focused physics/resource configuration.",
        )
        for field in dataclasses.fields(block_type)
    ]
    compilation = [
        setting(
            "warp.module_options." + name,
            None,
            type(value),
            json_default(value),
            "before kernel loading",
            "Native Warp module compilation policy remains unchanged.",
            "Process-wide compiler settings belong to backend execution, not per-simulator physics configuration.",
        )
        for name, value in sorted(wp.get_module_options().items())
    ]
    capture = [
        setting(
            "warp.ScopedCapture." + name,
            None,
            parameter.annotation if parameter.annotation is not inspect.Parameter.empty else type(parameter.default),
            json_default(parameter.default),
            "capture",
            "Core backend setup retains its native capture defaults.",
            "Device/stream objects are owned by simulator execution, not scalar configuration."
            if name in {"device", "stream"}
            else "Core backend owns graph capture; execution plumbing is outside this focused configuration.",
        )
        for name, parameter in inspect.signature(wp.ScopedCapture).parameters.items()
    ]
    batchable = sorted(
        field.name for field in dataclasses.fields(mjw.Model) if getattr(field.type, "shape", ())[:1] == ("*",)
    )
    return {
        "versions": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": "ecaef88917a3c90cd238bf76681ca770f58033df",
            "warp": wp.__version__,
        },
        "sources": [
            f"https://github.com/google-deepmind/mujoco/blob/{mujoco.__version__}/include/mujoco/mjmodel.h",
            f"https://github.com/google-deepmind/mujoco_warp/blob/{MJWARP_COMMIT}/mujoco_warp/_src/types.py",
            f"https://github.com/google-deepmind/mujoco_warp/blob/{MJWARP_COMMIT}/mujoco_warp/_src/io.py",
        ],
        "options": records,
        "warp_options": warp_options,
        "allocation": allocations,
        "block_dimensions": blocks,
        "compilation": compilation,
        "capture": capture,
        "conversion": [
            setting(
                "mujoco_warp.put_model.mjm",
                None,
                mujoco.MjModel,
                None,
                "conversion",
                "Compiled model comes from core world assembly.",
                "Model input is not a setting.",
            ),
            setting(
                "mujoco_warp.put_model.batch_sizes",
                None,
                int,
                "1 (shared) for each field",
                "conversion",
                "Core domain-randomization setup expands its owned model fields to nworld before capture.",
                "Per-field model batching is owned by core domain randomization, not a competing configuration input.",
            ),
        ],
        "batchable_model_fields": batchable,
        "authoring_exclusions": {
            "compiler/size/visual": "Authored in existing MJCF XML asset paths; no second model-authoring schema.",
            "pairs/joints/tendons/equalities": (
                "Per-element attributes authored in MJCF; do not mean global contact overrides."
            ),
            "native sensors": "Authored in MJCF; mounted cameras/LiDAR use existing shared SensorConfig.",
            "model/data arrays": (
                "Generated tables, addresses, transient state and arbitrary array writes are not settings."
            ),
        },
        "warp_option_fields": sorted(warp_fields),
        "make_data_parameters": list(inspect.signature(mjw.make_data).parameters),
        "extra_name_collisions": sorted(
            set(inspect.signature(mjw.make_data).parameters) & (set(shared) | set(warp_fields))
        ),
        "put_model_parameters": list(inspect.signature(mjw.put_model).parameters),
        "block_dim_fields": [f.name for f in dataclasses.fields(block_type)],
        "compilation_fields": sorted(wp.get_module_options()),
        "capture_parameters": list(inspect.signature(wp.ScopedCapture).parameters),
        "enum_choices": {
            name: list(getattr(mujoco, name).__members__)
            for name in ["mjtIntegrator", "mjtSolver", "mjtCone", "mjtJacobian", "mjtEnableBit", "mjtDisableBit"]
        },
        "warp_enum_choices": {
            enum.__name__: list(enum.__members__)
            for enum in [
                mjw.IntegratorType,
                mjw.SolverType,
                mjw.ConeType,
                mjw.DisableBit,
                mjw.EnableBit,
                *(field.type for field in warp_fields.values() if hasattr(field.type, "__members__")),
                *(type(value) for value in wp.get_module_options().values() if hasattr(type(value), "__members__")),
                *(
                    type(parameter.default)
                    for parameter in inspect.signature(wp.ScopedCapture).parameters.values()
                    if hasattr(type(parameter.default), "__members__")
                ),
            ]
        },
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/mujoco-parameter-inventory.json"
    path.write_text(json.dumps(inventory(), indent=2) + "\n")
    print(path)
