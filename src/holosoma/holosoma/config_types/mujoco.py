"""A small native configuration for humanoid and contact-rich MuJoCo simulations."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import field
from typing import Dict, Literal, Tuple, Union

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated
from tyro.constructors import PrimitiveConstructorSpec

Finite = Annotated[float, Field(allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, strict=True)]
SignedInt = Annotated[int, Field(ge=-2147483648, le=2147483647, strict=True)]
NativeValue = Union[bool, int, Finite, str, Tuple[Finite, ...]]


def _extra_from_cli(args: list[str]) -> dict[str, NativeValue]:
    try:
        value = json.loads(args[0])
    except ValueError:
        value = ast.literal_eval(args[0])
    if not isinstance(value, dict):
        raise ValueError("extra settings expect a JSON object or Python dictionary")
    return value


ExtraSettings = Annotated[
    Dict[str, NativeValue],
    PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=_extra_from_cli,
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [repr(value)],
    ),
]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class NativeConfig:
    """Frozen typed settings with a validated escape hatch for uncommon native inputs.

    Extra names are checked against the active upstream API during simulator setup.
    They never accept private attributes or bypass settings already owned by Holosoma.
    """

    extra: ExtraSettings = field(default_factory=dict)
    """Other public native properties, using bare upstream names; validated and applied during core setup."""

    @model_validator(mode="after")
    def validate_extra(self) -> NativeConfig:
        for name, value in self.extra.items():
            if not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"extra.{name}: use a bare public native name, without prefixes")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"extra.{name}: value must be finite")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MujocoWarpConfig:
    """MJWarp resources and execution settings, unused by classic MuJoCo.

    Core backend initialization applies these before allocation and graph capture.
    None retains Holosoma's resource defaults or the upstream execution default.
    """

    nconmax_per_env: Count | None = None
    """Average contacts per world in the shared pool; None uses 96. A world may exceed this average."""
    njmax_per_env: Count | None = None
    """Hard constraint-row capacity per world; None uses max(6*nconmax_per_env, 4*model.nv)."""
    naconmax: Count | None = None
    """Total contact-pool capacity across all worlds; None uses nconmax_per_env*training.num_envs."""
    graph_conditional: Annotated[bool, Field(strict=True)] | None = None
    """Use conditional CUDA graph nodes for solver loops; set before capture. None keeps MJWarp's default."""
    extras: ExtraSettings = field(default_factory=dict)
    """Other MJWarp-only Option fields or make_data inputs, using bare names; applied before capture."""

    @model_validator(mode="after")
    def validate_ownership(self) -> MujocoWarpConfig:
        allocation_owners = {
            "nconmax": "nconmax_per_env",
            "njmax": "njmax_per_env",
            "nworld": "training.num_envs",
            "mjm": "the model from core setup",
        }
        for name in self.extras:
            if not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"warp.extras.{name}: use a bare public native name, without prefixes")
            if name in self.__dataclass_fields__:
                raise ValueError(f"warp.extras.{name}: use the explicit warp.{name} field")
            if name in allocation_owners:
                raise ValueError(f"warp.extras.{name}: already owned by {allocation_owners[name]}")
            if name == "timestep":
                raise ValueError("warp.extras.timestep: sim.fps owns timestep; set sim.fps instead")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MujocoConfig(NativeConfig):
    """Native physics options, applied by core setup before compilation and upload.

    None preserves asset/current Holosoma behavior; sim.fps owns timestep.
    Other simulators ignore this block. Reconstruct the simulator to change it.
    """

    integrator: Literal["euler", "rk4", "implicit", "implicitfast"] | None = None
    """Native time integration method; None preserves the world setting. Both backends support all four."""
    solver: Literal["pgs", "cg", "newton"] | None = None
    """Constraint solver algorithm; MJWarp supports cg/newton. None preserves the world setting."""
    cone: Literal["pyramidal", "elliptic"] | None = None
    """Contact friction-cone representation; None preserves the world setting on either backend."""
    jacobian: Literal["dense", "sparse", "auto"] | None = None
    """Constraint Jacobian storage; auto selects by model size. None preserves the world setting."""
    impratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    """Friction-to-normal constraint impedance ratio; independent of geom friction. None preserves it."""
    iterations: Count | None = None
    """Maximum constraint-solver iterations; zero disables iterations. None preserves the world limit."""
    tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    """Constraint-solver stopping tolerance; zero disables early stopping. None retains backend behavior."""
    ls_iterations: Count | None = None
    """Maximum CG/Newton line-search iterations; zero disables line search. None preserves the limit."""
    ls_tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    """CG/Newton line-search stopping tolerance; zero disables early stopping. None preserves it."""
    noslip_iterations: Count | None = None
    """Classic no-slip post-solver iterations; zero disables it. MJWarp supports only zero."""
    gravity: tuple[Finite, Finite, Finite] | None = None
    """World-frame acceleration (x,y,z), in m/s²; None retains Holosoma's existing gravity."""
    disableflags: tuple[str, ...] | None = None
    """Replace the disable mask with upstream flag names (e.g. contact); () clears it, None preserves it."""
    enableflags: tuple[str, ...] | None = None
    """Replace the enable mask with upstream flag names; () clears it, None preserves it. Backend limits apply."""
    warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)
    """MJWarp-only allocation and execution settings; retained but unused when the backend is classic."""

    @model_validator(mode="after")
    def validate_ownership(self) -> MujocoConfig:
        for name in self.extra:
            if name == "timestep":
                raise ValueError(f"extra.{name}: sim.fps owns timestep; set sim.fps instead")
            if name in self.__dataclass_fields__:
                raise ValueError(f"extra.{name}: use the explicit {name} field")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MujocoGeomPhysicsConfig(NativeConfig):
    """Surface contact overrides, shared by scene objects and robot link_physics."""

    friction: tuple[Finite, Finite, Finite] | None = None
    """Sliding, torsional and rolling friction coefficients; applied to each geom, None preserves the asset."""
    solref: tuple[Finite, Finite] | None = None
    """Contact reference: positive (time constant in seconds,damping ratio) or nonpositive (-stiffness,-damping).
    None preserves each geom's values."""
    solimp: tuple[Finite, Finite, Finite, Finite, Finite] | None = None
    """Contact impedance (dmin,dmax,width,midpoint,power); width is in metres. None preserves each geom's values."""
    condim: Literal[1, 3, 4, 6] | None = None
    """Contact dimensions: 1 normal, 3 adds sliding, 4 adds torsion, 6 adds rolling. None preserves the asset."""
    contype: SignedInt | None = None
    """Collision-type bitmask tested against the other geom's conaffinity; zero disables that direction.
    None preserves the asset mask."""
    conaffinity: SignedInt | None = None
    """Geom collision-affinity bitmask; either direction's contype intersection enables collision. None preserves it."""
    priority: SignedInt | None = None
    """Higher-priority geom supplies mixed contact parameters; equal priority uses native mixing.
    None preserves each geom's priority."""
    solmix: Finite | None = None
    """Relative weight for equal-priority solver-parameter mixing; applied per geom. None preserves the asset."""
    margin: Finite | None = None
    """Contact detection distance in metres; contacts activate below margin-gap. None preserves each geom's value."""
    gap: Finite | None = None
    """Inactive-contact band inside margin, in metres; contacts above margin-gap exert no force.
    None preserves each geom's value."""

    @model_validator(mode="after")
    def validate_contact(self) -> MujocoGeomPhysicsConfig:
        if self.friction is not None and any(v < 0 for v in self.friction):
            raise ValueError("friction coefficients must be nonnegative")
        if self.solref is not None and not (all(v > 0 for v in self.solref) or all(v <= 0 for v in self.solref)):
            raise ValueError("solref requires positive time/damping or nonpositive stiffness/damping")
        if self.solimp is not None:
            lo, hi, width, midpoint, power = self.solimp
            if not (0 <= lo <= 1 and 0 <= hi <= 1 and width >= 0 and 0 <= midpoint <= 1 and power >= 1):
                raise ValueError("solimp requires dmin/dmax/midpoint in [0,1], width >= 0, power >= 1")
        for name in self.extra:
            if name in self.__dataclass_fields__:
                raise ValueError(f"extra.{name}: use the explicit {name} field")
        return self
