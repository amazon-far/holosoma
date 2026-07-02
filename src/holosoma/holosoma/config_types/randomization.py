"""Configuration types for randomization manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from pydantic.dataclasses import dataclass

# One randomized scalar's range: a [lo, hi] pair, a {kind, low, high, mean, std} dict, or a
# DistributionSpec. Defined in the import-light distribution module (no torch), so this config schema
# names the same range type the sampler parses via DistributionSpec.parse. DistributionSpec is
# imported so pydantic can resolve the forward reference inside the DistributionLike union.
from holosoma.config_types.distribution import DistributionLike, DistributionSpec  # noqa: F401


@dataclass(frozen=True)
class IsaacGymMaterialDR:
    """Material-DR ranges IsaacGym honors (per-shape ``RigidShapeProperties``).

    IsaacGym friction is a SINGLE sliding coefficient (no static/dynamic split), and restitution is
    native per-shape. A field left ``None`` is not randomized. Mirrors
    :class:`~holosoma.config_types.scene.IsaacGymPhysicsConfig`'s channel set.
    """

    friction: DistributionLike | None = None
    """Sliding-friction range. ``None`` keeps the spawned friction."""

    restitution: DistributionLike | None = None
    """Restitution range. ``None`` keeps the spawned restitution."""


@dataclass(frozen=True)
class IsaacSimMaterialDR:
    """Material-DR ranges IsaacSim honors (per-shape PhysX material, via a bucket table).

    IsaacSim splits friction into static/dynamic and has native restitution. Mirrors
    :class:`~holosoma.config_types.scene.IsaacSimPhysicsConfig`'s channel set.
    """

    static_friction: DistributionLike | None = None
    """Static-friction range. ``None`` keeps the spawned value."""

    dynamic_friction: DistributionLike | None = None
    """Dynamic-friction range. ``None`` keeps the spawned value."""

    restitution: DistributionLike | None = None
    """Restitution range. ``None`` keeps the spawned value."""


@dataclass(frozen=True)
class MujocoMaterialDR:
    """Material-DR ranges MuJoCo honors, on a geom's ``geom_friction`` / ``geom_solref`` / ``geom_solimp``.

    MuJoCo has no ``[0,1]`` restitution coefficient — contact bounce/stiffness live in the solver
    reference (``solref = [timeconst, dampratio]``) and impedance (``solimp = [dmin, dmax, width,
    midpoint, power]``). So instead of a faked restitution channel, MuJoCo exposes its real solver
    vectors. Mirrors the randomizable subset of :class:`~holosoma.config_types.scene.MujocoPhysicsConfig`.

    ``solref``/``solimp`` are VECTOR fields, so each is a dict mapping a 0-based component index to its
    range value (e.g. ``{0: [0.01, 0.03]}`` randomizes only ``solref[0]`` = the contact time constant);
    unlisted components keep their spawned value.
    """

    sliding_friction: DistributionLike | None = None
    """Sliding (tangential) friction range. ``None`` keeps the spawned value."""

    solref: dict[Literal[0, 1], DistributionLike] | None = None
    """Per-component ranges for ``geom_solref`` (index 0=timeconst, 1=dampratio). ``None`` -> unchanged."""

    solimp: dict[Literal[0, 1, 2, 3, 4], DistributionLike] | None = None
    """Per-component ranges for ``geom_solimp`` (indices 0..4). ``None`` -> unchanged."""


@dataclass(frozen=True)
class MaterialRandomizationConfig:
    """Per-backend material-DR ranges, mirroring scene.py's ``PhysicsConfig`` backend split.

    Each backend sub-block names ONLY the channels that backend's API honors, so a channel can never
    be silently dropped: if a backend can't do restitution, its block simply has no restitution
    field. A backend whose sub-block is ``None`` randomizes nothing. Each channel range is a
    ``[lo, hi]`` pair (uniform) or a spec dict for a non-uniform distribution.
    """

    isaacgym: IsaacGymMaterialDR | None = None
    isaacsim: IsaacSimMaterialDR | None = None
    mujoco: MujocoMaterialDR | None = None


@dataclass(frozen=True)
class BaseComRange:
    """Per-axis CoM-offset ranges for ``randomize_base_com_startup`` (added to the torso CoM, metres).

    Each axis is a ``[lo, hi]`` pair (uniform), spec dict, or :class:`DistributionSpec`. Offsets
    are typically signed and small, e.g. ``BaseComRange(x=[-0.025, 0.025], y=[-0.05, 0.05], z=[-0.05, 0.05])``.
    """

    x: DistributionLike = (0.0, 0.0)
    y: DistributionLike = (0.0, 0.0)
    z: DistributionLike = (0.0, 0.0)


@dataclass(frozen=True)
class InertiaScale:
    """Per-component inertia-tensor SCALE ranges for ``randomize_object_rigid_body_inertia_startup``.

    Each component is a ``[lo, hi]`` pair (uniform), spec dict, or :class:`DistributionSpec` that
    MULTIPLIES the spawned moment; the default ``(1.0, 1.0)`` leaves a component unchanged, so name
    only what you randomize, e.g. ``InertiaScale(Ixx=[0.5, 1.5])``. MuJoCo stores inertia as the three
    diagonal principal moments and scales only Ixx/Iyy/Izz (the off-diagonals stay at their identity
    default there); IsaacSim/IsaacGym scale all six.
    """

    Ixx: DistributionLike = (1.0, 1.0)
    Iyy: DistributionLike = (1.0, 1.0)
    Izz: DistributionLike = (1.0, 1.0)
    Ixy: DistributionLike = (1.0, 1.0)
    Iyz: DistributionLike = (1.0, 1.0)
    Ixz: DistributionLike = (1.0, 1.0)


@dataclass(frozen=True)
class RandomizationTermCfg:
    """Configuration for a single randomization hook."""

    func: str
    """Import path of the randomization hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class RandomizationManagerCfg:
    """Configuration for the randomization manager."""

    setup_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked every simulation step."""

    ignore_unsupported: bool = False
    """Flag to ignore errors when randomizers are not implemented by a simulator."""
