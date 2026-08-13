"""The validated meaning of one randomized scalar: :class:`DistributionSpec` and the config range
values (:data:`DistributionLike`) it is parsed from.

Import-light (no torch/numpy), so a config schema can depend on this without pulling the sampler.
The keyed sampler that DRAWS from a spec lives in :mod:`holosoma.utils.sampler`.

Distribution conventions (bounds-first, with optional explicit parameters)
--------------------------------------------------------------------------
- ``uniform``     — ``U[low, high]``.
- ``log_uniform`` — ``exp(U[log low, log high])``; requires ``low > 0`` and ``high > 0``.
- ``gaussian``    — a normal sampled from ``(mean, std)``. ``mean``/``std`` are taken explicitly when
  given, else derived from BOTH bounds as ``mean = (low + high) / 2`` and ``std = (high - low) / 6``
  (so ``[low, high]`` spans ±3 std). Truncation follows whichever bounds are present (TRUE truncated
  normal via inverse-CDF, no endpoint point-mass): both bounds -> ``[low, high]``; ONE-SIDED (only
  ``low`` or only ``high``, which requires explicit ``(mean, std)`` since one number can't derive
  them) -> a half-open truncation ``[low, +inf)`` / ``(-inf, high]``; no bounds (explicit ``(mean,
  std)`` only) -> an un-truncated normal. Truncation keeps a physical quantity (mass, damping, CoM
  offset) inside the configured band.

Examples
--------
>>> DistributionSpec.parse([0.5, 1.5])                              # bare pair -> uniform bounds
DistributionSpec(kind='uniform', low=0.5, high=1.5, mean=None, std=None)
>>> DistributionSpec.parse({"kind": "log_uniform", "low": 0.8, "high": 1.2})  # explicit kind
DistributionSpec(kind='log_uniform', low=0.8, high=1.2, mean=None, std=None)
>>> DistributionSpec.parse({"kind": "gaussian", "low": -1.0, "high": 1.0})  # truncated normal
DistributionSpec(kind='gaussian', low=-1.0, high=1.0, mean=None, std=None)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, Union, get_args

Distribution = Literal["uniform", "log_uniform", "gaussian"]
"""Sampling distributions a randomization term may request."""

# Derived from the Literal above so the two can't drift (one source of truth for the allowed kinds).
_SUPPORTED_DISTRIBUTIONS: tuple[str, ...] = get_args(Distribution)

# One randomized scalar's range, as written in a config: anything DistributionSpec.parse accepts.
# This is the type a term takes from config and forwards to the sampler — three equivalent forms:
#   - a [lo, hi] pair        -> uniform bounds (the common case)
#   - a {kind, low, high, mean, std} dict -> a non-uniform distribution (gaussian/log_uniform), or one
#                               with explicit params; also the form a DistributionSpec round-trips to
#                               through a JSON config save (so a loaded config arrives as this dict)
#   - a DistributionSpec     -> already-parsed (programmatic) passthrough
# All three are accepted (rather than requiring DistributionSpec) because configs are written and
# serialized as pairs/dicts, and a bare pair is the most common term input.
DistributionLike = Union["DistributionSpec", Sequence[float], dict]


@dataclass(frozen=True)
class DistributionSpec:
    """The validated meaning of one randomized scalar.

    Built once (typically via :meth:`parse` at a term boundary) and drawn through a bound
    ``TermSampler``. ``low``/``high`` are the configured bounds (``[lo, hi]``). For ``gaussian``
    only, ``mean`` and ``std`` may be supplied explicitly; otherwise they are derived from the bounds.
    Validation runs in :meth:`__post_init__`, so an invalid combination (unknown kind, non-positive
    ``log_uniform`` bound, missing parameters, ``high < low``) raises AT CONSTRUCTION — a config typo
    fails loudly at the term boundary, never as a silent NaN deep in the physics write.

    Frozen + validated-once means a spec is a safe, hashable value you can build, stash, and reuse.

    Fields
    ------
    kind : "uniform" | "log_uniform" | "gaussian"
    low, high : the configured bounds. Required for uniform/log_uniform; for gaussian they are the
        truncation band (and, absent explicit params, the source of the derived mean/std).
    mean, std : gaussian ONLY, and only when you want to set them explicitly (e.g. a tight std inside a
        wide band). Supply both or neither.

    Examples
    --------
    >>> DistributionSpec(low=0.5, high=1.5)                       # uniform is the default kind
    DistributionSpec(kind='uniform', low=0.5, high=1.5, mean=None, std=None)
    >>> DistributionSpec(kind="gaussian", low=-1.0, high=1.0)     # truncated normal, mean/std derived
    DistributionSpec(kind='gaussian', low=-1.0, high=1.0, mean=None, std=None)
    >>> DistributionSpec(kind="gaussian", mean=2.5, std=0.1)      # explicit params, un-truncated
    DistributionSpec(kind='gaussian', low=None, high=None, mean=2.5, std=0.1)
    >>> DistributionSpec(kind="log_uniform", low=-1.0, high=1.0)  # invalid: bounds must be positive
    Traceback (most recent call last):
        ...
    ValueError: log_uniform requires strictly positive bounds, got [-1.0, 1.0].
    """

    kind: Distribution = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"unknown distribution '{self.kind}'. Expected one of {_SUPPORTED_DISTRIBUTIONS}.")

        # NaN/inf slip past the ordering checks below (NaN compares False) and would surface as
        # silent NaN deep in a physics write; reject them here.
        for name in ("low", "high", "mean", "std"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"distribution '{self.kind}': '{name}' must be finite, got {v!r}.")

        if self.kind in ("uniform", "log_uniform"):
            if self.low is None or self.high is None:
                raise ValueError(f"distribution '{self.kind}' requires both 'low' and 'high' bounds.")
            if self.high < self.low:
                raise ValueError(f"distribution '{self.kind}': high ({self.high}) must be >= low ({self.low}).")
            if self.kind == "log_uniform" and (self.low <= 0.0 or self.high <= 0.0):
                raise ValueError(f"log_uniform requires strictly positive bounds, got [{self.low}, {self.high}].")
            if self.mean is not None or self.std is not None:
                raise ValueError(f"distribution '{self.kind}' does not accept 'mean'/'std' (bounds only).")

        else:  # gaussian
            if (self.mean is None) != (self.std is None):
                raise ValueError("gaussian: supply 'mean' and 'std' together, or neither.")
            has_explicit = self.mean is not None and self.std is not None
            has_both_bounds = self.low is not None and self.high is not None
            one_sided = (self.low is None) != (self.high is None)  # exactly one bound set
            # (mean, std) can be given explicitly, OR derived from BOTH bounds. A ONE-SIDED bound
            # (only low or only high) truncates on that side but cannot derive (mean, std) from a
            # single number, so it requires explicit params.
            if not has_explicit:
                if one_sided:
                    raise ValueError(
                        "gaussian with a single bound (only 'low' or only 'high') needs explicit "
                        "('mean', 'std'); a lone bound cannot derive them. Give both bounds or mean/std."
                    )
                if not has_both_bounds:
                    raise ValueError(
                        "gaussian requires either explicit ('mean', 'std') or ('low', 'high') bounds to derive them."
                    )
            if self.std is not None and self.std < 0.0:
                raise ValueError(f"gaussian: std must be >= 0, got {self.std}.")
            if self.low is not None and self.high is not None and self.high < self.low:
                raise ValueError(f"gaussian: high ({self.high}) must be >= low ({self.low}).")

    def resolved_mean_std(self) -> tuple[float, float]:
        """``(mean, std)`` for a gaussian: explicit when given, else derived from the bounds."""
        if self.mean is not None and self.std is not None:
            return float(self.mean), float(self.std)
        # low and high are guaranteed by __post_init__ when explicit params are absent.
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        return 0.5 * (lo + hi), (hi - lo) / 6.0

    @classmethod
    def parse(cls, value: DistributionLike) -> DistributionSpec:
        """Parse one config range value (:data:`DistributionLike`) into a validated spec.

        This is the boundary between loose config input and the validated :class:`DistributionSpec`
        the sampler draws from — ``TermSampler.draw`` calls it for you, so a term hands its config
        value straight through. A bare ``[lo, hi]`` pair is ALWAYS uniform; a non-uniform distribution
        must say so explicitly via the dict ``"kind"`` (or a ready :class:`DistributionSpec`). Accepts:
        - an existing :class:`DistributionSpec` (returned as-is);
        - a ``[lo, hi]`` / ``(lo, hi)`` pair → uniform bounds;
        - a dict ``{"kind"?, "low"?, "high"?, "mean"?, "std"?}`` (``kind`` defaults to ``"uniform"``;
          ``low``/``high`` may instead be given as a ``"range": [lo, hi]`` convenience key).

        Examples
        --------
        >>> DistributionSpec.parse([0.5, 1.5])                          # bare pair -> uniform
        DistributionSpec(kind='uniform', low=0.5, high=1.5, mean=None, std=None)
        >>> DistributionSpec.parse({"kind": "gaussian", "low": -1.0, "high": 1.0})  # explicit kind
        DistributionSpec(kind='gaussian', low=-1.0, high=1.0, mean=None, std=None)
        >>> DistributionSpec.parse({"range": [0.5, 1.5], "kind": "log_uniform"})  # 'range' shorthand
        DistributionSpec(kind='log_uniform', low=0.5, high=1.5, mean=None, std=None)
        >>> spec = DistributionSpec.parse([0.5, 1.5])
        >>> DistributionSpec.parse(spec) is spec                        # spec passes through
        True
        """
        if isinstance(value, DistributionSpec):
            return value
        if isinstance(value, dict):
            params = dict(value)
            if "range" in params:
                if "low" in params or "high" in params:
                    raise ValueError("distribution spec: pass either 'range' or 'low'/'high', not both.")
                rng = list(params.pop("range"))
                if len(rng) != 2:
                    raise ValueError(f"distribution spec: 'range' must have exactly 2 elements [lo, hi], got {rng}.")
                params["low"], params["high"] = rng[0], rng[1]
            kind = params.pop("kind", "uniform")
            unknown = set(params) - {"low", "high", "mean", "std"}
            if unknown:
                raise ValueError(f"unknown distribution spec keys {sorted(unknown)}; expected low/high/mean/std/kind.")
            return cls(kind=kind, **params)
        # Sequence form: [lo, hi] -> uniform.
        pair = list(value)
        if len(pair) != 2:
            raise ValueError(f"range value must have exactly 2 elements [lo, hi], got {pair}.")
        return cls(kind="uniform", low=float(pair[0]), high=float(pair[1]))
