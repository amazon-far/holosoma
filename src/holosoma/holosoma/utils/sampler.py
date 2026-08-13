"""Single source of truth for domain-randomization (DR) sampling.

Every DR term, on every backend, draws through the ONE keyed sampler here (:class:`TermSampler`),
driven by the ONE spec (:class:`DistributionSpec`, defined in
:mod:`holosoma.config_types.distribution`). This is what makes a config input mean the *same*
thing on every backend: the spec defines the meaning and is validated once, at construction; the
sampler defines the (counter-based, reproducible) draw; backends differ only in where they write the
result.

Public API
----------
- :class:`DistributionSpec` — the validated MEANING of one randomized scalar (kind + bounds /
  params); re-exported here for convenience, defined in :mod:`holosoma.config_types.distribution`
  (an import-light module, so a config schema can name it without pulling torch). A term passes a
  config range value (see :data:`DistributionLike`) and the sampler parses it via
  :meth:`DistributionSpec.parse`.
- :class:`TermSampler` — the keyed draw. The randomization manager binds one per term (it holds the
  seed, term name, lifecycle stage, and per-env episode counter) and passes it to the term; the term
  calls :meth:`TermSampler.draw` / :meth:`~TermSampler.draw_int` / :meth:`~TermSampler.permute`. A term
  receives a bound sampler rather than constructing one.
- :func:`quantiles` — a deterministic stratified bucket-fill, for backends whose physics API caps the
  number of unique materials (PhysX): fill N buckets, then select per shape with a keyed draw.

The distribution conventions (uniform / log_uniform / gaussian, bounds-first) are documented on
:class:`DistributionSpec`.

Keyed draws
-----------
A draw is ``value = f(coordinates)``: a hash of its coordinates, with no carried state. Two runs
with the same seed produce identical values for the same coordinate, independent of which other
terms exist, the order they run in, the number of envs, and which env subset is resetting. The
coordinate model is documented on :class:`TermSampler`.

Examples
--------
**Build a spec from a config range value** (what the sampler does internally; useful in tests):
>>> DistributionSpec.parse([0.5, 1.5])                       # bare pair -> uniform bounds
DistributionSpec(kind='uniform', low=0.5, high=1.5, mean=None, std=None)
>>> DistributionSpec.parse({"kind": "log_uniform", "low": 0.8, "high": 1.2})  # explicit kind
DistributionSpec(kind='log_uniform', low=0.8, high=1.2, mean=None, std=None)
>>> DistributionSpec.parse({"kind": "gaussian", "low": -1.0, "high": 1.0})  # non-default via dict
DistributionSpec(kind='gaussian', low=-1.0, high=1.0, mean=None, std=None)

**Draw inside a DR term** (you are given a bound ``sampler`` and the resetting ``env_ids``):
>>> # one value per env (a global scalar randomized per environment) -> shape [E]
>>> friction = sampler.draw([0.5, 1.5], env_ids=env_ids)
>>> # one value per (env, body): pass the body ids as a [1, N] coord -> shape [E, N]
>>> per_body = sampler.draw([0.9, 1.1], env_ids=env_ids, coords=(body_ids[None, :],))
>>> # several independent quantities from one term: give each a distinct int "stream" coord
>>> com_x = sampler.draw(x_range, env_ids=env_ids, coords=(0,))   # CoM x  (stream 0)
>>> com_y = sampler.draw(y_range, env_ids=env_ids, coords=(1,))   # CoM y  (stream 1, decorrelated)

**Reproducibility you can rely on** (same coordinate -> same value, regardless of how it is passed):
>>> a = sampler.draw([0.0, 1.0], env_ids=torch.arange(8))            # full population
>>> b = sampler.draw([0.0, 1.0], env_ids=torch.tensor([2, 5]))       # a resetting subset
>>> torch.equal(a[[2, 5]], b)                                        # subset rows match the full draw
True
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

# DistributionSpec and the config range-value types live in an import-light module so a config schema
# can name them without pulling torch; re-exported here so a term keeps a single import for the spec
# it parses and the sampler it draws through.
from holosoma.config_types.distribution import (  # noqa: F401  (re-export)
    _SUPPORTED_DISTRIBUTIONS,
    Distribution,
    DistributionLike,
    DistributionSpec,
)

# Phi(x) = 0.5 * (1 + erf(x / sqrt(2))); Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1).
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_SQRT2 = math.sqrt(2.0)
# Keep the inverse-CDF argument off the ±1 singularities of erfinv (which map to ±inf).
_P_EPS = 1e-7


def _inverse_cdf(u: torch.Tensor, spec: DistributionSpec) -> torch.Tensor:
    """Map uniforms ``u`` in [0, 1] to values of ``spec`` via its inverse CDF.

    This is the SINGLE transform shared by :meth:`TermSampler.draw` (keyed ``u``) and
    :func:`quantiles` (stratified ``u``), so a continuous draw and the bucket-fill agree in
    distribution.
    """
    if spec.kind == "uniform":
        lo, hi = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        return lo + (hi - lo) * u

    if spec.kind == "log_uniform":
        lo, hi = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        log_lo, log_hi = math.log(lo), math.log(hi)
        return torch.exp(log_lo + (log_hi - log_lo) * u)

    # gaussian — truncated to whichever bounds are present (none / low-only / high-only / both).
    mean, std = spec.resolved_mean_std()
    bound_lo: float | None = float(spec.low) if spec.low is not None else None
    bound_hi: float | None = float(spec.high) if spec.high is not None else None

    if std == 0.0:
        # Degenerate: a point mass at ``mean``, clamped into whatever bounds exist.
        return torch.clamp(
            u * 0.0 + mean,
            min=bound_lo if bound_lo is not None else -math.inf,
            max=bound_hi if bound_hi is not None else math.inf,
        )

    if bound_lo is not None and bound_hi is not None and bound_lo == bound_hi:
        return torch.full_like(u, bound_lo)

    # Map u into the CDF sub-interval [Phi(a), Phi(b)], then invert. A missing bound contributes the
    # open end of the unit interval (Phi(-inf)=0 / Phi(+inf)=1), so this one path covers the
    # un-truncated draw (no bounds -> [0, 1]), a one-sided truncation, and a two-sided one. Phi bounds
    # are scalars computed as Python floats (Phi(x)=0.5*(1+erf(x/sqrt2))) — no per-draw tensor alloc.
    cdf_a = 0.5 * (1.0 + math.erf((bound_lo - mean) / std * _INV_SQRT2)) if bound_lo is not None else 0.0
    cdf_b = 0.5 * (1.0 + math.erf((bound_hi - mean) / std * _INV_SQRT2)) if bound_hi is not None else 1.0
    p = cdf_a + u * (cdf_b - cdf_a)
    p = torch.clamp(p, min=_P_EPS, max=1.0 - _P_EPS)
    x = mean + std * _SQRT2 * torch.erfinv(2.0 * p - 1.0)
    # Guard against any residual numerical drift past the present bound(s).
    return torch.clamp(
        x,
        min=bound_lo if bound_lo is not None else -math.inf,
        max=bound_hi if bound_hi is not None else math.inf,
    )


def quantiles(
    spec: DistributionSpec,
    n: int,
    device: str | torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``n`` stratified quantile values of ``spec``, SORTED (inverse CDF at ``p = (i+0.5)/n``).

    Selecting (with replacement) over these ``n`` values reproduces ``spec``'s distribution as an
    ``n``-atom staircase — the BUCKET FILL for backends whose physics API caps the number of unique
    materials (PhysX allows only ~64k), where a per-shape continuous draw is impossible. Both this and
    :meth:`TermSampler.draw` go through :func:`_inverse_cdf`, so the bucket marginal matches a
    continuous keyed draw and converges to it as ``n`` grows (stratification keeps it low-variance
    even at ``n=64``).

    Deterministic by construction (no shuffling here): the result is the same sorted vector every call.
    The CALLER owns any per-channel shuffle, and the production caller does it with a KEYED permutation
    (:meth:`TermSampler.permute`) so the realized material is reproducible per seed but the channels are
    not rank-aligned. The usual pattern is "fill once, then select per shape":

    >>> table = quantiles(DistributionSpec(low=0.5, high=1.5), n=64, device="cpu")   # 64 friction atoms
    >>> shuffled = table[sampler.permute(64, (0,))]                  # keyed shuffle, channel stream 0
    >>> picks = sampler.draw_int(0, 63, env_ids=env_ids, coords=(shape_ids[None, :],))  # [E, n_shapes]
    >>> per_shape_friction = shuffled[picks]                         # gather -> each shape's value
    """
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + 0.5) / n  # midpoint grid in (0, 1), strictly interior
    return _inverse_cdf(u, spec)


# ==================================================================================================
# Keyed (counter-based) randomization — reproducible per (term, env, episode), order-independent.
# ==================================================================================================
#
# WHY NOT a plain RNG stream: a sequential generator makes draw N depend on how many draws happened
# before it, so adding/removing/reordering DR terms, changing num_envs, or resetting a subset of envs
# all perturb unrelated values. We instead derive each draw from a hash of its COORDINATES, with no
# carried state: value = f(coordinates). Standard counter-based-RNG idea (Philox/Threefry/JAX); we
# use a SplitMix64-fold, which is enough for DR and tiny to own. The coordinate model (which
# coordinates exist, who supplies them, how they shape the output) is documented on
# :class:`TermSampler`.

_SPLITMIX_IV = np.uint64(0x243F6A8885A308D3)  # nonzero init so all-zero coordinates don't hash to 0
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = (np.uint64(s) for s in (30, 27, 31, 11))


def _term_id(name: str) -> int:
    """Stable 63-bit integer id for a term name (process-independent, unlike builtin ``hash``)."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little") >> 1


def keyed_uniform(*coords: int | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Hash integer ``coords`` to uniforms in ``[0, 1)`` — vectorized, stateless, value = f(coords).

    Each coord is a scalar or an int64 array; NumPy broadcasting sets the output shape, so a per-env
    loop and a single vectorized call produce bit-identical values. Folds the coords through the
    SplitMix64 finalizer (a strong-avalanche bijection on 2^64) and takes the top 53 bits as an exact
    double in ``[0, 1)``.

    The hash stays CPU/numpy: uint64 wraparound is exact and platform-independent, whereas torch
    int64 ``>>`` is arithmetic (not logical), so the SplitMix shifts would need explicit masking.
    """
    arrays = [np.asarray(c).astype(np.int64).astype(np.uint64) for c in coords]
    shape = np.broadcast_shapes(*(a.shape for a in arrays)) if arrays else ()
    with np.errstate(over="ignore"):  # uint64 wraparound is the intended modular arithmetic
        h = np.broadcast_to(_SPLITMIX_IV, shape).copy()
        for a in arrays:
            z = h ^ a
            z = (z ^ (z >> _S30)) * _SPLITMIX_M1
            z = (z ^ (z >> _S27)) * _SPLITMIX_M2
            h = z ^ (z >> _S31)
        u = (h >> _S11).astype(np.float64) * (2.0**-53)
    return torch.as_tensor(np.ascontiguousarray(u), dtype=torch.float64).reshape(u.shape)


# Stage tags folded into the key so a term registered at two lifecycle stages can't collide.
STAGE_SETUP, STAGE_RESET, STAGE_STEP = 0, 1, 2

# Domain tag folded into permute() keys so a permutation can never alias a draw() of the same fold
# arity (draw folds (env, episode) where permute would fold (coord, index) — same-position small
# ints would otherwise hash identically).
_PERMUTE_TAG = _term_id("__permute__")


@dataclass(frozen=True)
class TermSampler:
    """The one sampler a randomization term uses — bound by the manager to a term's coordinates.

    The manager (which alone knows each term's stable name, lifecycle stage, and the per-env episode
    counter) calls :meth:`bind` once and passes the result into the term. The term then draws by
    MEANING — "a value for this config range, these envs, and these caller coordinates" — and gets back
    a tensor that is reproducible per (term, env, episode) and independent of the order or set of
    other terms.

    The draw is a generic keyed function of coordinates (see the module header). The framework fixes
    ``base_seed``, ``term_id``, ``stage``; the manager supplies the per-env ``episode`` counter, zipped
    with ``env_ids`` into one leading output dimension; the term supplies the rest as ``coords`` — a
    tuple in which each entry is a Python ``int`` (no dimension, a distinct stream) or a tensor (a new
    dimension). A per-entity draw is a tensor coord; an independent stream is an int coord.

    This is the single surface for all DR sampling: continuous draws (:meth:`draw`) and discrete
    indices (:meth:`draw_int`, e.g. material bucket selection / action delay). A per-actor loop
    pre-draws the full tensor once and indexes it, rather than drawing one float at a time. Material
    bucket VALUES are filled separately via :func:`quantiles` (deterministic given the spec); only the
    per-shape bucket selection is keyed, through :meth:`draw_int`.

    Coordinate shaping
    ------------------
    The env/episode dimension is always output axis 0 (length ``len(env_ids)``). Each tensor coord adds
    a trailing dimension, placed by giving the coord a leading size-1 env axis:

    ======================================  =========================================  ============
    intent                                  ``coords=``                                output shape
    ======================================  =========================================  ============
    one value per env (a global scalar)     ``()``  (omit it)                          ``[E]``
    two decorrelated per-env quantities     ``(0,)`` and ``(1,)`` in two calls         ``[E]`` each
    one value per (env, entity)             ``(ids[None, :],)``                        ``[E, N]``
    per (env, entity) on stream k           ``(k, ids[None, :])``                      ``[E, N]``
    a per-(env, i, j) matrix / geom-pair    ``(rows[None, :, None], cols[None, None, :])``  ``[E, I, J]``
    ======================================  =========================================  ============

    An ``int`` coord is a stream tag: it adds no dimension and decorrelates this draw from another. A
    tensor coord adds a dimension and keys on its VALUES, so a per-entity draw is reproducible by id,
    not by position. Since ``value = f(coordinate values)``, the same id keyed as a scalar ``int``, a
    1-element tensor, or one cell of a larger tensor draws the same number — only its position in the
    output changes. A per-actor scalar loop and a single vectorized tensor-coord draw therefore agree
    bit-for-bit.

    Coordinate layout is defined by the term: which int means which quantity (CoM x=0 / y=1, the six
    inertia components, link-scale vs base-add mass) is the term's own convention. Renumbering or
    reordering the coords changes the draws, so a term's coordinate scheme is fixed once shipped — a
    change breaks reproducibility against existing runs and checkpoints.

    Examples
    --------
    >>> # In production a bound sampler arrives from the manager; one is built directly here.
    >>> sampler = TermSampler.bind(base_seed=42, term_name="randomize_friction", stage=STAGE_RESET)
    >>> env_ids = torch.arange(4)
    >>> sampler.draw([0.5, 1.5], env_ids=env_ids).shape                       # per-env -> [E]
    torch.Size([4])
    >>> body_ids = torch.tensor([3, 7, 9])
    >>> sampler.draw([0.9, 1.1], env_ids=env_ids, coords=(body_ids[None, :],)).shape   # per-body -> [E, N]
    torch.Size([4, 3])
    >>> # The same body keyed as a scalar stream matches its column in the vectorized draw:
    >>> col = sampler.draw([0.9, 1.1], env_ids=env_ids, coords=(body_ids[None, :],))[:, 1]
    >>> one = sampler.draw([0.9, 1.1], env_ids=env_ids, coords=(int(body_ids[1]),))
    >>> torch.equal(col, one)
    True
    """

    base_seed: int
    term_id: int
    stage: int
    episode: torch.Tensor | None  # per-env episode counter (CPU long), or None at startup (sentinel 0)

    @classmethod
    def bind(
        cls,
        base_seed: int | None,
        term_name: str,
        stage: int,
        episode_count: torch.Tensor | None = None,
    ) -> TermSampler:
        """Bind a sampler to one term invocation. ``episode_count`` is the env's per-env reset counter.

        ``base_seed`` MUST be set, a keyed sampler with no seed cannot be reproducible. Pass the run's
        seed (env ``dr_base_seed`` = the training seed); a seedless run is a configuration error, raised here.
        """
        if base_seed is None:
            raise ValueError(
                "TermSampler requires a base_seed (the run's DR seed); there is no unseeded/global-RNG "
                "fallback. Set the training seed (env.dr_base_seed). A seedless DR run is not reproducible!"
            )
        ep = None if episode_count is None else episode_count.to(device="cpu", dtype=torch.long)
        return cls(base_seed=base_seed, term_id=_term_id(term_name), stage=stage, episode=ep)

    @staticmethod
    def _coord_array(c: int | Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
        """One caller coordinate as an int64 numpy array (a Python int -> 0-d; a tensor -> its shape).

        The VALUE is what keys the draw; the SHAPE only places it. So a coord may be a CUDA tensor of
        stable ids — it is pulled to CPU here — without affecting the drawn value.
        """
        if isinstance(c, torch.Tensor):
            return c.detach().to(device="cpu", dtype=torch.long).numpy()
        return np.asarray(c, dtype=np.int64)

    def _uniforms(
        self,
        env_ids: torch.Tensor,
        coords: Sequence[int | Sequence[int] | np.ndarray | torch.Tensor] = (),
    ):
        """Keyed uniforms for (this term, these envs+episodes, these caller ``coords``).

        The env/episode coordinate is the LEADING output axis: ``env_ids`` and the per-env ``episode``
        counter are ZIPPED elementwise (row k keys on ``(env_ids[k], episode[env_ids[k]])`` TOGETHER,
        NOT an outer-product grid), contributing ONE dimension of length ``len(env_ids)``. Episode is
        gathered per env-id, so resetting an env SUBSET keys each row on that env's own count and the
        draw stays subset-stable. Each entry of ``coords`` is folded as given — a Python ``int`` adds
        no dimension (it shifts the stream); a tensor adds its own dimension(s).

        Output shape = the env axis broadcast against the tensor coords (``np.broadcast_shapes``-style,
        right-aligned). The env axis sits on axis 0, so a caller shapes a tensor coord with a leading
        size-1 env axis: ``ids[None, :]`` (``[1, N]``) yields ``[E, N]``; ``rows[None, :, None]`` +
        ``cols[None, None, :]`` yields ``[E, R, C]``. No coords -> ``[E]``; only int coords -> ``[E]``.
        """
        env_np = env_ids.to(device="cpu", dtype=torch.long).numpy()
        if self.episode is None:
            ep_np = np.zeros_like(env_np)  # startup sentinel: no episode counter yet
        else:
            ep = self.episode.numpy()
            if env_np.size and int(env_np.max()) >= ep.shape[0]:
                raise IndexError(
                    f"episode counter has length {ep.shape[0]} but env_ids reference index "
                    f"{int(env_np.max())}; the per-env episode tensor must cover every drawn env id "
                    "(env and episode are zipped into one dimension and must be the same population)."
                )
            ep_np = ep[env_np]  # gather per env-id: row k -> (env_ids[k], episode[env_ids[k]])
        coord_arrays = [self._coord_array(c) for c in coords]
        # env/episode sit on axis 0; pad with trailing singletons so their E aligns there against each
        # tensor coord's leading (size-1) env axis. Scalar-int coords (rank 0) add nothing -> stays [E].
        pad = max(0, max((a.ndim for a in coord_arrays), default=0) - 1)
        env_shape = (env_np.shape[0],) + (1,) * pad
        env_k = env_np.reshape(env_shape)
        ep_k = ep_np.reshape(env_shape)
        u = keyed_uniform(self.base_seed, self.term_id, self.stage, env_k, ep_k, *coord_arrays)
        return u.to(torch.float32)

    def draw(
        self,
        spec: DistributionLike,
        *,
        env_ids: torch.Tensor,
        coords: Sequence[int | Sequence[int] | np.ndarray | torch.Tensor] = (),
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Draw continuous values for one config range value, keyed and reproducible per (term, env, episode).

        See the class "Coordinate shaping" section for how ``coords`` sets the output shape: leading
        axis = env/episode, each tensor coord = a trailing dimension, each int coord = a decorrelated
        stream.

        Parameters
        ----------
        spec : a :data:`DistributionLike` — a ``[lo, hi]`` pair (always uniform), a ``{kind, low, high,
            mean, std}`` dict, or a ready :class:`DistributionSpec`. Parsed via
            :meth:`DistributionSpec.parse`.
        env_ids : the envs to draw for (1-D long tensor). Becomes output axis 0; zipped with the bound
            per-env ``episode`` counter — so resetting a subset draws the SAME values those envs got in
            a full draw at the same episode.
        coords : tuple of extra coordinates (ints = streams, tensors = dimensions). Default ``()`` -> one
            value per env, shape ``[E]``.
        device : device of the returned tensor (the hash itself always runs on CPU; see the module note).

        Returns
        -------
        torch.Tensor : float32, shape ``[E, *coord_dims]`` (just ``[E]`` with no tensor coords).

        Examples
        --------
        >>> sampler.draw([0.5, 1.5], env_ids=env_ids)                          # per-env friction -> [E]
        >>> sampler.draw({"kind": "gaussian", "low": -1, "high": 1}, env_ids=env_ids)  # truncated normal
        >>> sampler.draw([0.9, 1.1], env_ids=env_ids, coords=(body_ids[None, :],))     # per-(env, body)
        """
        parsed = DistributionSpec.parse(spec)
        u = self._uniforms(env_ids, coords).to(device)
        return _inverse_cdf(u, parsed)

    def draw_int(
        self,
        low: int,
        high: int,
        *,
        env_ids: torch.Tensor,
        coords: Sequence[int | Sequence[int] | np.ndarray | torch.Tensor] = (),
    ) -> torch.Tensor:
        """Keyed discrete draw of integers in ``[low, high]`` INCLUSIVE; same coord shaping as :meth:`draw`.

        For material bucket selection and discrete delays — anywhere you need a reproducible integer
        index per (term, env, episode) rather than a continuous value. Maps a keyed uniform onto the
        integer range (each of the ``high - low + 1`` values equiprobable).

        Examples
        --------
        >>> sampler.draw_int(0, 3, env_ids=env_ids)                              # a per-env delay -> [E]
        >>> sampler.draw_int(0, 63, env_ids=env_ids, coords=(shape_ids[None, :]))  # per-shape bucket -> [E, S]
        """
        span = high - low + 1
        if span <= 0:
            raise ValueError(f"draw_int requires high >= low, got [{low}, {high}].")
        u = self._uniforms(env_ids, coords)
        return low + (u * span).floor().clamp(max=span - 1).to(torch.long)

    def permute(self, n: int, coords: Sequence[int | Sequence[int] | np.ndarray | torch.Tensor] = ()) -> torch.Tensor:
        """A keyed permutation of ``range(n)``, reproducible per (term, stage, ``coords``), env-INDEPENDENT.

        Unlike :meth:`draw`/:meth:`draw_int`, it takes no ``env_ids`` and does not key on env/episode.
        It shuffles a SHARED table that every env then indexes into — the PhysX material bucket case:
        fill one bucket column with :func:`quantiles`, ``permute`` it once (reproducibly, so the
        realized material is stable per seed), then have each (env, shape) pick a bucket via
        :meth:`draw_int`. Pass a distinct int coord per channel so several channels' shuffles are
        independent (not rank-aligned). Returns a ``[n]`` long tensor that is a true permutation.

        Examples
        --------
        >>> order = sampler.permute(64, (0,))           # channel 0's shuffle of 64 buckets, reproducible
        >>> sorted(order.tolist()) == list(range(64))
        True
        """
        coord_arrays = [self._coord_array(c) for c in coords]
        u = keyed_uniform(
            self.base_seed, self.term_id, self.stage, _PERMUTE_TAG, *coord_arrays, np.arange(n, dtype=np.int64)
        )
        return torch.argsort(u)
