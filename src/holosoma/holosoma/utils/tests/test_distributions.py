"""Unit tests for the shared DR distribution sampler (backend-free, Mac-runnable).

These exercise :class:`DistributionSpec` validation and the inverse-CDF shaping statistics with no
simulator dependency, so they run anywhere torch is importable. Production draws ONLY through the
keyed :class:`TermSampler` (covered in test_keyed_distributions.py); there is no public stateful
``sample`` — these tests apply the same inverse-CDF transform to a block of uniforms directly.
"""

import math

import pytest
import torch

from holosoma.utils.sampler import DistributionSpec, _inverse_cdf, quantiles

_N = 200_000
_DEVICE = "cpu"


def _draw(spec: DistributionSpec) -> torch.Tensor:
    """Continuous draw of ``spec`` via the SAME inverse-CDF the keyed sampler uses (test-only)."""
    torch.manual_seed(0)
    u = torch.rand(_N, device=_DEVICE)
    return _inverse_cdf(u, spec)


# --------------------------------------------------------------------------------------------------
# Validation (errors fire at construction, identically for every backend)
# --------------------------------------------------------------------------------------------------


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        DistributionSpec(kind="gaussain", low=0.0, high=1.0)  # type: ignore[arg-type]


def test_uniform_requires_bounds():
    with pytest.raises(ValueError, match="requires both 'low' and 'high'"):
        DistributionSpec(kind="uniform", low=1.0)


def test_high_below_low_raises():
    with pytest.raises(ValueError, match="must be >= low"):
        DistributionSpec(kind="uniform", low=1.0, high=0.0)


@pytest.mark.parametrize("bounds", [(-1.0, 3.0), (0.0, 2.0), (-2.0, -1.0)])
def test_log_uniform_non_positive_bounds_raise(bounds):
    with pytest.raises(ValueError, match="strictly positive bounds"):
        DistributionSpec(kind="log_uniform", low=bounds[0], high=bounds[1])


def test_uniform_rejects_mean_std():
    with pytest.raises(ValueError, match="does not accept 'mean'/'std'"):
        DistributionSpec(kind="uniform", low=0.0, high=1.0, mean=0.5, std=0.1)


def test_gaussian_requires_params_or_bounds():
    with pytest.raises(ValueError, match="requires either explicit"):
        DistributionSpec(kind="gaussian")


def test_gaussian_partial_mean_std_raises():
    with pytest.raises(ValueError, match="together, or neither"):
        DistributionSpec(kind="gaussian", mean=1.0)


def test_gaussian_negative_std_raises():
    with pytest.raises(ValueError, match="std must be >= 0"):
        DistributionSpec(kind="gaussian", mean=1.0, std=-0.1)


# --------------------------------------------------------------------------------------------------
# parse parsing
# --------------------------------------------------------------------------------------------------


def test_parse_pair_is_uniform():
    spec = DistributionSpec.parse([1.0, 4.0])
    assert spec.kind == "uniform"
    assert (spec.low, spec.high) == (1.0, 4.0)


def test_parse_dict_explicit_kind_and_params():
    spec = DistributionSpec.parse({"kind": "gaussian", "low": -1.0, "high": 1.0, "mean": 0.0, "std": 0.25})
    assert spec.kind == "gaussian"
    assert spec.resolved_mean_std() == (0.0, 0.25)


def test_parse_dict_range_key():
    spec = DistributionSpec.parse({"kind": "uniform", "range": [2.0, 5.0]})
    assert (spec.low, spec.high) == (2.0, 5.0)


def test_parse_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown distribution spec keys"):
        DistributionSpec.parse({"kind": "uniform", "low": 0.0, "high": 1.0, "loww": 2.0})


def test_parse_range_and_low_high_conflict_raises():
    # `range` and explicit low/high together is ambiguous -> fail loudly (don't silently let one win).
    with pytest.raises(ValueError, match="either 'range' or 'low'/'high'"):
        DistributionSpec.parse({"range": [0.0, 1.0], "low": 5.0})


def test_parse_passthrough_spec():
    spec = DistributionSpec(kind="uniform", low=0.0, high=1.0)
    assert DistributionSpec.parse(spec) is spec


# --------------------------------------------------------------------------------------------------
# Sampling statistics
# --------------------------------------------------------------------------------------------------


def test_uniform_in_band_and_mean():
    spec = DistributionSpec(kind="uniform", low=-2.0, high=6.0)
    x = _draw(spec)
    assert x.min() >= -2.0 and x.max() <= 6.0
    assert abs(x.mean().item() - 2.0) < 0.05


def test_log_uniform_in_band_and_log_mean():
    lo, hi = 1.0, 100.0
    spec = DistributionSpec(kind="log_uniform", low=lo, high=hi)
    x = _draw(spec)
    assert x.min() >= lo - 1e-4 and x.max() <= hi + 1e-4
    # Uniform in log-space -> mean of log(x) ~ midpoint of [log lo, log hi].
    expected_log_mean = 0.5 * (math.log(lo) + math.log(hi))
    assert abs(torch.log(x).mean().item() - expected_log_mean) < 0.02


def test_gaussian_bounds_derive_mean_std_and_truncate():
    lo, hi = -1.0, 1.0
    spec = DistributionSpec(kind="gaussian", low=lo, high=hi)
    x = _draw(spec)
    # Truncated strictly inside the band, centered, std ~ (hi-lo)/6.
    assert x.min() >= lo and x.max() <= hi
    assert abs(x.mean().item()) < 0.02
    assert abs(x.std().item() - (hi - lo) / 6.0) < 0.03


def test_gaussian_signed_band_centers_at_zero():
    spec = DistributionSpec(kind="gaussian", low=-0.025, high=0.025)
    x = _draw(spec)
    assert x.min() >= -0.025 and x.max() <= 0.025
    assert abs(x.mean().item()) < 1e-3


def test_gaussian_explicit_mean_std_no_bounds_untruncated():
    spec = DistributionSpec(kind="gaussian", mean=5.0, std=2.0)
    x = _draw(spec)
    assert abs(x.mean().item() - 5.0) < 0.05
    assert abs(x.std().item() - 2.0) < 0.05


def test_gaussian_one_sided_low_truncates_below_only():
    # Half-open [low, +inf): values never below low, but unbounded above; mean shifts up vs the
    # untruncated 0 because the lower tail is removed.
    spec = DistributionSpec(kind="gaussian", mean=0.0, std=1.0, low=0.0)
    x = _draw(spec)
    assert x.min() >= 0.0
    assert x.max() > 2.0  # upper tail intact (unbounded above)
    assert x.mean().item() > 0.5  # lower half removed -> mean pulled up (half-normal mean ~0.8)


def test_gaussian_one_sided_high_truncates_above_only():
    spec = DistributionSpec(kind="gaussian", mean=0.0, std=1.0, high=0.0)
    x = _draw(spec)
    assert x.max() <= 0.0
    assert x.min() < -2.0  # lower tail intact
    assert x.mean().item() < -0.5  # upper half removed -> mean pulled down


def test_gaussian_one_sided_requires_explicit_mean_std():
    # A lone bound cannot derive (mean, std) -> must raise (fail loud, don't guess).
    with pytest.raises(ValueError, match="single bound"):
        DistributionSpec(kind="gaussian", low=0.0)
    with pytest.raises(ValueError, match="single bound"):
        DistributionSpec(kind="gaussian", high=1.0)


def test_gaussian_explicit_params_with_bounds_truncate():
    # Explicit (mean, std) but bounds present -> clamp via truncation to the band.
    spec = DistributionSpec(kind="gaussian", low=0.0, high=1.0, mean=0.5, std=5.0)
    x = _draw(spec)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_gaussian_zero_std_is_constant():
    spec = DistributionSpec(kind="gaussian", mean=3.0, std=0.0)
    x = _draw(spec)
    assert torch.allclose(x, torch.full_like(x, 3.0))


def test_degenerate_equal_bounds_constant():
    spec = DistributionSpec(kind="gaussian", low=2.0, high=2.0)
    x = _draw(spec)
    assert torch.allclose(x, torch.full_like(x, 2.0))


def test_gaussian_zero_std_with_bounds_clamps_to_band():
    # Explicit mean outside [low, high] with std=0 -> constant clamped to the boundary.
    spec = DistributionSpec(kind="gaussian", low=0.0, high=1.0, mean=5.0, std=0.0)
    x = _draw(spec)
    assert torch.allclose(x, torch.full_like(x, 1.0))


def test_gaussian_equal_bounds_explicit_std_constant():
    # low == high with explicit std>0 -> constant at the bound (truncated-normal short-circuit).
    spec = DistributionSpec(kind="gaussian", low=2.0, high=2.0, mean=2.0, std=0.5)
    x = _draw(spec)
    assert torch.allclose(x, torch.full_like(x, 2.0))


def test_seeded_reproducible():
    spec = DistributionSpec(kind="gaussian", low=-1.0, high=1.0)
    a = _draw(spec)
    b = _draw(spec)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------------------------------
# quantiles() — the bucket-fill primitive for PhysX-material backends
# --------------------------------------------------------------------------------------------------


def test_quantiles_count_and_in_band():
    spec = DistributionSpec(kind="uniform", low=2.0, high=5.0)
    q = quantiles(spec, 64, _DEVICE)
    assert q.shape == (64,)
    assert q.min() >= 2.0 and q.max() <= 5.0


def test_quantiles_uniform_are_evenly_spaced():
    # Uniform quantiles at (i+0.5)/n are the midpoints of n equal sub-intervals.
    spec = DistributionSpec(kind="uniform", low=0.0, high=1.0)
    q = quantiles(spec, 10, _DEVICE)
    expected = (torch.arange(10, dtype=torch.float32) + 0.5) / 10.0
    assert torch.allclose(q, expected, atol=1e-6)


def test_quantiles_deterministic_and_sorted():
    # quantiles has NO internal shuffle/global-RNG: same call -> identical, sorted ascending. Any
    # shuffle is the caller's job (keyed, in the material writer).
    spec = DistributionSpec(kind="uniform", low=0.0, high=1.0)
    q1 = quantiles(spec, 64, _DEVICE)
    q2 = quantiles(spec, 64, _DEVICE)
    assert torch.equal(q1, q2)
    assert torch.equal(q1, q1.sort().values)


def test_quantiles_uniform_selection_reproduces_distribution():
    # The bucket scheme: uniform-select (with replacement) over quantile values. The realized
    # marginal must match a direct continuous draw of the same spec (this is the whole point).
    spec = DistributionSpec(kind="gaussian", low=-1.0, high=1.0)
    torch.manual_seed(0)
    buckets = quantiles(spec, 256, _DEVICE)
    pick = torch.randint(0, 256, (_N,))
    bucketed = buckets[pick]
    direct = _draw(spec)
    assert bucketed.min() >= -1.0 and bucketed.max() <= 1.0
    assert abs(bucketed.mean().item() - direct.mean().item()) < 0.02
    assert abs(bucketed.std().item() - direct.std().item()) < 0.02


def test_quantiles_log_uniform_positive_and_log_mean():
    spec = DistributionSpec(kind="log_uniform", low=1.0, high=100.0)
    q = quantiles(spec, 128, _DEVICE)
    assert q.min() >= 1.0 and q.max() <= 100.0
    # Log-spaced -> mean of log(q) near the midpoint of [log lo, log hi].
    assert abs(torch.log(q).mean().item() - 0.5 * (math.log(1.0) + math.log(100.0))) < 0.05
