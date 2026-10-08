"""Preserve broad motion coverage while prioritizing failures in PHP training."""

import pytest
import torch

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler

pytestmark = pytest.mark.no_sim


def make_sampler():
    return AdaptiveTimestepsSampler(
        device="cpu",
        motion_time_step_total=49950,
        env_fps=50,
        adaptive_kernel_size=3,
        adaptive_uniform_ratio=1e-6,
    )


def test_no_failures_samples_uniformly():
    sampler = make_sampler()
    p = sampler.sampling_probabilities
    torch.testing.assert_close(p, torch.full_like(p, 1 / sampler.num_bins))


def test_first_failure_retains_unfailed_motion_coverage():
    sampler = make_sampler()
    sampler.current_bin_failed_count[500] = 1
    sampler.update_bin_failed_count()
    p = sampler.sampling_probabilities
    unaffected = torch.ones(sampler.num_bins, dtype=torch.bool)
    unaffected[498:501] = False
    assert p[unaffected].sum().item() > 0.996
    assert p[500] > p[100]
    torch.testing.assert_close(p.sum(), torch.tensor(1.0))


def test_separate_failure_waves_are_accumulated():
    sampler = make_sampler()
    sampler.update_current_bin_failed_count(torch.tensor([1000, 1000]))
    sampler.update_current_bin_failed_count(torch.tensor([1000]))
    assert sampler.current_bin_failed_count.sum().item() == 3
    sampler.update_bin_failed_count()
    torch.testing.assert_close(sampler.bin_failed_count.sum(), torch.tensor(0.003))
    assert sampler.current_bin_failed_count.count_nonzero() == 0
