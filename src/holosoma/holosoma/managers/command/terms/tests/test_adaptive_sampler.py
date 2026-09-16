from __future__ import annotations

import pytest
import torch

from holosoma.managers.command.terms.wbt import (
    AdaptiveTimestepsSampler,
    TwoLayerAdaptiveSampler,
    _apply_prob_cap,
)

pytestmark = pytest.mark.no_sim


def _sampler(**kwargs) -> TwoLayerAdaptiveSampler:
    return TwoLayerAdaptiveSampler(
        motion_start_idx=torch.tensor([0, 3]),
        motion_end_idx=torch.tensor([3, 6]),
        device="cpu",
        env_fps=2,
        adaptive_uniform_ratio=0.0,
        **kwargs,
    )


def test_single_frame_motion_uses_one_phase_bin() -> None:
    sampler = TwoLayerAdaptiveSampler(
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([1]),
        device="cpu",
        env_fps=50,
    )

    assert sampler.K_per_motion.tolist() == [1]
    motion_ids, phases = sampler.sample(8)
    assert motion_ids.tolist() == [0] * 8
    assert bool(torch.all((phases >= 0.0) & (phases < 1.0)))


def test_uniform_motion_weighting_ignores_failure_counts() -> None:
    sampler = _sampler(motion_weighting="uniform")
    sampler.motion_failed_ema[:] = torch.tensor([1.0, 100.0])

    torch.testing.assert_close(sampler.motion_sampling_probabilities, torch.tensor([0.5, 0.5]))


def test_failure_weighted_phase_distribution_uses_failure_rates() -> None:
    sampler = _sampler(motion_weighting="failure", failure_weighted=True)
    sampler.bin_failed_ema[0, :2] = torch.tensor([10.0, 10.0])
    sampler.bin_episode_ema[0, :2] = torch.tensor([10.0, 1.0])

    probabilities = sampler.per_motion_phase_distribution(torch.tensor([0]))[0, :2]

    torch.testing.assert_close(probabilities, torch.tensor([1.0 / 11.0, 10.0 / 11.0]))


@pytest.mark.parametrize("failure_weighted", [False, True])
def test_zero_failure_scores_fall_back_to_uniform(failure_weighted: bool) -> None:
    sampler = _sampler(motion_weighting="failure", failure_weighted=failure_weighted)
    sampler.motion_failed_ema.zero_()
    sampler.bin_failed_ema.zero_()

    torch.testing.assert_close(sampler.motion_sampling_probabilities, torch.tensor([0.5, 0.5]))
    phase_probabilities = sampler.per_motion_phase_distribution(torch.tensor([0, 1]))
    torch.testing.assert_close(phase_probabilities, torch.full((2, 2), 0.5))

    motion_ids, phases = sampler.sample(8)
    assert motion_ids.shape == phases.shape == (8,)


def test_sampler_metrics_report_normalized_uniform_entropy() -> None:
    sampler = _sampler(motion_weighting="uniform")

    sampler.get_stats()

    torch.testing.assert_close(sampler.metrics["motion_sampling_entropy"], torch.tensor(1.0))
    torch.testing.assert_close(sampler.metrics["motion_phase_avg_entropy"], torch.tensor(1.0))


def test_sampler_reset_restores_all_pseudo_counts() -> None:
    sampler = _sampler(
        motion_weighting="failure",
        failure_weighted=True,
        failure_counts_monotonic=True,
    )
    sampler.motion_failed_ema.add_(5.0)
    sampler.bin_failed_ema.add_(5.0)
    sampler.bin_episode_ema.add_(5.0)

    sampler.reset()

    torch.testing.assert_close(sampler.motion_failed_ema, torch.ones(2))
    torch.testing.assert_close(sampler.bin_failed_ema, sampler.bin_valid_mask.float())
    torch.testing.assert_close(sampler.bin_episode_ema, sampler.bin_valid_mask.float())


def test_two_layer_sampler_rejects_noncontiguous_ranges() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        TwoLayerAdaptiveSampler(
            motion_start_idx=torch.tensor([0, 4]),
            motion_end_idx=torch.tensor([3, 6]),
            device="cpu",
            env_fps=2,
        )


def test_legacy_sampler_accepts_empty_failure_updates() -> None:
    sampler = AdaptiveTimestepsSampler(10, "cpu", env_fps=5)

    sampler.update_current_bin_failed_count(torch.empty(0, dtype=torch.long))

    assert torch.count_nonzero(sampler.current_bin_failed_count) == 0


def test_legacy_sampler_without_uniform_floor_starts_uniform() -> None:
    sampler = AdaptiveTimestepsSampler(10, "cpu", env_fps=5, adaptive_uniform_ratio=0.0)

    torch.testing.assert_close(sampler.sampling_probabilities, torch.full((3,), 1.0 / 3.0))
    assert sampler.sample(8).shape == (8,)


def test_probability_cap_preserves_mass_and_ceiling() -> None:
    capped = _apply_prob_cap(torch.tensor([0.8, 0.1, 0.1]), cap=0.5)

    torch.testing.assert_close(capped.sum(), torch.tensor(1.0))
    assert float(capped.max()) <= 0.5
