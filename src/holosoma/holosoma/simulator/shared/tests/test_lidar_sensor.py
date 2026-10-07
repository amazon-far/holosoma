# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Tests for backend-neutral LiDAR ray patterns and return conversion."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from holosoma.config_types.sensor import (
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
    LidarRangeClippingBehavior,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.simulator.shared.lidar_range import finalize_lidar_returns, points_from_ranges
from holosoma.simulator.shared.lidar_sensor import CyclicLidarPatternProvider, LidarPatternProvider
from holosoma.simulator.shared.range_clipping import clip_sensor_ranges
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _provider(
    config: GridLidarRayPatternConfig | CustomLidarRayPatternConfig,
    *,
    publish_hz: float = 1.0,
) -> LidarPatternProvider:
    """Construct a provider through the public configuration/provider boundary."""
    return config.get_provider_cls()(config, device="cpu", publish_hz=publish_hz)


_CYCLIC_SEQUENCE = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
]


def _cyclic_provider(
    *,
    publish_hz: float,
    height: float,
    width: float,
    sequence_offset: float,
) -> CyclicLidarPatternProvider:
    return CyclicLidarPatternProvider(
        torch.tensor(_CYCLIC_SEQUENCE, dtype=torch.float32),
        height=height,  # type: ignore[arg-type]
        width=width,  # type: ignore[arg-type]
        publish_hz=publish_hz,
        sequence_offset=sequence_offset,  # type: ignore[arg-type]
    )


def test_grid_pattern_is_channel_major_and_uses_optical_axes() -> None:
    pattern = _provider(
        GridLidarRayPatternConfig(
            horizontal_angles=[0.0, 90.0],
            vertical_angles=[0.0, 30.0],
        ),
    ).pattern_at(0.0)
    assert (pattern.height, pattern.width) == (2, 2)
    expected = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.5, -math.sqrt(3) / 2],
            [-math.sqrt(3) / 2, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(pattern.directions, expected, atol=1e-6)


def test_uniform_pattern_samples_both_axes_and_excludes_upper_fov_endpoints() -> None:
    pattern = _provider(
        GridLidarRayPatternConfig(
            horizontal_fov=[-180.0, 180.0],
            vertical_fov=[-30.0, 30.0],
            horizontal_resolution=90.0,
            vertical_resolution=30.0,
        ),
    ).pattern_at(0.0)
    assert (pattern.height, pattern.width) == (2, 4)
    assert torch.allclose(pattern.directions[0], torch.tensor([0.0, -0.5, math.sqrt(3) / 2]), atol=1e-6)
    assert torch.allclose(pattern.directions[-1], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


def test_explicit_directions_are_normalized_and_keep_shape() -> None:
    pattern = _provider(
        CustomLidarRayPatternConfig(
            ray_directions=[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            organized_shape=[2, 1],
        ),
    ).pattern_at(0.0)
    assert (pattern.height, pattern.width) == (2, 1)
    assert torch.allclose(pattern.directions, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_pattern_provider_boundary_normalizes_extension_directions() -> None:
    class _ExtensionProvider(LidarPatternProvider):
        def __init__(self) -> None:
            super().__init__(device="cpu", height=1, width=2)

        @property
        def time_varying(self) -> bool:
            return False

        def directions_at(self, sim_time: float) -> torch.Tensor:
            return torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, -5.0]])

    assert torch.equal(
        _ExtensionProvider().pattern_at(0.0).directions,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    )


@pytest.mark.parametrize(
    ("directions", "message"),
    [
        (torch.tensor([[float("nan"), 0.0, 1.0]]), "non-finite"),
        (torch.tensor([[0.0, 0.0, 0.0]]), "zero ray"),
    ],
)
def test_pattern_provider_boundary_rejects_invalid_extension_directions(
    directions: torch.Tensor,
    message: str,
) -> None:
    class _InvalidExtensionProvider(LidarPatternProvider):
        def __init__(self) -> None:
            super().__init__(device="cpu", height=1, width=1)

        @property
        def time_varying(self) -> bool:
            return False

        def directions_at(self, sim_time: float) -> torch.Tensor:
            return directions

    with pytest.raises(ValueError, match=message):
        _InvalidExtensionProvider().pattern_at(0.0)


def test_pattern_provider_boundary_rejects_runtime_shape_changes() -> None:
    class _ShapeChangingExtensionProvider(LidarPatternProvider):
        def __init__(self) -> None:
            super().__init__(device="cpu", height=1, width=2)

        @property
        def time_varying(self) -> bool:
            return True

        def directions_at(self, sim_time: float) -> torch.Tensor:
            ray_count = 2 if sim_time == 0.0 else 3
            return torch.tensor([[0.0, 0.0, -1.0]] * ray_count)

    provider = _ShapeChangingExtensionProvider()
    assert provider.pattern_at(0.0).directions.shape == (2, 3)
    with pytest.raises(RuntimeError, match=r"changed shape: expected \(2, 3\), got torch.Size\(\[3, 3\]\)"):
        provider.pattern_at(0.1)


def test_cyclic_pattern_uses_contiguous_windows_with_a_fixed_shape() -> None:
    provider = _cyclic_provider(publish_hz=10.0, height=2, width=2, sequence_offset=0)

    first = provider.pattern_at(0.0)
    second = provider.pattern_at(0.1)

    assert (first.height, first.width) == (2, 2)
    assert first.directions.shape == second.directions.shape == (4, 3)
    assert torch.equal(first.directions, torch.tensor(_CYCLIC_SEQUENCE[:4]))
    assert torch.equal(second.directions, torch.tensor([*_CYCLIC_SEQUENCE[4:], *_CYCLIC_SEQUENCE[:2]]))


def test_cyclic_pattern_respects_offset_and_wraps() -> None:
    provider = _cyclic_provider(publish_hz=10.0, height=1, width=3, sequence_offset=5)

    assert torch.equal(
        provider.directions_at(0.0),
        torch.tensor([_CYCLIC_SEQUENCE[5], _CYCLIC_SEQUENCE[0], _CYCLIC_SEQUENCE[1]]),
    )
    assert torch.equal(
        provider.directions_at(0.1),
        torch.tensor([_CYCLIC_SEQUENCE[2], _CYCLIC_SEQUENCE[3], _CYCLIC_SEQUENCE[4]]),
    )


def test_cyclic_pattern_normalizes_each_sequence_direction() -> None:
    provider = CyclicLidarPatternProvider(
        torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, -5.0]], dtype=torch.float32),
        height=1,
        width=2,
        publish_hz=10.0,
    )

    assert torch.equal(
        provider.directions_at(0.0),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        (torch.tensor([[0.0, 0.0, 0.0]]), "zero direction"),
        (torch.tensor([[float("nan"), 0.0, 1.0]]), "finite directions"),
        (torch.tensor([[float("inf"), 0.0, 1.0]]), "finite directions"),
    ],
)
def test_cyclic_pattern_rejects_invalid_sequence_directions(sequence: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CyclicLidarPatternProvider(sequence, height=1, width=1, publish_hz=10.0)


def test_cyclic_pattern_does_not_advance_before_the_next_measurement_period() -> None:
    provider = _cyclic_provider(publish_hz=10.0, height=1, width=2, sequence_offset=0)

    first = provider.directions_at(0.0)
    assert torch.equal(first, provider.directions_at(0.05))
    assert torch.equal(first, provider.directions_at(0.1 - 1e-7))
    assert not torch.equal(first, provider.directions_at(0.1))


def test_cyclic_pattern_advances_at_a_float32_measurement_boundary_undershoot() -> None:
    provider = _cyclic_provider(publish_hz=10.0, height=1, width=2, sequence_offset=0)
    first = provider.directions_at(0.0)
    float32_boundary_undershoot = float(np.nextafter(np.float32(0.1), np.float32(0.0)))

    assert not torch.equal(first, provider.directions_at(float32_boundary_undershoot))


def test_cyclic_pattern_nonintegral_rate_keeps_windows_contiguous() -> None:
    provider = _cyclic_provider(publish_hz=3.0, height=1, width=3, sequence_offset=0)
    offset_provider = _cyclic_provider(publish_hz=3.0, height=1, width=3, sequence_offset=3)

    assert torch.equal(provider.directions_at(1.0 / 3.0), offset_provider.directions_at(0.0))


def test_cyclic_pattern_restarts_sequence_after_simulation_time_reset() -> None:
    provider = _cyclic_provider(publish_hz=10.0, height=1, width=2, sequence_offset=0)
    baseline = _cyclic_provider(publish_hz=10.0, height=1, width=2, sequence_offset=0)
    provider.directions_at(5.0)
    provider.directions_at(5.1)

    assert torch.equal(provider.directions_at(1.0), baseline.directions_at(0.0))
    assert torch.equal(provider.directions_at(1.1), baseline.directions_at(0.1))


@pytest.mark.parametrize("publish_hz", [0.0, -1.0, math.inf, math.nan])
def test_cyclic_pattern_rejects_nonpositive_or_nonfinite_publish_rate(publish_hz: float) -> None:
    with pytest.raises(ValueError, match="publish_hz"):
        _cyclic_provider(publish_hz=publish_hz, height=1, width=1, sequence_offset=0)


@pytest.mark.parametrize(
    ("height", "width", "sequence_offset", "message"),
    [
        (0, 1, 0, "snapshot shape"),
        (1, 0, 0, "snapshot shape"),
        (1.0, 1, 0, "integer dimensions"),
        (1, 1, -1, "sequence_offset"),
        (1, 1, 0.0, "sequence_offset"),
    ],
)
def test_cyclic_pattern_rejects_invalid_snapshot_shape_or_offset(
    height: float,
    width: float,
    sequence_offset: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _cyclic_provider(
            publish_hz=10.0,
            height=height,
            width=width,
            sequence_offset=sequence_offset,
        )


def test_default_range_clipping_preserves_nan_point_no_return_semantics() -> None:
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    distances = torch.tensor([[2.0, float("inf")], [0.1, 5.0]])
    result = clip_sensor_ranges(distances, near=0.2, far=4.0, behavior="none")
    points = points_from_ranges(directions, result.ranges)
    assert torch.equal(result.hit_mask, torch.tensor([[True, False], [False, False]]))
    assert torch.equal(result.ranges[0, 0], torch.tensor(2.0))
    assert torch.isinf(result.ranges[0, 1])
    assert torch.isinf(result.ranges[1]).all()
    assert torch.equal(points[0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.isnan(points[0, 1]).all()
    assert torch.isnan(points[1, 0]).all()
    assert torch.isnan(points[1, 1]).all()


@pytest.mark.parametrize(
    ("behavior", "expected_ranges", "expected_points"),
    [
        (
            "max",
            [[2.0, 4.0], [4.0, 4.0]],
            [
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ],
        ),
        (
            "zero",
            [[2.0, 0.0], [0.0, 0.0]],
            [
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
        ),
    ],
)
def test_range_clipping_reports_explicit_endpoint_or_zero_sentinels(
    behavior: LidarRangeClippingBehavior,
    expected_ranges: list[list[float]],
    expected_points: list[list[list[float]]],
) -> None:
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    distances = torch.tensor([[2.0, float("inf")], [0.1, 5.0]])
    result = clip_sensor_ranges(distances, near=0.2, far=4.0, behavior=behavior)
    assert torch.equal(result.hit_mask, torch.tensor([[True, False], [False, False]]))
    assert torch.equal(result.ranges, torch.tensor(expected_ranges))
    assert torch.equal(points_from_ranges(directions, result.ranges), torch.tensor(expected_points))


@pytest.mark.parametrize(
    ("behavior", "expected_ranges", "expected_points"),
    [
        (
            "none",
            [[2.0, float("inf")], [float("inf"), float("inf")]],
            [
                [[2.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]],
                [[float("nan"), float("nan"), float("nan")], [float("nan"), float("nan"), float("nan")]],
            ],
        ),
        (
            "max",
            [[2.0, 4.0], [4.0, 4.0]],
            [
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ],
        ),
        (
            "zero",
            [[2.0, 0.0], [0.0, 0.0]],
            [
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
        ),
    ],
)
def test_range_finalizer_never_attributes_synthetic_points_to_geometry(
    behavior: LidarRangeClippingBehavior,
    expected_ranges: list[list[float]],
    expected_points: list[list[list[float]]],
) -> None:
    record = SimpleNamespace(
        config=LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            near=0.2,
            far=4.0,
            range_clipping_behavior=behavior,
        ),
        buffers={},
    )
    record.set_buffer = record.buffers.__setitem__
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    distances = torch.tensor([[2.0, float("inf")], [0.1, 5.0]])
    geom_ids = torch.tensor([[7, -1], [8, 9]], dtype=torch.int32)

    finalize_lidar_returns(record, directions, distances, geom_ids=geom_ids)

    assert torch.allclose(record.buffers["ranges"], torch.tensor(expected_ranges), equal_nan=True)
    assert torch.allclose(
        record.buffers["points"],
        torch.tensor(expected_points),
        equal_nan=True,
    )
    assert torch.equal(record.buffers["geom_ids"], torch.tensor([[7, -1], [-1, -1]], dtype=torch.int32))


@pytest.mark.parametrize(
    ("behavior", "expected_sentinel"),
    [
        ("none", float("inf")),
        ("max", 4.0),
        ("zero", 0.0),
    ],
)
def test_range_clipping_treats_nan_as_no_return_and_keeps_inclusive_bounds(
    behavior: LidarRangeClippingBehavior,
    expected_sentinel: float,
) -> None:
    distances = torch.tensor([[0.2, 4.0, float("nan")]])

    result = clip_sensor_ranges(distances, near=0.2, far=4.0, behavior=behavior)

    assert torch.equal(result.hit_mask, torch.tensor([[True, True, False]]))
    assert torch.allclose(
        result.ranges,
        torch.tensor([[0.2, 4.0, expected_sentinel]]),
        equal_nan=True,
    )
