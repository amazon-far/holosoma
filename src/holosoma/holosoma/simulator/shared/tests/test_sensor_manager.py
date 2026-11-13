"""Unit tests for the SensorManager decimation lifecycle and modality gating (pure, no simulator).

Every backend's ``render_sensors`` drives ``collect_due`` (increment + gate), so the first frame
renders even at ``decimation > 1`` and ``frames_produced`` is an exact render count. Also pins that
an unknown modality is rejected at config construction, so a backend never silently diverges on an
unsupported data_type.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.frequency import DecimationLike
from holosoma.config_types.sensor import (
    CameraDataType,
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    DepthClippingBehavior,
    GridLidarRayPatternConfig,
    LidarRayPatternConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.simulator.shared.lidar_sensor import LidarPatternProvider
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


def _cam(
    *, update_decimation: DecimationLike = 1, data_types: list[CameraDataType] | None = None
) -> CameraSensorConfig:
    return CameraSensorConfig(mount=_MOUNT, update_decimation=update_decimation, data_types=data_types or ["rgb"])


def _drive(manager: SensorManager, steps: int) -> dict[str, list[bool]]:
    """Drive ``collect_due`` for ``steps`` steps; return per-camera due/not-due sequences."""
    seq: dict[str, list[bool]] = {name: [] for name in manager.names}
    for _ in range(steps):
        due_names = {rt.name for rt in manager.collect_due()}
        for name in manager.names:
            seq[name].append(name in due_names)
    return seq


def test_decimation_one_renders_every_step() -> None:
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=1))
    assert _drive(sm, 4)["c"] == [True, True, True, True]


def test_decimation_two_renders_on_first_step_then_every_other() -> None:
    # A dec>1 camera renders on its FIRST step (counter -1 -> 0), not skipping until step `dec`.
    # So the sequence is [True, False, True, False, ...].
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=2))
    assert _drive(sm, 5)["c"] == [True, False, True, False, True]


def test_frequency_string_resolves_and_gates() -> None:
    # "50Hz" against a 200Hz control rate floors to decimation 4: render on step 0, then every 4th.
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation="50Hz"))
    assert sm.get("c").effective_decimation == 4
    assert _drive(sm, 9)["c"] == [True, False, False, False, True, False, False, False, True]


def test_frames_produced_counts_renders() -> None:
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=2))
    assert sm.frames_produced("c") == 0  # nothing rendered yet
    sm.collect_due()  # step 0 -> renders frame 1
    assert sm.frames_produced("c") == 1
    sm.collect_due()  # step 1 -> not due
    assert sm.frames_produced("c") == 1
    sm.collect_due()  # step 2 -> renders frame 2
    assert sm.frames_produced("c") == 2


def test_frames_produced_dec3_holds_within_window() -> None:
    # The realistic slow-camera case: dec=3 over 7 steps. frames_produced must stay flat across the
    # two non-due steps of each window and increment exactly on the due step.
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=3))
    counts = []
    for _ in range(7):
        sm.collect_due()
        counts.append(sm.frames_produced("c"))
    assert counts == [1, 1, 1, 2, 2, 2, 3]


def test_mixed_decimation_cameras_independent() -> None:
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("fast", _cam(update_decimation=1))
    sm.register_camera("slow", _cam(update_decimation=3))
    seq = _drive(sm, 4)
    assert seq["fast"] == [True, True, True, True]
    assert seq["slow"] == [True, False, False, True]


@pytest.mark.parametrize("bad", ["thermal", "segmentation"])
def test_unimplemented_modality_rejected_at_config_construction(bad: CameraDataType) -> None:
    # rgb/depth are the only implemented modalities; anything else (a typo, or the not-yet-built
    # segmentation) fails loud at CameraSensorConfig construction (the CameraDataType literal),
    # before reaching a backend.
    with pytest.raises(ValueError, match="should be 'rgb' or 'depth'"):
        _cam(data_types=["rgb", bad])


def test_duplicate_name_rejected() -> None:
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam())
    with pytest.raises(ValueError, match="already registered"):
        sm.register_camera("c", _cam())


def test_depth_modality_supported() -> None:
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(data_types=["rgb", "depth"]))  # must not raise
    assert sm.names == ["c"]


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("none", [float("inf"), 0.2, 4.0, float("inf"), float("inf")]),
        ("max", [4.0, 0.2, 4.0, 4.0, 4.0]),
        ("zero", [0.0, 0.2, 4.0, 0.0, 0.0]),
    ],
)
def test_camera_depth_uses_the_shared_range_clipping_policy(
    behavior: DepthClippingBehavior,
    expected: list[float],
) -> None:
    manager = SensorManager(device="cpu", control_hz=200.0)
    record = manager.register_camera(
        "depth",
        CameraSensorConfig(
            mount=_MOUNT,
            data_types=["depth"],
            near=0.2,
            far=4.0,
            depth_clipping_behavior=behavior,
        ),
    )

    record.set_buffer(
        "depth",
        torch.tensor([[[[0.1], [0.2], [4.0], [5.0], [float("inf")], [float("nan")]]]], dtype=torch.float64),
    )

    expected_with_nonfinite = [*expected, expected[-1]]
    assert record.buffers["depth"].dtype == torch.float32
    assert torch.equal(record.buffers["depth"], torch.tensor([[[[value] for value in expected_with_nonfinite]]]))


def test_camera_and_lidar_cadence_are_independent() -> None:
    sm = SensorManager(device="cpu", control_hz=100.0)
    sm.register_camera("camera", _cam(update_decimation=2))
    sm.register_lidar("lidar", LidarSensorConfig(mount=_MOUNT, update_decimation=3))

    camera_due = []
    lidar_due = []
    for _ in range(7):
        camera_due.append(bool(sm.collect_due()))
        lidar_due.append(bool(sm.collect_lidars_due()))

    assert camera_due == [True, False, True, False, True, False, True]
    assert lidar_due == [True, False, False, True, False, False, True]
    assert sm.camera_names == ["camera"]
    assert sm.lidar_names == ["lidar"]


@pytest.mark.parametrize(
    ("pattern", "expected_directions"),
    [
        (
            GridLidarRayPatternConfig(horizontal_angles=[0.0, 90.0]),
            torch.tensor([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
        ),
        (
            CustomLidarRayPatternConfig(ray_directions=[[0.0, 2.0, 0.0], [0.0, 0.0, -3.0]]),
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
        ),
    ],
)
def test_sensor_manager_uses_selected_static_pattern_provider(
    pattern: LidarRayPatternConfig,
    expected_directions: torch.Tensor,
) -> None:
    manager = SensorManager(device="cpu", control_hz=100.0)
    record = manager.register_lidar("lidar", LidarSensorConfig(mount=_MOUNT, pattern=pattern))

    assert type(record.pattern_provider) is pattern.get_provider_cls()
    assert torch.allclose(record.pattern_at(0.0).directions, expected_directions, atol=1e-6)


def test_sensor_manager_delegates_provider_selection_to_pattern_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class SelectedProvider(LidarPatternProvider):
        def __init__(
            self,
            config: GridLidarRayPatternConfig,
            *,
            device: torch.device | str,
            publish_hz: float,
        ) -> None:
            assert config.horizontal_angles == [0.0]
            assert publish_hz == 50.0
            super().__init__(device=device, height=1, width=1)

        @property
        def time_varying(self) -> bool:
            return False

        def directions_at(self, sim_time: float) -> torch.Tensor:
            return torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)

    pattern = GridLidarRayPatternConfig(horizontal_angles=[0.0])
    monkeypatch.setattr(GridLidarRayPatternConfig, "get_provider_cls", lambda _: SelectedProvider)
    record = SensorManager(device="cpu", control_hz=100.0).register_lidar(
        "lidar",
        LidarSensorConfig(mount=_MOUNT, pattern=pattern, update_decimation=2),
    )

    assert type(record.pattern_provider) is SelectedProvider
    assert torch.equal(record.pattern_at(0.0).directions, torch.tensor([[0.0, 0.0, -1.0]]))


def test_sensor_names_must_be_unique_across_types() -> None:
    sm = SensorManager(device="cpu", control_hz=100.0)
    sm.register_camera("sensor", _cam())
    with pytest.raises(ValueError, match="already registered"):
        sm.register_lidar("sensor", LidarSensorConfig(mount=_MOUNT))
