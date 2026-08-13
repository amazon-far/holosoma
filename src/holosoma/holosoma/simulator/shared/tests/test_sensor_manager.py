"""Unit tests for the SensorManager decimation lifecycle and modality gating (pure, no simulator).

Every backend's ``render_sensors`` drives ``collect_due`` (increment + gate), so the first frame
renders even at ``decimation > 1`` and ``frames_produced`` is an exact render count. Also pins that
an unknown modality is rejected at config construction, so a backend never silently diverges on an
unsupported data_type.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.shared.camera_sensor import SensorManager

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


def _cam(*, update_decimation=1, data_types=None) -> CameraSensorConfig:
    return CameraSensorConfig(mount=_MOUNT, update_decimation=update_decimation, data_types=data_types or ["rgb"])


def _drive(manager: SensorManager, steps: int) -> dict[str, list[bool]]:
    """Drive ``collect_due`` for ``steps`` steps; return per-camera due/not-due sequences."""
    seq: dict[str, list[bool]] = {name: [] for name in manager.names}
    for _ in range(steps):
        due_names = {rt.name for rt in manager.collect_due()}
        for name in manager.names:
            seq[name].append(name in due_names)
    return seq


def test_decimation_one_renders_every_step():
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=1))
    assert _drive(sm, 4)["c"] == [True, True, True, True]


def test_decimation_two_renders_on_first_step_then_every_other():
    # A dec>1 camera renders on its FIRST step (counter -1 -> 0), not skipping until step `dec`.
    # So the sequence is [True, False, True, False, ...].
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=2))
    assert _drive(sm, 5)["c"] == [True, False, True, False, True]


def test_frequency_string_resolves_and_gates():
    # "50Hz" against a 200Hz control rate floors to decimation 4: render on step 0, then every 4th.
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation="50Hz"))
    assert sm.get("c").effective_decimation == 4
    assert _drive(sm, 9)["c"] == [True, False, False, False, True, False, False, False, True]


def test_frames_produced_counts_renders():
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=2))
    assert sm.frames_produced("c") == 0  # nothing rendered yet
    sm.collect_due()  # step 0 -> renders frame 1
    assert sm.frames_produced("c") == 1
    sm.collect_due()  # step 1 -> not due
    assert sm.frames_produced("c") == 1
    sm.collect_due()  # step 2 -> renders frame 2
    assert sm.frames_produced("c") == 2


def test_frames_produced_dec3_holds_within_window():
    # The realistic slow-camera case: dec=3 over 7 steps. frames_produced must stay flat across the
    # two non-due steps of each window and increment exactly on the due step.
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(update_decimation=3))
    counts = []
    for _ in range(7):
        sm.collect_due()
        counts.append(sm.frames_produced("c"))
    assert counts == [1, 1, 1, 2, 2, 2, 3]


def test_mixed_decimation_cameras_independent():
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("fast", _cam(update_decimation=1))
    sm.register_camera("slow", _cam(update_decimation=3))
    seq = _drive(sm, 4)
    assert seq["fast"] == [True, True, True, True]
    assert seq["slow"] == [True, False, False, True]


@pytest.mark.parametrize("bad", ["thermal", "segmentation"])
def test_unimplemented_modality_rejected_at_config_construction(bad):
    # rgb/depth are the only implemented modalities; anything else (a typo, or the not-yet-built
    # segmentation) fails loud at CameraSensorConfig construction (the CameraDataType literal),
    # before reaching a backend.
    with pytest.raises(ValueError, match="should be 'rgb' or 'depth'"):
        _cam(data_types=["rgb", bad])


def test_duplicate_name_rejected():
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam())
    with pytest.raises(ValueError, match="already registered"):
        sm.register_camera("c", _cam())


def test_depth_modality_supported():
    sm = SensorManager(device="cpu", control_hz=200.0)
    sm.register_camera("c", _cam(data_types=["rgb", "depth"]))  # must not raise
    assert sm.names == ["c"]
