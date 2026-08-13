"""Unit tests for frequency-string -> decimation resolution (pure; no simulator)."""

from __future__ import annotations

import pydantic
import pytest

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.plugin import CameraVizPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.config_types.simulator import PhysxConfig, SimEngineConfig

pytestmark = pytest.mark.no_sim

# A minimal valid mount required on every camera.
_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


# ----- resolve_decimation -----


def test_int_passthrough():
    assert resolve_decimation(4, 200, field="x") == 4
    assert resolve_decimation(1, 200, field="x") == 1


def test_int_below_one_rejected():
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_decimation(0, 200, field="x")


def test_bool_rejected():
    # bool is an int subclass; must not pass through as a decimation.
    with pytest.raises(ValueError, match="bool"):
        resolve_decimation(True, 200, field="x")


def test_frequency_string_exact_divisor():
    assert resolve_decimation("50Hz", 200, field="x") == 4


def test_bare_frequency_requires_exact_divisor():
    # 200/60 = 3.33 is not a whole decimation; a bare frequency must be exactly achievable.
    with pytest.raises(ValueError, match="not exactly achievable"):
        resolve_decimation("60Hz", 200, field="x")


def test_greater_than_floors_to_faster_rate():
    # 200/60 = 3.33 -> floor 3, so the achieved rate (66.7Hz) is >= the 60Hz target.
    assert resolve_decimation(">60Hz", 200, field="x") == 3


def test_less_than_ceils_to_slower_rate():
    # 200/60 = 3.33 -> ceil 4, so the achieved rate (50Hz) is <= the 60Hz target.
    assert resolve_decimation("<60Hz", 200, field="x") == 4


def test_comparison_prefixes_on_exact_divisor():
    # An exact divisor resolves identically regardless of prefix.
    assert resolve_decimation("50Hz", 200, field="x") == 4
    assert resolve_decimation(">50Hz", 200, field="x") == 4
    assert resolve_decimation("<50Hz", 200, field="x") == 4


def test_comparison_prefix_whitespace():
    assert resolve_decimation("  > 60 hz ", 200, field="x") == 3
    assert resolve_decimation(" <60Hz ", 200, field="x") == 4


def test_frequency_string_case_whitespace_and_float():
    assert resolve_decimation("  50 hz ", 200, field="x") == 4
    assert resolve_decimation("0.5Hz", 1, field="x") == 2


def test_frequency_faster_than_base_clamps_to_one():
    # floor(200/400) = 0 -> clamped to 1. A bare "400Hz" would instead error (not exact).
    assert resolve_decimation(">400Hz", 200, field="x") == 1
    assert resolve_decimation("<400Hz", 200, field="x") == 1
    with pytest.raises(ValueError, match="not exactly achievable"):
        resolve_decimation("400Hz", 200, field="x")


def test_malformed_strings_rejected():
    for bad in ("50", "Hz", "50khz", "-50Hz", "fastHz"):
        with pytest.raises(ValueError, match="invalid"):
            resolve_decimation(bad, 200, field="x")


def test_zero_frequency_rejected():
    with pytest.raises(ValueError, match="positive"):
        resolve_decimation("0Hz", 200, field="x")


# ----- SimEngineConfig integration -----


def _sim(control_decimation, render_interval=1, fps=200) -> SimEngineConfig:
    return SimEngineConfig(
        fps=fps,
        control_decimation=control_decimation,
        substeps=1,
        physx=PhysxConfig(solver_type=1, num_position_iterations=4, num_velocity_iterations=0),
        render_interval=render_interval,
    )


def test_sim_config_keeps_value_as_written_and_resolves_steps():
    cfg = _sim("50Hz", render_interval="100Hz")
    assert cfg.control_decimation == "50Hz"  # field holds what the user wrote
    assert cfg.control_decimation_steps == 4
    assert cfg.render_interval == "100Hz"
    assert cfg.render_interval_steps == 2


def test_sim_config_int_fields_resolve_identity():
    cfg = _sim(4, render_interval=2)
    assert cfg.control_decimation_steps == 4
    assert cfg.render_interval_steps == 2


def test_sim_config_rejects_bad_rate_at_construction():
    with pytest.raises(pydantic.ValidationError):
        _sim("banana")
    with pytest.raises(pydantic.ValidationError):
        _sim(0)


# ----- sensor configs keep the raw value (resolved later, at the simulator) -----


def test_camera_keeps_raw_string():
    cam = CameraSensorConfig(mount=_MOUNT, update_decimation="20Hz")
    assert cam.update_decimation == "20Hz"  # resolved by SensorManager at registration


def test_camera_rejects_malformed_at_construction():
    with pytest.raises(ValueError, match="update_decimation"):
        CameraSensorConfig(mount=_MOUNT, update_decimation="20hz!")


def test_viz_plugin_keeps_raw_string():
    rec = CameraVizPluginConfig(update_decimation="10Hz")
    assert rec.update_decimation == "10Hz"
