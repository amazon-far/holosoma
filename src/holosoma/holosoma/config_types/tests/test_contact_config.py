"""Construction and serialization contracts for contact sensors."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import TypeAdapter, ValidationError

from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig, IsaacSimContactConfig, SensorConfig

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sources": []},
        {"sources": [ContactTargetConfig(target_kind="terrain")]},
        {"sources": [ContactTargetConfig(), ContactTargetConfig()]},
        {"counterparts": [ContactTargetConfig(), ContactTargetConfig()]},
        {"force_mode": "total"},
        {"force_mode": "invalid"},
        {"force_threshold": -1.0},
        {"force_threshold": float("nan")},
        {"force_threshold": float("inf")},
        {"update_decimation": 0},
        {"unexpected": True},
    ],
)
def test_invalid_contact_configuration_fails(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ContactSensorConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_kind": "actor"},
        {"target_kind": "actor", "target": "robot"},
        {"target_kind": "robot_link", "target": " pelvis"},
        {"target_kind": "terrain", "target": "floor"},
        {"target_kind": "robot", "target": "pelvis"},
        {"target_kind": "world"},
    ],
)
def test_invalid_contact_target_fails(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ContactTargetConfig(**kwargs)  # type: ignore[arg-type]


def test_heterogeneous_sensor_union_round_trips_contact_configuration() -> None:
    config = ContactSensorConfig(
        sources=[ContactTargetConfig(target_kind="robot_link", target="foot")],
        counterparts=[
            ContactTargetConfig(target_kind="actor", target="box"),
            ContactTargetConfig(target_kind="terrain"),
        ],
        force_mode="total",
        update_decimation="10Hz",
    )
    adapter: TypeAdapter[SensorConfig] = TypeAdapter(SensorConfig)
    assert adapter.validate_json(adapter.dump_json(config)) == config
    assert dataclasses.asdict(config)["sources"] == [{"target_kind": "robot_link", "target": "foot"}]


def test_contact_capacity_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        IsaacSimContactConfig(max_contact_data_count_per_prim=0)


def test_contact_defaults_update_each_physics_step_without_history() -> None:
    assert ContactSensorConfig().update_decimation == 1
    assert not ContactSensorConfig().record_substeps
    with pytest.raises(ValidationError):
        ContactSensorConfig(update_phase="frame_end")  # type: ignore[call-arg]
