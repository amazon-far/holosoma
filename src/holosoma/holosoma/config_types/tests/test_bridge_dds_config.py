from __future__ import annotations

from dataclasses import asdict, field, replace
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim

_XML = ' \n<CycloneDDS><Domain Id="0"/></CycloneDDS>\t'


@dataclass(frozen=True)
class _SimulatorInit:
    bridge: BridgeConfig = field(default_factory=BridgeConfig)


@dataclass(frozen=True)
class _Simulator:
    config: _SimulatorInit = field(default_factory=_SimulatorInit)


@dataclass(frozen=True)
class _CLI:
    simulator: _Simulator = field(default_factory=_Simulator)


def test_bridge_dds_config_defaults_roundtrip_replace_and_schema() -> None:
    defaults = BridgeConfig()
    assert defaults.dds_config is None
    assert defaults.domain_id == 0
    assert defaults.interface is None
    assert not defaults.enabled
    assert defaults.publish_odom
    config = BridgeConfig(dds_config=_XML)
    assert BridgeConfig(**asdict(config)).dds_config == _XML
    assert replace(config, interface="lo").dds_config == _XML
    adapter = TypeAdapter(BridgeConfig)
    assert adapter.validate_json(adapter.dump_json(config)).dds_config == _XML
    assert adapter.json_schema()["properties"]["dds_config"]["default"] is None


@pytest.mark.parametrize("value", ["", " \n\t\r", b"", b" \n\t\r"])
def test_bridge_dds_config_rejects_blank_after_type_normalization(value: Any) -> None:
    with pytest.raises(ValidationError, match="dds_config must be a non-empty inline XML string or None"):
        BridgeConfig(dds_config=value)


@pytest.mark.parametrize("value", [0, False, [], {}])
def test_bridge_dds_config_uses_pydantic_string_validation(value: Any) -> None:
    with pytest.raises(ValidationError) as error:
        BridgeConfig(dds_config=value)
    assert error.value.errors()[0]["type"] == "string_type"


def test_bridge_dds_config_accepts_pydantic_decoded_bytes() -> None:
    assert BridgeConfig(dds_config=cast("Any", _XML.encode())).dds_config == _XML


def test_bridge_dds_config_leaves_xml_semantics_to_native() -> None:
    # Consumer validation is intentionally not a second XML parser (nor a file/URI loader).
    assert BridgeConfig(dds_config="not XML").dds_config == "not XML"


def test_bridge_dds_config_cli_spelling_and_byte_preservation() -> None:
    config = parse_config(_CLI, args=["--simulator.config.bridge.dds-config", _XML])
    assert config.simulator.config.bridge.dds_config == _XML
    assert config.simulator.config.bridge.domain_id == 0
    assert parse_config(_CLI, args=[]).simulator.config.bridge.dds_config is None
