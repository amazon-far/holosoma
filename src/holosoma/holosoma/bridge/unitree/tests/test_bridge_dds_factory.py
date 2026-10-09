from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import holosoma.bridge as bridge_factory
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import BridgeConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

if TYPE_CHECKING:
    from holosoma.bridge.base import BasicSdk2Bridge

pytestmark = pytest.mark.no_sim
_XML = '<CycloneDDS><Domain Id="0"/></CycloneDDS>'


def _create(sdk_type: str, config: BridgeConfig) -> BasicSdk2Bridge:
    return bridge_factory.create_sdk2py_bridge(
        cast("BaseSimulator", object()),
        cast("RobotConfig", SimpleNamespace(bridge=SimpleNamespace(sdk_type=sdk_type))),
        config,
    )


@pytest.mark.parametrize("sdk_type", ["unitree", "booster", "extension", "unknown"])
def test_explicit_config_rejects_nonisolated_backend_before_loading(
    monkeypatch: pytest.MonkeyPatch, sdk_type: str
) -> None:
    loader = MagicMock()
    monkeypatch.setattr(bridge_factory, "_bridge_registry", {sdk_type: loader})
    with pytest.raises(ValueError, match="requires the isolated unitree_mp backend"):
        _create(sdk_type, BridgeConfig(dds_config=_XML))
    loader.assert_not_called()


@pytest.mark.parametrize("domain_id", [1, 42])
def test_explicit_config_rejects_nonzero_sdk_domain_before_loading(
    monkeypatch: pytest.MonkeyPatch, domain_id: int
) -> None:
    loader = MagicMock()
    monkeypatch.setattr(bridge_factory, "_bridge_registry", {"unitree_mp": loader})
    with pytest.raises(ValueError, match="requires SDK domain_id=0"):
        _create("unitree_mp", BridgeConfig(dds_config=_XML, domain_id=domain_id))
    loader.assert_not_called()


@pytest.mark.parametrize("sdk_type", ["unitree", "unitree_mp", "booster", "extension"])
def test_no_config_preserves_backend_selection_and_domain(monkeypatch: pytest.MonkeyPatch, sdk_type: str) -> None:
    constructor = MagicMock()
    loader = MagicMock(return_value=constructor)
    monkeypatch.setattr(bridge_factory, "_bridge_registry", {sdk_type: loader})
    config = BridgeConfig(domain_id=42)
    assert _create(sdk_type, config) is constructor.return_value
    loader.assert_called_once_with()
    assert constructor.call_args.args[2] is config


def test_explicit_config_reaches_isolated_backend_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock()
    loader = MagicMock(return_value=constructor)
    monkeypatch.setattr(bridge_factory, "_bridge_registry", {"unitree_mp": loader})
    config = BridgeConfig(dds_config=_XML)
    assert _create("unitree_mp", config) is constructor.return_value
    loader.assert_called_once_with()
    assert constructor.call_args.args[2] is config


def test_missing_dds_config_is_an_integration_error_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MagicMock()
    monkeypatch.setattr(bridge_factory, "_bridge_registry", {"unitree_mp": loader})
    with pytest.raises(AttributeError, match="dds_config"):
        _create("unitree_mp", cast("BridgeConfig", SimpleNamespace(interface="lo")))
    loader.assert_not_called()
