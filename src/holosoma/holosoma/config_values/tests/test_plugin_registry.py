"""CLI/registry resolution for the ``RunSimConfig.plugin`` dynamic-dict field."""

from __future__ import annotations

import dataclasses

import pytest
from typing_extensions import Annotated, TypeAlias

from holosoma.config_types.plugin import (
    ClockPublishPluginConfig,
    GantryControlPluginConfig,
    NoOpPluginConfig,
    PluginConfig,
)
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_values.plugin import PLUGIN_REGISTRY
from holosoma.utils import config_registry as registry
from holosoma.utils.config_registry import UseRegistry, parse_config, registry_from_annotated
from holosoma.utils.tyro_utils import find_dynamic_dict_fields

pytestmark = pytest.mark.no_sim

# Ensure every real registry is discovered so parse_config sees the full menu.
registry.load_plugins()


def test_plugin_registry_holds_builtin_preset() -> None:
    assert isinstance(PLUGIN_REGISTRY["clock_publish"], ClockPublishPluginConfig)
    assert PLUGIN_REGISTRY.config_type is PluginConfig


def test_run_sim_plugin_field_is_scanned_and_bound() -> None:
    found = find_dynamic_dict_fields(RunSimConfig)
    assert "plugin" in found
    assert registry_from_annotated(found["plugin"]) is PLUGIN_REGISTRY


def test_run_sim_defaults_to_empty_plugin() -> None:
    cfg = parse_config(RunSimConfig, args=["simulator:mujoco"])
    assert cfg.plugin == {}


def test_declare_builtin_plugin() -> None:
    cfg = parse_config(RunSimConfig, args=["simulator:mujoco", "plugin.clk:clock_publish"])
    assert set(cfg.plugin) == {"clk"}
    assert isinstance(cfg.plugin["clk"], ClockPublishPluginConfig)


def test_none_preset_resolves_to_noop_config() -> None:
    # `plugin.<key>:none` selects the no-op preset (mirrors every other family's `none`),
    # resolving to a real NoOpPluginConfig rather than a literal None.
    cfg = parse_config(RunSimConfig, args=["simulator:mujoco", "plugin.disabled:none"])
    assert isinstance(cfg.plugin["disabled"], NoOpPluginConfig)


def test_ros2_presets_resolve_with_independent_gantry_topics() -> None:
    # The two ROS2 example plugins resolve from the registry (their configs are rclpy-free);
    # per-topic gantry overrides target one property's topic without touching the others.
    cfg = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "plugin.clk:clock_publish",
            "--plugin.clk.topic=/sim_clock",
            "plugin.g:gantry_control",
            "--plugin.g.length-topic=/g/len",
        ],
    )
    assert isinstance(cfg.plugin["clk"], ClockPublishPluginConfig)
    assert cfg.plugin["clk"].topic == "/sim_clock"
    gantry = cfg.plugin["g"]
    assert isinstance(gantry, GantryControlPluginConfig)
    assert gantry.length_topic == "/g/len"
    # Untouched topics keep their defaults (independent control).
    assert gantry.position_topic == "/gantry/position"
    assert gantry.enabled_topic == "/gantry/enabled"


def test_declare_plugin_with_leaf_override() -> None:
    cfg = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "plugin.clk:clock_publish",
            "--plugin.clk.publish-every=25",
            "--plugin.clk.node-name=my_clock",
        ],
    )
    plugin = cfg.plugin["clk"]
    assert plugin.publish_every == 25
    assert plugin.node_name == "my_clock"


def test_frequency_string_rate_parses_and_validates() -> None:
    # A frequency string is accepted on a rate field and held as-written (resolved to a
    # decimation at install time against the phase base rate).
    cfg = parse_config(
        RunSimConfig,
        args=["simulator:mujoco", "plugin.clk:clock_publish", "--plugin.clk.publish-every=20Hz"],
    )
    assert cfg.plugin["clk"].publish_every == "20Hz"


def test_bad_frequency_string_fails_at_config_time() -> None:
    with pytest.raises(ValueError, match="publish_every"):
        ClockPublishPluginConfig(publish_every="not-a-rate")


def test_unknown_plugin_variant_fails_loud() -> None:
    with pytest.raises(SystemExit, match="Unknown plugin variant 'bogus'"):
        parse_config(RunSimConfig, args=["simulator:mujoco", "plugin.x:bogus"])


# A second PluginConfig subclass, resolved from the same registry alongside the built-in,
# proves the heterogeneous-subclass menu the spec's plugins rely on.
@dataclasses.dataclass(frozen=True)
class _PublishPluginConfig(PluginConfig):
    topic: str = "state"

    def get_cls(self):  # pragma: no cover - not instantiated in this test
        raise NotImplementedError


_PluginField: TypeAlias = Annotated[PluginConfig, UseRegistry(PLUGIN_REGISTRY)]


@dataclasses.dataclass(frozen=True)
class _MultiPluginConfig:
    plugin: dict[str, _PluginField] = dataclasses.field(default_factory=dict)


def test_multiple_heterogeneous_plugins_resolve() -> None:
    PLUGIN_REGISTRY.add("publish", _PublishPluginConfig())
    try:
        cfg = parse_config(
            _MultiPluginConfig,
            args=[
                "plugin.clk:clock_publish",
                "plugin.your_biz:publish",
                "--plugin.clk.node-name=base_clock",
                "--plugin.your-biz.topic=poses",
            ],
        )
        assert isinstance(cfg.plugin["clk"], ClockPublishPluginConfig)
        assert cfg.plugin["clk"].node_name == "base_clock"
        assert isinstance(cfg.plugin["your_biz"], _PublishPluginConfig)
        assert cfg.plugin["your_biz"].topic == "poses"
    finally:
        PLUGIN_REGISTRY.pop("publish", None)
