"""CLI/registry resolution for the ``RunSimConfig.plugin`` dynamic-dict field."""

from __future__ import annotations

import pytest

from holosoma.config_types.plugin import (
    ClockPublishPluginConfig,
    NoOpPluginConfig,
)
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils import config_registry as registry
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim

# Ensure every real registry is discovered so parse_config sees the full menu.
registry.load_plugins()


def test_none_preset_resolves_to_noop_config() -> None:
    # `plugin.<key>:none` selects the no-op preset (mirrors every other family's `none`),
    # resolving to a real NoOpPluginConfig rather than a literal None.
    cfg = parse_config(RunSimConfig, args=["simulator:mujoco", "plugin.disabled:none"])
    assert isinstance(cfg.plugin["disabled"], NoOpPluginConfig)


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
