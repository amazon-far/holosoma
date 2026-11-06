"""Isaac presets must be selectable from both public configuration entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("config_type", ["ExperimentConfig", "RunSimConfig"])
@pytest.mark.parametrize("environment_backend", ["isaaclab2", "isaaclab3"])
def test_explicit_lab3_preset_parses_in_a_fresh_process(config_type: str, environment_backend: str) -> None:
    module = "experiment" if config_type == "ExperimentConfig" else "run_sim"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import dataclasses
import json
from holosoma.config_types.{module} import {config_type}
from holosoma.utils.config_registry import parse_config

legacy = parse_config({config_type}, args=["simulator:isaacsim"]).simulator
native = parse_config({config_type}, args=["simulator:isaacsim3"]).simulator
override = parse_config(
    {config_type}, args=["simulator:isaacsim", "--simulator.config.isaacsim-backend=ISAACLAB3"]
).simulator
print(json.dumps({{
    "name": native.config.name,
    "backend": native.config.isaacsim_backend,
    "override_backend": override.config.isaacsim_backend,
    "legacy_backend": legacy.config.isaacsim_backend,
    "same_defaults": dataclasses.replace(
        native.config, isaacsim_backend=legacy.config.isaacsim_backend
    ) == legacy.config,
    "same_target": native._target_ == legacy._target_,
    "bridge": native.config.bridge.enabled,
    "gantry": native.config.virtual_gantry.enabled,
    "fps": native.config.sim.fps,
}}))
""",
        ],
        env={**os.environ, "HOLOSOMA_ISAACSIM_BACKEND": environment_backend},
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    actual = json.loads(child.stdout.strip())
    assert actual == {
        "name": "isaacsim",
        "backend": "isaaclab3",
        "override_backend": "isaaclab3",
        "legacy_backend": environment_backend,
        "same_defaults": True,
        "same_target": True,
        "bridge": config_type == "RunSimConfig",
        "gantry": config_type == "RunSimConfig",
        "fps": 200,
    }
