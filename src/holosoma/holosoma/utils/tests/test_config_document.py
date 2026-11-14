from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.config_document import ConfigDocumentError, expand_config_args
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim

DOCUMENT: dict[str, Any] = {
    "version": 2,
    "backend": "mujoco",
    "arguments": {
        "robot": "g1-29dof",
        "sensor": {"scan": "g1-pelvis-lidar"},
        "plugin": {"clock": "clock_publish"},
        "overrides": {
            "training.headless": True,
            "robot.init_state.pos": [1.0, -2.0, 0.8],
            "plugin.clock": {"publish_every": "100Hz"},
        },
    },
}
EQUIVALENT_CLI = [
    "simulator:mujoco",
    "robot:g1-29dof",
    "sensor.scan:g1-pelvis-lidar",
    "plugin.clock:clock_publish",
    "--training.headless",
    "True",
    "--robot.init_state.pos",
    "[1.0, -2.0, 0.8]",
    "--plugin.clock.publish_every",
    "100Hz",
]


def _write(tmp_path: Path, content: dict[str, Any]) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(content))
    return str(path)


@pytest.mark.parametrize(
    ("content", "config_path"),
    [
        (DOCUMENT, None),
        ({"task": "demo", "holosoma_sim": DOCUMENT}, None),
        ({"version": 1, "nested": {"run": DOCUMENT}}, "nested.run"),
    ],
    ids=["root", "embedded", "config-path"],
)
def test_document_parses_like_equivalent_cli(tmp_path: Path, content: dict[str, Any], config_path: str | None) -> None:
    argv = ["--config", _write(tmp_path, content)]
    if config_path:
        argv += ["--config-path", config_path]
    assert parse_config(RunSimConfig, args=expand_config_args(argv)) == parse_config(RunSimConfig, args=EQUIVALENT_CLI)


def test_command_line_overrides_document(tmp_path: Path) -> None:
    argv = ["--config", _write(tmp_path, DOCUMENT), "robot:t1-29dof-waist-wrist", "--training.headless", "False"]
    config = parse_config(RunSimConfig, args=expand_config_args(argv))
    cli_only = parse_config(RunSimConfig, args=["robot:t1-29dof-waist-wrist"])
    assert config.robot.asset.robot_type == cli_only.robot.asset.robot_type
    assert config.robot.init_state.pos == [1.0, -2.0, 0.8]
    assert config.training.headless is False


@pytest.mark.parametrize(
    "content",
    [
        {**DOCUMENT, "version": 1},
        {**DOCUMENT, "extensions": ["example"]},
        {**DOCUMENT, "arguments": {**DOCUMENT["arguments"], "simulator": "isaacsim"}},
    ],
    ids=["old-version", "launcher-field", "backend-and-simulator"],
)
def test_rejects_invalid_documents(tmp_path: Path, content: dict[str, Any]) -> None:
    with pytest.raises(ConfigDocumentError):
        expand_config_args(["--config", _write(tmp_path, content)])
