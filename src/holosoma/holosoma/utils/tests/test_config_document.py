from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_values.experiment import get_annotated_experiment_config
from holosoma.utils.config_document import select_task
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim

DOCUMENT: dict[str, Any] = {
    "version": 1,
    "arguments": {
        "simulator": "sim-mjwarp",
        "robot": "t1-29dof-waist-wrist",
        "plugin": {"clock": "clock_publish"},
        "overrides": {
            "training.headless": False,
            "robot.init_state.pos": [1.0, -2.0, 0.8],
            "plugin.clock": {"publish_every": "100Hz"},
        },
    },
}
EQUIVALENT_CLI = [
    "simulator:sim-mjwarp",
    "robot:t1-29dof-waist-wrist",
    "plugin.clock:clock_publish",
    "--training.headless",
    "False",
    "--robot.init_state.pos",
    "[1.0, -2.0, 0.8]",
    "--plugin.clock.publish_every",
    "100Hz",
]


def _write(tmp_path: Path, content: dict[str, Any] | str, name: str = "run.yaml") -> str:
    path = tmp_path / name
    path.write_text(content if isinstance(content, str) else yaml.safe_dump(content))
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
    args = ["--config", _write(tmp_path, content)]
    if config_path:
        args += ["--config-path", config_path]
    assert parse_config(RunSimConfig, args=args) == parse_config(RunSimConfig, args=EQUIVALENT_CLI)


def test_command_line_overrides_document(tmp_path: Path) -> None:
    doc = {
        "version": 1,
        "arguments": {"robot": "g1-29dof", "robot.asset.link_physics": "none", "overrides": {"training.seed": 3}},
    }
    args = ["--config", _write(tmp_path, doc), "--training.seed", "-1", "robot:t1-29dof-waist-wrist"]
    expected = ["robot:t1-29dof-waist-wrist", "robot.asset.link_physics:none", "--training.seed", "-1"]
    assert parse_config(RunSimConfig, args=args) == parse_config(RunSimConfig, args=expected)


def test_scalars_follow_yaml_1_2(tmp_path: Path) -> None:
    text = (
        "version: 1\n"
        "arguments:\n"
        "  overrides:\n"
        "    training.seed: 010\n"
        "    training.name: off\n"
        "    training.project: 2026-11-17\n"
        "    robot.init_state.pos: [1e-05, 0.0, 0.8]\n"
    )
    config = parse_config(RunSimConfig, args=["--config", _write(tmp_path, text)])
    assert (config.training.seed, config.training.name, config.training.project) == (10, "off", "2026-11-17")
    assert config.robot.init_state.pos == [1e-05, 0.0, 0.8]


def test_root_subcommand_comes_first(tmp_path: Path) -> None:
    doc = {"version": 1, "arguments": {"simulator": "mujoco", "exp": "g1-29dof", "overrides": {"training.num_envs": 8}}}
    parsed = parse_config(get_annotated_experiment_config, args=["--config", _write(tmp_path, doc)])
    expected = parse_config(
        get_annotated_experiment_config, args=["exp:g1-29dof", "simulator:mujoco", "--training.num_envs", "8"]
    )
    assert parsed == expected


@pytest.mark.parametrize(
    ("content", "extra_args"),
    [
        ({**DOCUMENT, "version": 2}, []),
        ({**DOCUMENT, "backend": "mujoco"}, []),
        ("version: 1\narguments: {}\narguments: {}\n", []),
        (DOCUMENT, ["--config", "other.yaml"]),
        ({**DOCUMENT, "task": 5}, []),
    ],
    ids=["unsupported-version", "unknown-field", "duplicate-key", "repeated-config", "task-not-a-name"],
)
def test_rejects_invalid_documents(tmp_path: Path, content: dict[str, Any] | str, extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_config(RunSimConfig, args=["--config", _write(tmp_path, content), *extra_args])


def test_json_document(tmp_path: Path) -> None:
    path = _write(tmp_path, json.dumps(DOCUMENT, indent="\t"), name="run.json")
    assert parse_config(RunSimConfig, args=["--config", path]) == parse_config(RunSimConfig, args=EQUIVALENT_CLI)


@pytest.mark.parametrize(
    ("name", "config_type"),
    [("g1_lidar", RunSimConfig), ("g1_locomotion_train", get_annotated_experiment_config)],
)
def test_packaged_examples_parse(name: str, config_type: Any) -> None:
    parse_config(config_type, args=["--config", f"@holosoma/examples/configs/{name}.yaml"])


@pytest.mark.parametrize(
    ("cli", "doc_task", "expected"),
    [
        ([], None, None),
        ([], "train", "train"),
        (["task:eval"], "train", "eval"),
        (["--training.name", "task:x", "task:replay"], None, "replay"),
    ],
    ids=["none", "document", "cli-wins", "flag-value-is-not-a-task"],
)
def test_select_task(tmp_path: Path, cli: list[str], doc_task: str | None, expected: str | None) -> None:
    doc = {**DOCUMENT, "task": doc_task} if doc_task else DOCUMENT
    args = ["--config", _write(tmp_path, doc), *cli]
    task, remaining = select_task(args)
    assert task == expected
    assert remaining == [t for t in args if t != f"task:{expected}"]


def test_run_sim_runs_the_selected_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from holosoma import run_sim
    from holosoma.tasks.replay import ReplayTask

    seen: list[Any] = []

    class _Parsed:
        def run(self) -> None:
            seen.append("run")

    def from_args(_cls: type, args: list[str]) -> _Parsed:
        seen.append(args)
        return _Parsed()

    monkeypatch.setattr(ReplayTask, "from_args", classmethod(from_args))
    config = _write(tmp_path, {**DOCUMENT, "task": "replay"})
    monkeypatch.setattr(sys, "argv", ["run_sim", "--config", config, "--training.seed", "3"])
    run_sim.main()
    assert seen == [["--config", config, "--training.seed", "3"], "run"]

    monkeypatch.setattr(sys, "argv", ["run_sim", "task:fly"])
    with pytest.raises(SystemExit, match="Unknown task 'fly'"):
        run_sim.main()
