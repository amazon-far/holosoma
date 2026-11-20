from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import experiment as experiment_values
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
            "run.headless": False,
            "robot.init_state.pos": [1.0, -2.0, 0.8],
            "plugin.clock": {"publish_every": "100Hz"},
        },
    },
}
EQUIVALENT_CLI = [
    "simulator:sim-mjwarp",
    "robot:t1-29dof-waist-wrist",
    "plugin.clock:clock_publish",
    "--run.headless",
    "False",
    "--robot.init_state.pos",
    "[1.0, -2.0, 0.8]",
    "--plugin.clock.publish_every",
    "100Hz",
]


def _parse_sim(args: list[str], **kwargs: Any) -> Any:
    return parse_config(ExperimentConfig, default=experiment_values.sim, args=args, **kwargs)


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
    assert _parse_sim(args) == _parse_sim(EQUIVALENT_CLI)


def test_command_line_overrides_document(tmp_path: Path) -> None:
    doc = {
        "version": 1,
        "arguments": {"robot": "g1-29dof", "robot.asset.link_physics": "none", "overrides": {"run.seed": 3}},
    }
    args = ["--config", _write(tmp_path, doc), "--run.seed", "-1", "robot:t1-29dof-waist-wrist"]
    expected = ["robot:t1-29dof-waist-wrist", "robot.asset.link_physics:none", "--run.seed", "-1"]
    assert _parse_sim(args) == _parse_sim(expected)


def test_scalars_follow_yaml_1_2(tmp_path: Path) -> None:
    text = (
        "version: 1\n"
        "arguments:\n"
        "  overrides:\n"
        "    run.seed: 010\n"
        "    name: off\n"
        "    project: 2026-11-17\n"
        "    robot.init_state.pos: [1e-05, 0.0, 0.8]\n"
    )
    config = _parse_sim(["--config", _write(tmp_path, text)])
    assert (config.run.seed, config.name, config.project) == (10, "off", "2026-11-17")
    assert config.robot.init_state.pos == [1e-05, 0.0, 0.8]


def test_root_subcommand_comes_first(tmp_path: Path) -> None:
    doc = {"version": 1, "arguments": {"simulator": "mujoco", "exp": "g1-29dof", "overrides": {"run.num_envs": 8}}}
    parsed = parse_config(get_annotated_experiment_config, args=["--config", _write(tmp_path, doc)])
    expected = parse_config(
        get_annotated_experiment_config, args=["exp:g1-29dof", "simulator:mujoco", "--run.num_envs", "8"]
    )
    assert parsed == expected


@pytest.mark.parametrize(
    ("content", "extra_args"),
    [
        ({**DOCUMENT, "version": 2}, []),
        ({**DOCUMENT, "backend": "mujoco"}, []),
        ("version: 1\narguments: {}\narguments: {}\n", []),
        (DOCUMENT, ["--config", "other.yaml"]),
        ({**DOCUMENT, "task": "sim"}, []),
    ],
    ids=["unsupported-version", "unknown-field", "duplicate-key", "repeated-config", "top-level-task"],
)
def test_rejects_invalid_documents(tmp_path: Path, content: dict[str, Any] | str, extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_sim(["--config", _write(tmp_path, content), *extra_args])


def test_json_document(tmp_path: Path) -> None:
    path = _write(tmp_path, json.dumps(DOCUMENT, indent="\t"), name="run.json")
    assert _parse_sim(["--config", path]) == _parse_sim(EQUIVALENT_CLI)


@pytest.mark.parametrize(
    ("name", "config_type", "default"),
    [
        ("g1_lidar", ExperimentConfig, experiment_values.sim),
        ("g1_locomotion_train", get_annotated_experiment_config, None),
    ],
)
def test_packaged_examples_parse(name: str, config_type: Any, default: Any) -> None:
    parse_config(config_type, default=default, args=["--config", f"@holosoma/examples/configs/{name}.yaml"])


@pytest.mark.parametrize(
    ("cli", "doc_task", "expected"),
    [
        ([], None, None),
        ([], "train", "train"),
        (["task:eval"], "train", "eval"),
        (["--run.name", "task:x", "task:replay"], None, "replay"),
    ],
    ids=["none", "document", "cli-wins", "flag-value-is-not-a-task"],
)
def test_select_task(tmp_path: Path, cli: list[str], doc_task: str | None, expected: str | None) -> None:
    doc = {**DOCUMENT, "arguments": {**DOCUMENT["arguments"], "task": doc_task}} if doc_task else DOCUMENT
    args = ["--config", _write(tmp_path, doc), *cli]
    task, remaining = select_task(args)
    assert task == expected
    assert remaining == [t for t in args if t != f"task:{expected}"]


def test_cli_runs_the_selected_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from holosoma import cli
    from holosoma.tasks.replay import ReplayTask

    seen: list[Any] = []

    class _Parsed:
        def run(self) -> None:
            seen.append("run")

    def from_args(_cls: type, args: list[str]) -> _Parsed:
        seen.append(args)
        return _Parsed()

    monkeypatch.setattr(ReplayTask, "from_args", classmethod(from_args))
    config = _write(tmp_path, {**DOCUMENT, "arguments": {**DOCUMENT["arguments"], "task": "replay"}})
    monkeypatch.setattr(sys, "argv", ["holosoma", "--config", config, "--run.seed", "3"])
    cli.main()
    assert seen == [["--config", config, "--run.seed", "3"], "run"]

    monkeypatch.setattr(sys, "argv", ["holosoma", "task:fly"])
    with pytest.raises(SystemExit, match="Unknown task 'fly'"):
        cli.main()


@pytest.mark.parametrize("alias", ["train_agent", "eval_agent", "replay"])
def test_task_aliases_run_the_holosoma_command_with_their_task(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    import importlib

    from holosoma import cli

    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda: seen.append(list(sys.argv[1:])))
    monkeypatch.setattr(sys, "argv", [alias, "--run.seed", "3"])
    importlib.import_module(f"holosoma.{alias}").main()
    task = {"train_agent": "train", "eval_agent": "eval", "replay": "replay"}[alias]
    assert seen == [[f"task:{task}", "--run.seed", "3"]]


def test_run_sim_is_the_holosoma_command() -> None:
    from holosoma import cli, run_sim

    assert run_sim.main is cli.main


def test_document_selects_the_task_and_sets_its_fields(tmp_path: Path) -> None:
    overrides = {"task.checkpoint": "/runs/a.pt", "task.callbacks.push.config.enabled": True}
    doc = {"version": 1, "arguments": {"task": "eval", "overrides": overrides}}
    from_document = parse_config(ExperimentConfig, args=["--config", _write(tmp_path, doc)])
    from_flags = parse_config(
        ExperimentConfig,
        args=["task:eval", "--task.checkpoint=/runs/a.pt", "--task.callbacks.push.config.enabled=True"],
    )
    assert from_document == from_flags
    assert from_document.task.checkpoint == "/runs/a.pt"


def test_rejected_document_tokens_name_their_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, {"version": 1, "arguments": {"overrides": {"robot.init_state.posx": [1.0, 2.0]}}})
    with pytest.raises(SystemExit):
        _parse_sim(["--config", path])
    assert f"<- {path}: arguments.overrides.robot.init_state.posx" in capsys.readouterr().err


def test_print_args_prints_the_equivalent_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse_sim(["--config", _write(tmp_path, DOCUMENT), "--print-args"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("holosoma ")
    import shlex

    assert _parse_sim(shlex.split(printed)[1:]) == _parse_sim(EQUIVALENT_CLI)


def test_moved_flags_keep_working(tmp_path: Path) -> None:
    old = [
        "exp:g1-29dof",
        "--training.num_envs=8",
        "--training.checkpoint=/runs/a.pt",
        "--eval_overrides.num_envs",
        "3",
    ]
    doc = {"version": 1, "arguments": {"overrides": {"training.project": "p", "time_scale": 0.5}}}
    new = [
        "exp:g1-29dof",
        "--run.num_envs=8",
        "--task.resume_from=/runs/a.pt",
        "--project=p",
        "--run.time_scale=0.5",
    ]
    parsed_old = parse_config(get_annotated_experiment_config, args=[*old, "--config", _write(tmp_path, doc)])
    assert parsed_old == parse_config(get_annotated_experiment_config, args=new)


@pytest.mark.parametrize(
    ("config_type", "default", "args"),
    [
        (
            ExperimentConfig,
            experiment_values.sim,
            [
                "simulator:sim-mjwarp",
                "robot:t1-29dof-waist-wrist",
                "sensor.scan:g1-pelvis-lidar",
                "plugin.clock:clock_publish",
                "--run.headless=False",
                "--run.time_scale",
                "0.50",
                "--robot.init_state.pos=[1.0, -2.0, 0.8]",
                "--plugin.clock.publish_every=100Hz",
            ],
        ),
        (
            get_annotated_experiment_config,
            None,
            ["exp:g1-29dof", "task:train", "simulator:mjwarp", "--run.num_envs=8", "--task.resume_from=/runs/a.pt"],
        ),
    ],
    ids=["sim", "train"],
)
def test_written_document_reproduces_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], config_type: Any, default: Any, args: list[str]
) -> None:
    from holosoma.utils.config_document import args_to_document, dump_document, write_document

    record: list[str] = []
    parsed = parse_config(config_type, default=default, args=args, record=record)
    path = tmp_path / "holosoma_run.yaml"
    write_document(path, record)
    assert parse_config(config_type, default=default, args=["--config", str(path)]) == parsed

    with pytest.raises(SystemExit):
        parse_config(config_type, default=default, args=[*args, "--to-config"])
    assert capsys.readouterr().out == dump_document(args_to_document(record))


@pytest.mark.parametrize(("arguments", "num_envs"), [({}, 1), ({"run": "batch"}, 4096)], ids=["default", "selected"])
def test_default_selection_applies_only_when_the_group_is_unselected(
    tmp_path: Path, arguments: dict[str, Any], num_envs: int
) -> None:
    doc = {"version": 1, "arguments": {"exp": "g1-29dof", **arguments}}
    config = parse_config(
        get_annotated_experiment_config,
        args=["--config", _write(tmp_path, doc)],
        default_selections=("run:interactive",),
    )
    assert config.run.num_envs == num_envs
