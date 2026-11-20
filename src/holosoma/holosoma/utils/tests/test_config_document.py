from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from holosoma import cli
from holosoma.config_types.experiment import SimulationExperimentConfig, TrainingExperimentConfig
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager

pytestmark = pytest.mark.no_sim

DOCUMENT: dict[str, Any] = {
    "version": 1,
    "arguments": {
        "simulator": "mjwarp-sim2sim",
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
    "simulator:mjwarp-sim2sim",
    "robot:t1-29dof-waist-wrist",
    "plugin.clock:clock_publish",
    "--run.headless",
    "False",
    "--robot.init_state.pos",
    "[1.0, -2.0, 0.8]",
    "--plugin.clock.publish_every",
    "100Hz",
]


class _Recorder:
    """A task class whose entry function records the config it is given instead of running it."""

    seen: ClassVar[list[SimulationExperimentConfig]] = []

    @classmethod
    def simulate(cls, config: SimulationExperimentConfig) -> None:
        cls.seen.append(config)


_SIM = f"--task={__name__}:_Recorder.simulate"


def _sim(args: list[str]) -> SimulationExperimentConfig:
    _Recorder.seen.clear()
    cli.run([_SIM, *args])
    return _Recorder.seen[-1]


@pytest.fixture
def trained(monkeypatch: pytest.MonkeyPatch) -> list[TrainingExperimentConfig]:
    """Records the configs the locomotion class's ``train`` is called with, instead of training."""
    seen: list[TrainingExperimentConfig] = []

    def train(_cls: type, config: TrainingExperimentConfig) -> None:
        seen.append(config)

    monkeypatch.setattr(LeggedRobotLocomotionManager, "train", classmethod(train))
    return seen


def _write(tmp_path: Path, content: dict[str, Any] | str, name: str = "run.yaml") -> str:
    path = tmp_path / name
    path.write_text(content if isinstance(content, str) else yaml.safe_dump(content))
    return str(path)


@pytest.mark.parametrize(
    ("content", "config_path", "name"),
    [
        (DOCUMENT, None, "run.yaml"),
        ({"launcher": "demo", "holosoma_sim": DOCUMENT}, None, "run.yaml"),
        ({"version": 1, "nested": {"run": DOCUMENT}}, "nested.run", "run.yaml"),
        (json.dumps(DOCUMENT, indent="\t"), None, "run.json"),
    ],
    ids=["root", "embedded", "config-path", "json"],
)
def test_document_parses_like_equivalent_cli(
    tmp_path: Path, content: dict[str, Any] | str, config_path: str | None, name: str
) -> None:
    args = ["--config", _write(tmp_path, content, name)]
    if config_path:
        args += ["--config-path", config_path]
    assert _sim(args) == _sim(EQUIVALENT_CLI)


def test_command_line_wins_at_the_same_path_and_deeper_document_edits_still_apply(tmp_path: Path) -> None:
    doc = {
        "version": 1,
        "arguments": {
            "robot": "g1-29dof",
            "robot.asset.link_physics": "none",
            "overrides": {"run.seed": 3},
        },
    }
    args = ["--config", _write(tmp_path, doc), "--run.seed", "-1", "robot:t1-29dof-waist-wrist"]
    expected = ["robot:t1-29dof-waist-wrist", "robot.asset.link_physics:none", "--run.seed", "-1"]
    assert _sim(args) == _sim(expected)


def test_scalars_follow_yaml_1_2(tmp_path: Path) -> None:
    text = (
        "version: 1\n"
        "arguments:\n"
        "  overrides:\n"
        "    run.seed: 010\n"
        "    run.name: off\n"
        "    run.project: 2026-11-17\n"
        "    robot.init_state.pos: [1e-05, 0.0, 0.8]\n"
    )
    config = _sim(["--config", _write(tmp_path, text)])
    assert (config.run.seed, config.run.name, config.run.project) == (10, "off", "2026-11-17")
    assert config.robot.init_state.pos == [1e-05, 0.0, 0.8]


@pytest.mark.parametrize(
    ("content", "extra_args"),
    [
        ({**DOCUMENT, "version": 2}, []),
        ({**DOCUMENT, "backend": "mujoco"}, []),
        ("version: 1\narguments: {}\narguments: {}\n", []),
        (DOCUMENT, ["--config", "other.yaml"]),
        ({**DOCUMENT, "task": "x"}, []),
    ],
    ids=["unsupported-version", "unknown-field", "duplicate-key", "repeated-config", "top-level-task"],
)
def test_rejects_invalid_documents(tmp_path: Path, content: dict[str, Any] | str, extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _sim(["--config", _write(tmp_path, content), *extra_args])


def test_packaged_examples_parse(trained: list[TrainingExperimentConfig]) -> None:
    _sim(["--config", "@holosoma/examples/configs/g1_lidar.yaml"])
    cli.run(["--config", "@holosoma/examples/configs/g1_locomotion_train.yaml", "simulator:mjwarp"])
    (config,) = trained
    assert (config.run.project, config.simulator.config.name, config.run.seed) == ("hv-g1-manager", "mujoco", 1)


def test_rejected_document_tokens_name_their_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, {"version": 1, "arguments": {"overrides": {"robot.init_state.posx": [1.0, 2.0]}}})
    with pytest.raises(SystemExit):
        _sim(["--config", path])
    assert f"<- {path}: arguments.overrides.robot.init_state.posx" in capsys.readouterr().err


def test_print_args_prints_the_equivalent_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.run([_SIM, "--config", _write(tmp_path, DOCUMENT), "--print-args"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert printed.startswith("holosoma ")
    _Recorder.seen.clear()
    cli.run(shlex.split(printed)[1:])
    assert _Recorder.seen[-1] == _sim(EQUIVALENT_CLI)


def test_old_spellings_give_the_same_config(tmp_path: Path, trained: list[TrainingExperimentConfig]) -> None:
    doc = {"version": 1, "arguments": {"overrides": {"training.project": "p", "time_scale": 0.5}}}
    old = ["exp:g1-29dof", "simulator:mjwarp", "--training.num_envs=8", "--training.checkpoint=/runs/a.pt"]
    new = [
        "exp:g1-29dof",
        "simulator:mjwarp",
        "--run.num_envs=8",
        "--algo.resume_from=/runs/a.pt",
        "--run.project=p",
        "--run.time_scale=0.5",
    ]
    cli.run([*old, "--config", _write(tmp_path, doc)])
    cli.run(new)
    assert trained[0] == trained[1]


def test_written_record_reproduces_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [_SIM, "simulator:mjwarp-sim2sim", "sensor.scan:g1-pelvis-lidar", "--run.time_scale", "0.50"]
    _Recorder.seen.clear()
    cli.run(args)
    parsed = _Recorder.seen[-1]
    cli.write_run_record(tmp_path)
    recorded = tmp_path / cli.RUN_RECORD_NAME

    _Recorder.seen.clear()
    cli.run(["--config", str(recorded)])
    assert _Recorder.seen[-1] == parsed

    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.run([*args, "--to-config"])
    assert (
        capsys.readouterr().out.splitlines()[-len(recorded.read_text().splitlines()) + 1 :]
        == (recorded.read_text().splitlines()[1:])
    )
