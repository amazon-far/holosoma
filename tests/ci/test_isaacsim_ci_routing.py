"""Run the real CI shell routing with activation and Python at external boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_sim
ROOT = Path(__file__).resolve().parents[2]
ShellTree = tuple[Path, dict[str, str], Path]


@pytest.fixture
def shell_tree(tmp_path: Path) -> ShellTree:
    checkout = tmp_path / "checkout"
    ci = checkout / "tests/ci"
    ci.mkdir(parents=True)
    for name in ["isaaclab_ci_tests.sh", "no_sim_ci_tests.sh"]:
        source = (ROOT / "tests/ci" / name).read_text()
        assert source.count("cd /workspace/holosoma") == 1
        (ci / name).write_text(source.replace("cd /workspace/holosoma", 'cd "$CI_ROUTING_CHECKOUT"'))
    (ci / "isaacsim_ci_tests.sh").write_text((ROOT / "tests/ci/isaacsim_ci_tests.sh").read_text())
    scripts = checkout / "scripts"
    scripts.mkdir()
    for family, name in [
        ("isaaclab2", "source_isaaclab2_setup.sh"),
        ("isaaclab3", "source_isaaclab3_setup.sh"),
    ]:
        (scripts / name).write_text(f"export CI_ACTIVATED_PROFILE={family}\n")

    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text(
        """#!/bin/bash
"$CI_ROUTING_PYTHON" - "$@" <<'PY'
import json, os, sys
with open(os.environ["CI_ROUTING_LOG"], "a") as log:
    log.write(json.dumps([os.environ["CI_ACTIVATED_PROFILE"], *sys.argv[1:]]) + "\\n")
if sys.argv[1] == "scripts/validate_isaaclab_runtime.py":
    sys.exit(int(os.environ["CI_ROUTING_PREFLIGHT_EXIT"]))
PY
"""
    )
    python.chmod(0o755)
    log = tmp_path / "invocations.jsonl"
    log.touch()
    return (
        ci,
        {
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "CI_ROUTING_CHECKOUT": str(checkout),
            "CI_ROUTING_PYTHON": sys.executable,
            "CI_ROUTING_LOG": str(log),
        },
        log,
    )


def _run(
    shell_tree: ShellTree,
    script: str,
    args: list[str],
    *,
    inherited_backend: str = "isaaclab3",
    multigpu: str = "False",
    preflight_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    ci, env, log = shell_tree
    result = subprocess.run(
        ["bash", str(ci / script), *args],
        env={
            **env,
            # This obsolete setting must have no effect on version selection.
            "HOLOSOMA_ISAACSIM_BACKEND": inherited_backend,
            "HOLOSOMA_MULTIGPU": multigpu,
            "CI_ROUTING_PREFLIGHT_EXIT": str(preflight_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize(
    ("backend", "multigpu", "preflight_exit"),
    [
        ("isaaclab2", "False", 0),
        ("isaaclab2", "True", 0),
        ("isaaclab3", "False", 0),
        ("isaaclab3", "True", 0),
        ("isaaclab3", "False", 13),
    ],
)
def test_ci_routes_isaac_and_pure_suites(
    shell_tree: ShellTree, backend: str, multigpu: str, preflight_exit: int
) -> None:
    result, invocations = _run(
        shell_tree,
        "isaaclab_ci_tests.sh",
        ["--backend", backend],
        inherited_backend="isaaclab2" if backend == "isaaclab3" else "isaaclab3",
        multigpu=multigpu,
        preflight_exit=preflight_exit,
    )
    assert result.returncode == preflight_exit, result.stderr
    holosoma_install = ["-m", "pip", "install", "-e", "src/holosoma[unitree,booster]"]
    preflight = ["scripts/validate_isaaclab_runtime.py", "--backend", backend]
    if multigpu == "True":
        preflight.append("--multigpu")
    marker = "isaacsim and multi_gpu" if multigpu == "True" else "isaacsim and not multi_gpu"
    pytest_args = ["-m", "pytest", "-s", "--strict-markers", "--isaaclab-backend", backend, "-m"]
    ignores = ["--ignore=thirdparty", "--ignore=src/holosoma_inference"]
    expected = [holosoma_install, preflight]
    if preflight_exit == 0:
        expected.append([*pytest_args, marker, *ignores])
    if preflight_exit == 0 and backend == "isaaclab3" and multigpu == "False":
        expected.extend(
            [
                holosoma_install,
                ["-m", "pip", "install", "-e", "src/holosoma_inference"],
                [*pytest_args, "no_sim", *ignores],
            ]
        )
    assert invocations == [[backend, *args] for args in expected]


@pytest.mark.parametrize(
    ("args", "inherited_backend", "expected_backend"),
    [
        ([], "isaaclab3", "isaaclab2"),
        (["--backend", "isaaclab2"], "isaaclab3", "isaaclab2"),
        (["--backend", "isaaclab3"], "isaaclab2", "isaaclab3"),
        (["--backend", "isaacsim"], "isaaclab3", "isaaclab2"),
    ],
)
def test_pure_suite_selects_version_only_from_explicit_arguments(
    shell_tree: ShellTree, args: list[str], inherited_backend: str, expected_backend: str
) -> None:
    result, invocations = _run(shell_tree, "no_sim_ci_tests.sh", args, inherited_backend=inherited_backend)
    assert result.returncode == 0, result.stderr
    assert invocations == [
        [expected_backend, "-m", "pip", "install", "-e", "src/holosoma[unitree,booster]"],
        [expected_backend, "-m", "pip", "install", "-e", "src/holosoma_inference"],
        [
            expected_backend,
            "-m",
            "pytest",
            "-s",
            "--strict-markers",
            "--isaaclab-backend",
            expected_backend,
            "-m",
            "no_sim",
            "--ignore=thirdparty",
            "--ignore=src/holosoma_inference",
        ],
    ]


def test_released_ci_script_selects_lab2(shell_tree: ShellTree) -> None:
    result, invocations = _run(shell_tree, "isaacsim_ci_tests.sh", [])
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in invocations] == ["isaaclab2"] * 3
    assert invocations[1][1:] == ["scripts/validate_isaaclab_runtime.py", "--backend", "isaaclab2"]


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("isaaclab_ci_tests.sh", []),
        ("isaaclab_ci_tests.sh", ["--backend"]),
        ("isaaclab_ci_tests.sh", ["--backend", "unsupported"]),
        ("no_sim_ci_tests.sh", ["--backend"]),
        ("no_sim_ci_tests.sh", ["--backend", "unsupported"]),
    ],
)
def test_invalid_version_selection_fails_before_activation_or_install(
    shell_tree: ShellTree, script: str, args: list[str]
) -> None:
    result, invocations = _run(shell_tree, script, args)
    assert result.returncode == 2
    assert "Usage:" in result.stderr or "Unknown Isaac Sim backend:" in result.stderr
    assert invocations == []
