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
def test_ci_routes_isaac_and_pure_suites(tmp_path: Path, backend: str, multigpu: str, preflight_exit: int) -> None:
    checkout = tmp_path / "checkout"
    ci = checkout / "tests/ci"
    ci.mkdir(parents=True)
    for name in ["isaacsim_ci_tests.sh", "no_sim_ci_tests.sh"]:
        source = (ROOT / "tests/ci" / name).read_text()
        assert source.count("cd /workspace/holosoma") == 1
        (ci / name).write_text(source.replace("cd /workspace/holosoma", 'cd "$CI_ROUTING_CHECKOUT"'))
    scripts = checkout / "scripts"
    scripts.mkdir()
    for family, name in [
        ("isaaclab2", "source_isaacsim_setup.sh"),
        ("isaaclab3", "source_isaacsim3_setup.sh"),
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
if sys.argv[1] == "scripts/validate_isaacsim_runtime.py":
    sys.exit(int(os.environ["CI_ROUTING_PREFLIGHT_EXIT"]))
PY
"""
    )
    python.chmod(0o755)
    log = tmp_path / "invocations.jsonl"
    result = subprocess.run(
        ["bash", str(ci / "isaacsim_ci_tests.sh")],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "HOLOSOMA_ISAACSIM_BACKEND": backend,
            "HOLOSOMA_MULTIGPU": multigpu,
            "CI_ROUTING_CHECKOUT": str(checkout),
            "CI_ROUTING_PYTHON": sys.executable,
            "CI_ROUTING_LOG": str(log),
            "CI_ROUTING_PREFLIGHT_EXIT": str(preflight_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == preflight_exit, result.stderr
    holosoma_install = ["-m", "pip", "install", "-e", "src/holosoma[unitree,booster]"]
    preflight = ["scripts/validate_isaacsim_runtime.py", "--backend", backend]
    if multigpu == "True":
        preflight.append("--multigpu")
    marker = "isaacsim and multi_gpu" if multigpu == "True" else "isaacsim and not multi_gpu"
    pytest_args = ["-m", "pytest", "-s", "--strict-markers", "-m"]
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
    assert [json.loads(line) for line in log.read_text().splitlines()] == [[backend, *args] for args in expected]
