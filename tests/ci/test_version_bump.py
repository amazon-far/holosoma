"""Bind the version-bump gate on throwaway git repositories."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.no_sim

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_version_bump", REPO_ROOT / "scripts/ci/check_version_bump.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
# dataclasses resolves the defining module through sys.modules.
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

PYPROJECT = "src/holosoma_inference/pyproject.toml"
PYPROJECT_TEXT = """[build-system]
requires = ["scikit-build-core"]

[project]
name = "holosoma-inference"
version = "{version}"

[tool.other]
version = "9.9.9"
"""


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def _write(repo: Path, path: str, text: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, PYPROJECT, PYPROJECT_TEXT.format(version="0.1.4"))
    _write(tmp_path, "src/holosoma_inference/holosoma_inference/depth.py", "A = 1\n")
    _write(tmp_path, "src/holosoma_inference/holosoma_cpp/src/trt_backend.cpp", "int a;\n")
    _write(tmp_path, "src/holosoma/holosoma/train_agent.py", "B = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "switch", "-q", "-c", "feature")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit_and_gate(repo: Path, changes: dict[str, str]) -> int:
    for path, text in changes.items():
        _write(repo, path, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return gate.main(["--base", "main", "--head", "HEAD"])


@pytest.mark.parametrize(
    "path",
    [
        "src/holosoma_inference/holosoma_inference/depth.py",
        "src/holosoma_inference/holosoma_cpp/src/trt_backend.cpp",
    ],
)
def test_a_shipped_change_under_an_unchanged_version_fails(repo, path, capsys):
    assert _commit_and_gate(repo, {path: "changed\n"}) == 1
    assert path in capsys.readouterr().out


def test_a_shipped_change_with_a_bump_passes(repo):
    changes = {
        "src/holosoma_inference/holosoma_inference/depth.py": "A = 2\n",
        PYPROJECT: PYPROJECT_TEXT.format(version="0.1.5"),
    }
    assert _commit_and_gate(repo, changes) == 0


@pytest.mark.parametrize(
    "path",
    [
        "src/holosoma_inference/holosoma_inference/tests/test_depth.py",
        "src/holosoma_inference/holosoma_cpp/tests/test_backend.cpp",
        "src/holosoma_inference/holosoma_inference/README.md",
        "src/holosoma/holosoma/train_agent.py",
    ],
)
def test_a_change_that_ships_nothing_passes(repo, path):
    assert _commit_and_gate(repo, {path: "changed\n"}) == 0


def test_a_version_that_goes_backwards_fails(repo):
    changes = {
        "src/holosoma_inference/holosoma_inference/depth.py": "A = 2\n",
        PYPROJECT: PYPROJECT_TEXT.format(version="0.1.3"),
    }
    assert _commit_and_gate(repo, changes) == 1


def test_the_version_is_read_from_the_project_table_only():
    assert gate.read_version(PYPROJECT_TEXT.format(version="0.1.4")) == "0.1.4"
    with pytest.raises(ValueError, match=r"no static \[project\] version"):
        gate.read_version('[tool.other]\nversion = "1.0"\n')


def test_the_published_pyproject_has_a_readable_version():
    assert gate.read_version((REPO_ROOT / PYPROJECT).read_text())


def test_pull_requests_run_the_gate_unless_labelled():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/version-bump.yaml").read_text())
    # PyYAML reads the bare `on` key as True.
    triggers = workflow[True]["pull_request"]["types"]
    assert {"opened", "synchronize", "labeled", "unlabeled"} <= set(triggers)
    steps = workflow["jobs"]["version-bump"]["steps"]
    gate_step = next(step for step in steps if "check_version_bump.py" in step.get("run", ""))
    assert "--base" in gate_step["run"]
    assert gate_step["if"] == "${{ !contains(github.event.pull_request.labels.*.name, 'skip-version-bump') }}"
