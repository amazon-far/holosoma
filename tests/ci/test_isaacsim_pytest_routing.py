"""Exercise explicit pytest profiles through real child argument parsing and completion files."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.no_sim
ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--simulator", required=True, choices=["isaaclab2", "isaaclab3", "isaacsim"])
parser.add_argument("--report", required=True)
parser.add_argument("--result-file", required=True)
parser.add_argument("--completion", choices=["ok", "missing", "fail"], default="ok")
parser.add_argument("--skip", action="store_true")
args = parser.parse_args()
if args.skip:
    print("SKIP: controlled external feature unavailable")
    sys.exit(77)
Path(args.report).write_text(json.dumps({"simulator": args.simulator, "argv": sys.argv[1:]}))
if args.completion != "missing":
    Path(args.result_file).write_text("OK\\n" if args.completion == "ok" else "FAIL\\n")
"""

_PROFILE_TESTS = """
import json
from pathlib import Path

import pytest

from tests.simulators._run_harness import run_isaacsim_harness as run_selected

CHILD = Path(__file__).with_name("child.py")
EXPECTED_BACKEND = {backend!r}
EXPECTED_PRESET = {preset!r}

@pytest.mark.parametrize("caller_args", [
    [],
    ["--simulator", "isaacsim"],
    ["--simulator=isaaclab3"],
    ["--simulator", "isaacsim", "--simulator=isaaclab3"],
])
def test_child_receives_selected_profile(
    tmp_path, isaaclab_backend, isaacsim_simulator, run_isaacsim_harness, caller_args
):
    assert isaaclab_backend == EXPECTED_BACKEND
    assert isaacsim_simulator == EXPECTED_PRESET
    report = tmp_path / "child.json"
    result_file = tmp_path / "complete.txt"
    result = run_isaacsim_harness(
        CHILD, *caller_args, "--report", str(report), "--result-file", str(result_file),
        label="profile child", timeout=10, result_file=result_file
    )
    assert result.returncode == 0
    actual = json.loads(report.read_text())
    assert actual["simulator"] == EXPECTED_PRESET
    assert actual["argv"].count("--simulator") == 1
    assert result_file.read_text() == "OK\\n"

@pytest.mark.parametrize("completion", ["missing", "fail"])
def test_routing_does_not_admit_exit_zero_without_completion(tmp_path, run_isaacsim_harness, completion):
    report = tmp_path / "child.json"
    result_file = tmp_path / "complete.txt"
    with pytest.raises(AssertionError, match="profile child failed"):
        run_isaacsim_harness(
            CHILD, "--report", str(report), "--result-file", str(result_file),
            "--completion", completion, label="profile child", timeout=10, result_file=result_file
        )
    assert json.loads(report.read_text())["simulator"] == EXPECTED_PRESET

def test_runtime_skip_survives_profile_routing(tmp_path, run_isaacsim_harness):
    with pytest.raises(pytest.skip.Exception, match="controlled external feature unavailable"):
        run_isaacsim_harness(
            CHILD, "--report", str(tmp_path / "child.json"),
            "--result-file", str(tmp_path / "complete.txt"), "--skip",
            label="profile child", timeout=10, result_file=tmp_path / "complete.txt"
        )

def test_non_isaac_preset_cannot_enter_native_runner():
    with pytest.raises(ValueError, match="Expected an Isaac Sim preset"):
        run_selected(CHILD, simulator="mujoco", label="profile child", timeout=10)

def test_missing_caller_preset_is_not_silently_consumed(run_isaacsim_harness):
    with pytest.raises(ValueError, match="requires a preset value"):
        run_isaacsim_harness(CHILD, "--simulator", label="profile child", timeout=10)
"""


@pytest.mark.parametrize(
    ("option_name", "backend_option"),
    [
        ("--isaaclab-backend", None),
        ("--isaaclab-backend", "isaaclab2"),
        ("--isaaclab-backend", "isaaclab3"),
        ("--isaaclab-backend", "isaacsim"),
        ("--isaacsim-backend", "isaacsim"),
    ],
    ids=["default", "lab2", "lab3", "legacy-preset", "legacy-option"],
)
def test_pytest_profile_routes_real_children(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, option_name: str, backend_option: str | None
) -> None:
    backend = "isaaclab2" if backend_option in (None, "isaacsim") else backend_option
    preset = backend
    monkeypatch.setenv("HOLOSOMA_ISAACSIM_BACKEND", "isaaclab2" if backend == "isaaclab3" else "isaaclab3")
    pytester.makeconftest((ROOT / "conftest.py").read_text())
    pytester.makepyfile(child=_CHILD, test_profile=_PROFILE_TESTS.format(backend=backend, preset=preset))
    options = [] if backend_option is None else [option_name, backend_option]
    result = pytester.runpytest_subprocess("-q", "--strict-markers", *options)
    result.assert_outcomes(passed=9)


def test_pytest_rejects_unknown_backend(pytester: pytest.Pytester) -> None:
    pytester.makeconftest((ROOT / "conftest.py").read_text())
    pytester.makepyfile("def test_unused(): pass")
    result = pytester.runpytest_subprocess("--isaaclab-backend", "unknown")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "invalid choice: 'unknown'" in result.stderr.str()
