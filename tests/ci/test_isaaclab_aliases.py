"""Released script names forward to Lab 2 without changing arguments or activation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_sim
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.parametrize("name", ["setup", "reset"])
def test_released_commands_forward_arguments_and_exit_status(tmp_path: Path, name: str) -> None:
    alias = tmp_path / f"{name}_isaacsim.sh"
    shutil.copyfile(SCRIPTS / alias.name, alias)
    (tmp_path / f"{name}_isaaclab2.sh").write_text('printf "%s\\n" "$@" > "$ALIAS_ARGUMENT_LOG"\nexit 23\n')
    log = tmp_path / "arguments"
    result = subprocess.run(
        ["bash", str(alias), "argument with spaces", "--flag"],
        env={**os.environ, "ALIAS_ARGUMENT_LOG": str(log)},
        check=False,
    )
    assert result.returncode == 23
    assert log.read_text().splitlines() == ["argument with spaces", "--flag"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 23])
def test_released_activation_exports_environment_and_preserves_failure(tmp_path: Path, shell: str, status: int) -> None:
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is not installed")
    assert executable is not None
    alias = tmp_path / "source_isaacsim_setup.sh"
    shutil.copyfile(SCRIPTS / alias.name, alias)
    (tmp_path / "source_isaaclab2_setup.sh").write_text(f"export ALIAS_ACTIVATED_PROFILE=isaaclab2\nreturn {status}\n")
    result = subprocess.run(
        [
            executable,
            "-c",
            'source "$1"; status_code=$?; echo "$ALIAS_ACTIVATED_PROFILE"; exit "$status_code"',
            "test",
            str(alias),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr
    assert result.stdout.strip() == "isaaclab2"
