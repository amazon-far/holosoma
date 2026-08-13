"""Shared subprocess-harness runner for the per-process simulator tests.

Several backends must build their sim in a fresh process (IsaacGym segfaults on a second
gymapi sim per process; IsaacSim/Warp teardown can mask the exit code). Those tests all run
an ``*_assert.py`` harness via ``subprocess`` and assert success the same way — either on the
process return code, or on an ``--result-file`` sentinel the harness writes after its checks
pass (more robust when teardown corrupts the exit code). That boilerplate lived copy-pasted
across ~8 runner tests; it lives here once.

Helpers raise ``AssertionError`` (with truncated stdout/stderr) on failure, so they read as
plain assertions inside a test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Autotools-convention "skip" exit code. A harness that hits a genuinely-runtime skip condition
# (one the wrapper cannot predict statically — a feature probe / library-availability check)
# exits with this code and prints a ``SKIP: <reason>`` line; run_harness turns that into a real
# pytest.skip. Chosen so it cannot collide with: 0 (pass), 1 (the harness's own FAIL), or a
# crash (a process killed by signal N exits 128+N >= 129).
SKIP_EXIT_CODE = 77


def _fail_message(label: str, returncode: int, stdout: str, stderr: str, extra: str = "") -> str:
    """Standard failure message: a one-line summary plus truncated stdout/stderr tails."""
    head = f"{label} failed (exit {returncode}){extra}."
    return f"{head}\n--- stdout ---\n{stdout[-3000:]}\n--- stderr ---\n{stderr[-2000:]}"


def _skip_reason(stdout: str, label: str) -> str:
    """Extract the harness's ``SKIP: <reason>`` line from stdout (last one wins)."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("SKIP:"):
            return f"{label}: {stripped[len('SKIP:') :].strip()}"
    return f"{label}: harness reported skip (exit {SKIP_EXIT_CODE})"


def run_harness(
    harness: str | Path,
    *args: str,
    label: str,
    timeout: float,
    result_file: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``harness`` (under the current interpreter) with ``args`` and assert it succeeded.

    Parameters
    ----------
    harness:
        Path to the ``*_assert.py`` harness script.
    *args:
        Extra CLI args passed after the harness path (e.g. ``"--simulator", "isaacgym"``).
    label:
        Human-readable scenario name for the failure message (e.g. ``"isaacgym/multibody"``).
    timeout:
        Subprocess timeout in seconds.
    result_file:
        If given, the harness is expected to write ``"OK"`` to this path once all checks pass,
        and success is judged on that sentinel rather than the return code (robust against
        teardown that corrupts the exit code). If ``None``, success is ``returncode == 0``.

    A harness exit code of ``SKIP_EXIT_CODE`` (77) is translated into ``pytest.skip`` with the
    reason read from the harness's ``SKIP:`` stdout line — for runtime-only skip conditions the
    wrapper cannot predict statically. Statically-knowable skips belong in the wrapper as
    ``pytest.mark.skip``/``skipif``, not here.

    Returns the ``CompletedProcess`` so callers can make further assertions if needed.
    """
    result = subprocess.run(
        [sys.executable, str(harness), *args],
        capture_output=True,
        text=True,
        # Decode harness output as UTF-8 (it prints non-ASCII chars like ✓/—); otherwise text=True
        # uses the locale codec, which is ASCII in the hsgym Py3.8 container and crashes on 0xe2.
        # errors="replace" so stray bytes can never mask a real result.
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    # Runtime skip: the harness signalled (via the autotools-convention exit 77) that this cell is
    # unsupported in a way only knowable at runtime. Surface it as a real pytest skip with reason,
    # not a pass. Checked before the success assertions so a skip is never mistaken for a failure.
    if result.returncode == SKIP_EXIT_CODE:
        pytest.skip(_skip_reason(result.stdout, label))
    if result_file is not None:
        result_file = Path(result_file)
        ok = result_file.exists() and result_file.read_text().strip() == "OK"
        assert ok, _fail_message(
            label,
            result.returncode,
            result.stdout,
            result.stderr,
            extra=f", result-file present={result_file.exists()}",
        )
    else:
        assert result.returncode == 0, _fail_message(label, result.returncode, result.stdout, result.stderr)
    return result
