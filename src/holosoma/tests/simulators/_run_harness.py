"""Shared subprocess-harness runner for the per-process simulator tests.

Several backends must build their sim in a fresh process (IsaacSim/Warp teardown can mask the
exit code, so a failure would otherwise read as a pass). Those tests all run
an ``*_assert.py`` harness via ``subprocess`` and assert success the same way — either on the
process return code, or on an ``--result-file`` sentinel the harness writes after its checks
pass (more robust when teardown corrupts the exit code). That boilerplate lived copy-pasted
across ~8 runner tests; it lives here once.

Helpers raise ``AssertionError`` (with truncated stdout/stderr) on failure, so they read as
plain assertions inside a test.
"""

from __future__ import annotations

import json
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
        Extra CLI args passed after the harness path (e.g. ``"--simulator", "isaacsim"``).
    label:
        Human-readable scenario name for the failure message (e.g. ``"isaacsim/multibody"``).
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


_BATCH_RUNNER = Path(__file__).resolve().parent / "_batch_harness.py"
_BATCHES: dict[str, tuple[dict, subprocess.CompletedProcess, Path]] = {}


def run_batched(request: pytest.FixtureRequest, cell_of, *, label: str, timeout: float) -> None:
    """Run this test's harness cell in a subprocess shared with every selected cell of its group.

    ``cell_of(item)`` returns ``(group, cell_id, harness_stem, args)`` for a test item that runs a
    batched cell, else ``None``. On the first call for a group, every selected, non-``skip``-marked
    item of that group is run in one ``_batch_harness.py`` subprocess (``timeout`` covers the whole
    batch); each test then asserts its own cell: exit 77 skips, otherwise the harness must have written
    ``OK`` to the ``--result-file`` appended to its args.
    """
    group, cell_id, _, _ = cell_of(request.node)
    if group not in _BATCHES:
        out_dir = request.getfixturevalue("tmp_path_factory").mktemp("harness_batch")
        cells = {}
        for item in request.session.items:
            cell = cell_of(item)
            if cell is None or cell[0] != group or item.get_closest_marker("skip"):
                continue
            _, cid, stem, args = cell
            cells[cid] = [stem, [*args, "--result-file", str(out_dir / f"{cid}.ok")]]
        spec, out = out_dir / "spec.json", out_dir / "out.json"
        spec.write_text(json.dumps(cells))
        try:
            proc = subprocess.run(
                [sys.executable, str(_BATCH_RUNNER), str(spec), str(out)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            proc = subprocess.CompletedProcess(e.cmd, -9, str(e.stdout or ""), str(e.stderr or ""))
        results = json.loads(out.read_text()) if out.exists() else {}
        _BATCHES[group] = (results, proc, out_dir)
    results, proc, out_dir = _BATCHES[group]
    cell = results.get(cell_id)
    if cell is None:
        extra = f" (batch {group} did not reach this cell)"
        raise AssertionError(_fail_message(label, proc.returncode, proc.stdout, proc.stderr, extra=extra))
    rc, stdout = cell["returncode"], cell["stdout"]
    if rc == SKIP_EXIT_CODE:
        pytest.skip(_skip_reason(stdout, label))
    ok_file = out_dir / f"{cell_id}.ok"
    verdict = ok_file.read_text().strip() if ok_file.exists() else "<no result file>"
    assert verdict == "OK", _fail_message(label, rc, f"{stdout}\n--- result file ---\n{verdict}", proc.stderr)
