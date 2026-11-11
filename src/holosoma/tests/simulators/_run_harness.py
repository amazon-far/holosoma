"""Shared runner for the simulator tests that drive an ``*_assert.py`` harness.

A harness runs in a fresh subprocess by default (``run_harness``), in a long-lived worker shared with
other cases (``run_harness(..., reuse_process=True)``), or in one subprocess with the rest of its group
(``run_batched``). Success is judged on the return code, or on an ``--result-file`` sentinel the
harness writes after its checks pass (robust when teardown corrupts the exit code).

Helpers raise ``AssertionError`` (with truncated stdout/stderr) on failure, so they read as
plain assertions inside a test.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Autotools-convention "skip" exit code. A harness that hits a genuinely-runtime skip condition
# (one the wrapper cannot predict statically — a feature probe / library-availability check)
# exits with this code and prints a ``SKIP: <reason>`` line; run_harness turns that into a real
# pytest.skip. Chosen so it cannot collide with: 0 (pass), 1 (the harness's own FAIL), or a
# crash (a process killed by signal N exits 128+N >= 129).
SKIP_EXIT_CODE = 77

_WORKER_SCRIPT = Path(__file__).with_name("_harness_worker.py")
_worker: dict[str, Any] = {}  # "proc": the warm worker Popen, "reply": its reply pipe


def _stop_worker() -> None:
    proc = _worker.pop("proc", None)
    _worker.pop("reply", None)
    if proc is not None:
        proc.kill()
        proc.wait()


atexit.register(_stop_worker)


def _run_in_worker(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run ``cmd`` (harness path + args) in the shared warm worker; same result shape as ``subprocess.run``.

    A case that fails, crashes or times out also discards the worker, so the next case starts clean; a
    timeout is reported as returncode -9 with the output the case printed before it.
    """
    if "proc" not in _worker or _worker["proc"].poll() is not None:
        read_fd, write_fd = os.pipe()
        _worker["proc"] = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), str(write_fd)], stdin=subprocess.PIPE, pass_fds=(write_fd,), text=True
        )
        os.close(write_fd)
        _worker["reply"] = os.fdopen(read_fd)
    proc, reply = _worker["proc"], _worker["reply"]
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as log:
        log_path = Path(log.name)
    try:
        proc.stdin.write(json.dumps({"harness": cmd[0], "args": cmd[1:], "log": str(log_path)}) + "\n")
        proc.stdin.flush()
        if not select.select([reply], [], [], timeout)[0]:
            _stop_worker()
            output = log_path.read_text(encoding="utf-8", errors="replace")
            return subprocess.CompletedProcess(cmd, -9, f"{output}\n[timed out after {timeout:.0f} s]", "")
        line = reply.readline()
        returncode = json.loads(line)["returncode"] if line else proc.wait()
        if returncode not in (0, SKIP_EXIT_CODE):
            _stop_worker()
        output = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        log_path.unlink()
    return subprocess.CompletedProcess(cmd, returncode, output, "")


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
    reuse_process: bool = False,
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
    reuse_process:
        Run the harness ``main()`` in a warm worker process shared with other ``reuse_process`` calls
        instead of a fresh interpreter. Only for MuJoCo harnesses that return their exit code from
        ``main()`` and build no viewer; stdout and stderr are merged into ``stdout``.

    A harness exit code of ``SKIP_EXIT_CODE`` (77) is translated into ``pytest.skip`` with the
    reason read from the harness's ``SKIP:`` stdout line — for runtime-only skip conditions the
    wrapper cannot predict statically. Statically-knowable skips belong in the wrapper as
    ``pytest.mark.skip``/``skipif``, not here.

    Returns the ``CompletedProcess`` so callers can make further assertions if needed.
    """
    if reuse_process:
        result = _run_in_worker([str(harness), *args], timeout)
    else:
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


def _will_run(item: pytest.Item) -> bool:
    """Whether pytest will run ``item``'s body: not skipped (``skip``/``skipif``) and not ``xfail(run=False)``."""
    from _pytest.skipping import evaluate_skip_marks, evaluate_xfail_marks

    if evaluate_skip_marks(item) is not None:
        return False
    xfail = evaluate_xfail_marks(item)
    return xfail is None or xfail.run


def run_batched(request: pytest.FixtureRequest, cell_of, *, label: str, timeout: float) -> None:
    """Run this test's harness cell in a subprocess shared with every selected cell of its group.

    ``cell_of(item)`` returns ``(group, cell_id, harness_stem, args)`` for a test item that runs a
    batched cell, else ``None``. On the first call for a group, every selected item of that group that
    pytest will run (see ``_will_run``) is run in one ``_batch_harness.py`` subprocess (``timeout`` covers the whole
    batch); each test then asserts its own cell: exit 77 skips, otherwise the harness must have written
    ``OK`` to the ``--result-file`` appended to its args.
    """
    group, cell_id, _, _ = cell_of(request.node)
    if group not in _BATCHES:
        out_dir = request.getfixturevalue("tmp_path_factory").mktemp("harness_batch")
        cells = {}
        for item in request.session.items:
            cell = cell_of(item)
            if cell is None or cell[0] != group or not _will_run(item):
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
