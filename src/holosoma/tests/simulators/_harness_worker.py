"""Long-lived process that runs ``*_assert.py`` harness ``main()`` functions in-process, one at a time.

Started by :func:`tests.simulators._run_harness.run_harness` with ``reuse_process=True``; cases
share the interpreter and its imports. Each stdin line is a JSON request ``{"harness", "args", "log"}``: the
harness module is loaded fresh, its fd-level stdout/stderr go to ``log``, and ``{"returncode": n}``
is written to the reply fd given as ``argv[1]``.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import traceback

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)


def _run(harness: str, args: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("__harness__", harness)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.argv = [harness, *args]
    try:
        spec.loader.exec_module(module)
        rc = module.main()
    except SystemExit as exc:
        rc = exc.code
    except BaseException:
        traceback.print_exc()
        rc = 1
    if rc is None:
        return 0
    return rc if isinstance(rc, int) else 1


def main() -> None:
    reply = os.fdopen(int(sys.argv[1]), "w")
    for line in sys.stdin:
        request = json.loads(line)
        sys.stdout.flush()
        sys.stderr.flush()
        log_fd = os.open(request["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        saved = os.dup(1), os.dup(2)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        try:
            rc = _run(request["harness"], request["args"])
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in (log_fd, *saved):
                os.close(fd)
        gc.collect()
        reply.write(json.dumps({"returncode": rc}) + "\n")
        reply.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
