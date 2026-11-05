"""Run several harness ``main()`` cells in one process and record each cell's outcome.

Usage: ``python _batch_harness.py SPEC.json OUT.json``. ``SPEC`` maps a cell id to
``[harness_module_stem, [argv...]]``; ``OUT`` is rewritten after every cell with
``{cell_id: {"returncode": int, "stdout": str}}``, so cells finished before a crash keep their result.
Each cell builds its own sim; only the interpreter and imports are shared.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)


def main() -> int:
    spec_path, out_path = sys.argv[1:3]
    with open(spec_path) as f:
        cells = json.load(f)
    results: dict[str, dict] = {}
    for cell_id, (module, args) in cells.items():
        buf = io.StringIO()
        sys.argv = [module, *args]
        with contextlib.redirect_stdout(buf):
            print(f"=== cell {cell_id}: {module} {' '.join(args)}", flush=True)
            try:
                rc = importlib.import_module(f"tests.simulators.{module}").main()
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
            except Exception:
                traceback.print_exc(file=buf)
                rc = 1
        gc.collect()
        results[cell_id] = {"returncode": rc, "stdout": buf.getvalue()}
        print(buf.getvalue(), end="", flush=True)
        tmp = Path(f"{out_path}.tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(out_path)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
