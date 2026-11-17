"""TensorRT engines built under different NVIDIA_TF32_OVERRIDE settings must not share a cache entry.

The setting is read at process start, so each arm runs the backend gtest binary in its own process
against one shared cache directory:

1. TF32 on, empty cache: builds one engine.
2. TF32 off, same model: must build a second engine rather than reuse the first (which fails
   to create an execution context, "Inconsistent setting of NVIDIA_TF32_OVERRIDE").
3. TF32 off, a different model: builds a third engine, proving the count can see a new plan.

Usage: python trt_cache_precision_check.py <holosoma_backend_tests> <working dir>
Exits 77 (skipped) when the binary reports no usable TensorRT device.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKIPPED = 77


def main(binary: Path, cwd: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="holosoma_trt_precision_") as directory:
        cache = Path(directory)

        def run(tf32_override: str, case: str) -> tuple[dict[str, str], bool, subprocess.CompletedProcess]:
            env = dict(os.environ, NVIDIA_TF32_OVERRIDE=tf32_override, HOLOSOMA_TRT_CACHE_DIR=str(cache))
            result = subprocess.run(
                [str(binary), f"--gtest_filter={case}"],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            plans = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.glob("*.engine")}
            return plans, "building an engine" in result.stderr, result

        first, built, result = run("1", "Backend.TensorRtMatchesOnnxRuntime")
        if "[  SKIPPED ]" in result.stdout:
            print("skipped: no usable TensorRT device")
            return SKIPPED
        assert result.returncode == 0, result.stdout + result.stderr
        assert built and len(first) == 1, "the first run must build one engine into an empty cache"

        second, rebuilt, result = run("0", "Backend.TensorRtMatchesOnnxRuntime")
        print(f"precision change: engines={len(second)} rebuilt={rebuilt} rc={result.returncode}")

        third, built, control = run("0", "Backend.TensorRtReusesACachedEngine")
        assert control.returncode == 0, control.stdout + control.stderr
        assert built and len(third) == len(second) + 1, "the count must see a different model's engine"

        assert result.returncode == 0, "TF32 off reused the TF32 engine:\n" + result.stderr
        assert re.search(r"\[\s+PASSED\s+\] 1 test", result.stdout), result.stdout
        assert rebuilt and len(second) == 2, "TF32 off must build its own engine"
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
