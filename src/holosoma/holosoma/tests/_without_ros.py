"""Run optional dependency tests without ROS imports from other test modules."""

from __future__ import annotations

import functools
import inspect
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def without_ros(test: Callable[[], None]) -> Callable[[], None]:
    """Run the same test in a fresh interpreter that rejects ROS imports."""

    @functools.wraps(test)
    def run() -> None:
        if os.environ.get("HOLOSOMA_TEST_WITHOUT_ROS") == "1":
            test()
            return
        node = f"{Path(inspect.getfile(test)).resolve()}::{test.__name__}"
        script = """
import importlib.abc
import sys

attempts = []
class NoROS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rclpy" or fullname.startswith("rclpy."):
            attempts.append(fullname)
            raise AssertionError(f"Unexpected optional ROS import: {fullname}")

sys.meta_path.insert(0, NoROS())
import pytest
result = pytest.main([sys.argv[1], "-q", "-p", "no:cacheprovider"])
assert not attempts, f"Unexpected optional ROS imports: {attempts}"
raise SystemExit(result)
"""
        env = {**os.environ, "HOLOSOMA_TEST_WITHOUT_ROS": "1"}
        result = subprocess.run(
            [sys.executable, "-c", script, node], env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run
