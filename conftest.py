"""Pytest config: backend-marker taxonomy.

CI selects tests positively: each job runs ``pytest -m "<backend>"`` for one run-target marker.

Run-target markers (exactly one per test; mutually exclusive):
    isaacsim / mujoco             — needs that simulator backend
    no_sim                        — pure test, no simulator (CPU job)
    requires_inference            — needs the inference env (e2e job)

Sub-tags (not used to pick a job; for manual ``-m`` selection):
    mujoco_classic / mujoco_warp  — which MuJoCo backend a mujoco test uses
    multi_gpu                     — needs multiple GPUs

``pytest_collection_modifyitems`` auto-applies the umbrella by directory and falls back to
no_sim, so every item ends with exactly one run-target and positive ``-m`` selection cannot
skip a test by omission. A sim test outside ``tests/simulators/<backend>/`` must set its
umbrella explicitly; a pure test inside such a dir opts out with ``pytest.mark.no_sim``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from functools import partial

import pytest

_RUN_TARGETS = ("isaacsim", "mujoco", "no_sim", "requires_inference")
_SIM_DIRS = ("isaacsim", "mujoco")

_ALL_MARKERS = {
    "isaacsim": "Isaac Sim",
    "mujoco": "MuJoCo (either backend)",
    "mujoco_classic": "the MuJoCo ClassicBackend (CPU)",
    "mujoco_warp": "the MuJoCo WarpBackend (CUDA)",
    "multi_gpu": "multiple GPUs",
    "no_sim": "no simulator (pure / CPU-only)",
    "requires_inference": "the inference environment",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--isaacsim-backend",
        choices=["isaacsim5", "isaacsim6"],
        default="isaacsim5",
        help="Isaac Sim version for the simulator tests (default: isaacsim5)",
    )


@pytest.fixture(scope="session")
def isaacsim_backend(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("--isaacsim-backend")


@pytest.fixture(scope="session")
def isaacsim_simulator(isaacsim_backend: str) -> str:
    return "isaacsim6" if isaacsim_backend == "isaacsim6" else "isaacsim"


@pytest.fixture
def run_isaacsim_harness(isaacsim_simulator: str) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run each native child with the explicitly selected pytest profile."""
    from tests.simulators._run_harness import run_isaacsim_harness as run_selected_harness

    return partial(run_selected_harness, simulator=isaacsim_simulator)


def pytest_configure(config):
    for marker, msg in _ALL_MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: marks tests as requiring {msg}")


def pytest_collection_modifyitems(config, items):
    for item in items:
        own = {m.name for m in item.iter_markers()}

        # mujoco_classic/mujoco_warp imply the mujoco umbrella.
        if (own & {"mujoco_classic", "mujoco_warp"}) and "mujoco" not in own:
            item.add_marker(pytest.mark.mujoco)
            own.add("mujoco")

        # Directory umbrella, unless an explicit run-target already set it.
        if not (own & set(_RUN_TARGETS)):
            path = str(getattr(item, "path", item.fspath)).replace(os.sep, "/")
            for backend in _SIM_DIRS:
                if f"/tests/simulators/{backend}/" in path:
                    item.add_marker(getattr(pytest.mark, backend))
                    own.add(backend)
                    break

        # Fallback for anything without a run-target.
        if not (own & set(_RUN_TARGETS)):
            item.add_marker(pytest.mark.no_sim)
