"""Pytest config: safe torch import order + backend-marker taxonomy.

CI selects tests positively: each job runs ``pytest -m "<backend>"`` for one run-target marker.

Run-target markers (exactly one per test; mutually exclusive):
    isaacgym / isaacsim / mujoco  — needs that simulator backend
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

import os

import pytest

# Import torch before any isaacgym import during collection.
from holosoma.utils.safe_torch_import import torch  # noqa: F401

_RUN_TARGETS = ("isaacgym", "isaacsim", "mujoco", "no_sim", "requires_inference")
_SIM_DIRS = ("isaacgym", "isaacsim", "mujoco")

_ALL_MARKERS = {
    "isaacgym": "Isaac Gym",
    "isaacsim": "Isaac Sim",
    "mujoco": "MuJoCo (either backend)",
    "mujoco_classic": "the MuJoCo ClassicBackend (CPU)",
    "mujoco_warp": "the MuJoCo WarpBackend (CUDA)",
    "no_sim": "no simulator (pure / CPU-only)",
    "multi_gpu": "multiple GPUs",
    "requires_inference": "the inference environment",
}


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
