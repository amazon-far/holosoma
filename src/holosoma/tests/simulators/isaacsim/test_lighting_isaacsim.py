"""Live IsaacSim lighting test: a ``SceneConfig.lights`` rig authors the expected ``UsdLux`` prims.

``pxr`` is only importable inside a launched Isaac app, so this drives the ``scene_spawn_assert.py``
harness (``--probe-lights``): it injects a rig (one of each supported light type), launches IsaacSim
headless with two environments, and asserts the global lights (``sun``, ``dome``) were authored once at
``/World`` and the positioned lights (``lamp``, ``panel``) got a per-environment copy, all with the
right UsdLux types and attributes (and no fallback dome). Runs in a subprocess (SimulationContext is a
process singleton).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


def _has_isaacsim() -> bool:
    if importlib.util.find_spec("isaaclab") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.isaacsim
@pytest.mark.skipif(not _has_isaacsim(), reason="IsaacSim requires Isaac Lab and a CUDA device")
def test_scene_lights_authored(tmp_path, run_isaacsim_harness) -> None:
    """Launch IsaacSim with a declarative light rig and assert the UsdLux prims were authored."""
    result_file = tmp_path / "scene_lights.txt"
    run_isaacsim_harness(
        _HARNESS,
        "--headless",
        "true",
        "--num-envs",
        "2",
        "--probe-lights",
        "--result-file",
        str(result_file),
        label="isaacsim SceneConfig.lights authoring",
        timeout=900,
        result_file=result_file,
    )
