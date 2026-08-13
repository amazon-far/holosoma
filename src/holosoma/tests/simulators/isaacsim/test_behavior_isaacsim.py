"""Behavioral physics tests on the IsaacSim backend via the behavior_assert.py harness.

Each test runs one behavioral scenario in its own subprocess at num_envs=4 and asserts the
harness exited 0 — the harness asserts a physical OUTCOME in EVERY env. Marked ``isaacsim`` so the
IsaacSim CI job (``-m isaacsim``) collects it; importorskip
("isaaclab")/CUDA-gated for direct/unfiltered runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "behavior_assert.py"

# The full behavioral catalog runs on IsaacSim at num_envs=4 — every scenario (including the
# multi-env ones) is supported here, so there are no static skips to mark. USD-backed; every
# behavioral preset ships a usd file. damping/restitution use the native PhysX fields here.
_SCENARIOS = [
    "collide-into-fixed",
    "momentum-transfer",
    "static-support",
    "friction-slide",
    "dr-friction-governs",
    "dr-damping-governs",
    "galileo-freefall",
    "damping-decay",
    "restitution-bounce",
    "angular-velocity-spin",
    "velocity-restore-reset",
    "object-obs-robot-pose",
    "pose-jitter-settle",
    "loader-invariance",
    "loader-invariance-file",
    "multibody-independence",
    "per-env-relocation",
    "actor-set-states-robot-object",
    # link_physics robot/object parity (one-link robot vs object twin) plus the friction-combine-mode
    # semantics guard (the material bind is IsaacSim-specific). damped-fall-twin exercises the
    # physx damping path.
    "galileo-twin",
    "damped-fall-twin",
    "restitution-twin",
    "combine-mode-collision",
    "robot-combine-mode",
]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_behavior_isaacsim(scenario):
    run_harness(
        _HARNESS,
        "--scenario",
        scenario,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        label=f"isaacsim/{scenario} (num_envs=4)",
        timeout=900,
    )
