"""Behavioral physics tests on the IsaacSim backend via the behavior_assert.py harness.

Each test asserts one behavioral scenario at num_envs=4; all selected scenarios run in one shared
subprocess and Kit app, each on a fresh stage. The harness asserts a physical OUTCOME in EVERY env.
Marked ``isaacsim`` so the IsaacSim CI job (``-m isaacsim``) collects it; importorskip
("isaaclab")/CUDA-gated for direct/unfiltered runs.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_batched  # noqa: E402

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
    "external-force",
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
    pytest.param(
        "galileo-twin",
        marks=pytest.mark.xfail(
            strict=True,
            reason="robot link drops 0.508 m vs object twin 0.478 m (predicted 0.500): robot/object fall parity fails",
        ),
    ),
    "damped-fall-twin",
    "restitution-twin",
    "combine-mode-collision",
    "robot-combine-mode",
]


def _cell(item):
    """Batch cell for a behavior test item: one subprocess runs every selected scenario."""
    if getattr(item, "module", None) is not sys.modules[__name__] or item.originalname != "test_behavior_isaacsim":
        return None
    scenario = item.callspec.params["scenario"]
    args = ["--scenario", scenario, "--simulator", "isaacsim", "--num-envs", "4"]
    return "behavior/isaacsim", scenario, "behavior_assert", args


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_behavior_isaacsim(scenario, request):
    run_batched(request, _cell, label=f"isaacsim/{scenario} (num_envs=4)", timeout=1800)
