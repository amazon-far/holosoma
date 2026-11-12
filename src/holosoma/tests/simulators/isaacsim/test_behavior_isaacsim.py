"""Behavioral physics tests on the IsaacSim backend via the behavior_assert.py harness.

Each test runs one behavioral scenario in its own subprocess at num_envs=4 and checks its
completion file — the harness asserts a physical OUTCOME in EVERY env. Marked ``isaacsim`` so the
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

_HARNESS = Path(__file__).resolve().parents[1] / "behavior_assert.py"

# The released PhysX catalog is preserved. Newton runs the same downstream assertions
# except explicitly PhysX-only damping/combine-mode contracts. All checks run in every world.
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
    "mass-force-response",
    "inertia-torque-response",
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
@pytest.mark.parametrize("physics", ["physx", "newton"])
def test_behavior_isaacsim(scenario, physics, isaacsim_backend, tmp_path, run_isaacsim_harness):
    if physics == "newton":
        if isaacsim_backend != "isaacsim6":
            pytest.skip("Newton MJWarp requires Isaac Lab 3")
        from tests.simulators.behavior_assert import NEWTON_UNSUPPORTED

        if scenario in NEWTON_UNSUPPORTED:
            pytest.skip(f"Newton MJWarp: {NEWTON_UNSUPPORTED[scenario]}")
    result_file = tmp_path / f"behavior_{scenario}.txt"
    result = run_isaacsim_harness(
        _HARNESS,
        "--scenario",
        scenario,
        "--physics-backend",
        physics,
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label=f"isaacsim/{physics}/{scenario} (num_envs=4)",
        timeout=900,
        result_file=result_file,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    output = (result.stdout + result.stderr).lower()
    assert all(message not in output for message in ("nefc overflow", "ncon overflow", "narrowphase overflow")), (
        result.stdout[-3000:]
    )
