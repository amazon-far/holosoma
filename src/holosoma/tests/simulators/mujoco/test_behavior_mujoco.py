"""Behavioral physics tests on the MuJoCo backends via the behavior_assert.py harness.

Each test runs one behavioral scenario in its own subprocess (one sim per process) and asserts
the harness exited 0. The harness asserts a physical OUTCOME after stepping real physics (a body
stops at a wall, rests on a post, bounces, spins by the commanded angle, ...) in EVERY env — a
behavioral pass proves the configured value does physically meaningful work, not just that an API
echoes it.

Two MuJoCo backends:
  - ClassicBackend (CPU, single-env): the single-env scenarios.
  - WarpBackend (GPU, multi-env, CUDA-gated): the full catalog at num_envs=4, including the
    multi-env-only scenarios (per-env independence / indexing).

The full scenario catalog is parametrized for BOTH backends; cells that a backend genuinely
cannot run are marked ``pytest.mark.skip`` with a reason (so they show as ``s`` in the summary
and never spawn a subprocess) rather than being silently omitted. The statically-unsupported
MuJoCo cells are the multi-env-only scenarios on the single-env ClassicBackend, plus the Isaac-only
``damping-decay`` (no static MuJoCo freejoint-damping config; see behavior_assert.SCENARIOS).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "behavior_assert.py"

# The full behavioral catalog. ``multi_env`` flags scenarios that need >1 env (per-env
# independence / indexing) — unsupported on the single-env ClassicBackend.
_SCENARIOS = [
    ("collide-into-fixed", False),
    ("momentum-transfer", False),
    ("static-support", False),
    ("friction-slide", False),
    ("dr-friction-governs", False),
    ("dr-damping-governs", False),
    ("galileo-freefall", False),
    ("damping-decay", False),
    ("restitution-bounce", False),
    ("angular-velocity-spin", False),
    ("velocity-restore-reset", False),
    ("loader-invariance", False),
    ("loader-invariance-file", False),
    ("multibody-independence", False),
    ("object-obs-robot-pose", True),
    ("pose-jitter-settle", True),
    ("per-env-relocation", True),
    ("actor-set-states-robot-object", True),
    # link_physics robot/object parity twins (one-link robot vs free object twin). Drop-from-rest,
    # single-env, run on the MuJoCo backends too.
    ("galileo-twin", False),
    ("restitution-twin", False),
    # Isaac-only physx/combine-mode scenarios; see _ISAAC_ONLY below.
    ("damped-fall-twin", False),
    ("combine-mode-collision", False),
    ("robot-combine-mode", False),
]

# Scenarios that statically cannot run on either MuJoCo backend (Isaac-only).
# - ``damping-decay`` / ``damped-fall-twin``: driven by the PhysX (``physx``) sub-config, which MuJoCo
#   ignores (it reads only its own ``mujoco`` sub-config).
# - ``combine-mode-collision`` / ``robot-combine-mode``: PhysX friction-combine-mode semantics; MuJoCo
#   combines geom friction differently and has no per-shape combine mode.
_ISAAC_ONLY = {"damping-decay", "damped-fall-twin", "combine-mode-collision", "robot-combine-mode"}
_ISAAC_ONLY_SKIP = pytest.mark.skip(reason="Isaac-only scenario (PhysX-specific; no MuJoCo equivalent)")

# Classic CPU is single-env: statically skip the multi-env-only scenarios and the Isaac-only ones
# (shown as ``s`` with a reason, never spawned). The rest run at num_envs=1.
_CLASSIC_PARAMS = [
    pytest.param(name, marks=_ISAAC_ONLY_SKIP)
    if name in _ISAAC_ONLY
    else pytest.param(
        name,
        marks=pytest.mark.skip(reason="multi-env scenario; MuJoCo ClassicBackend is single-env (covered on mjwarp)"),
    )
    if multi_env
    else name
    for name, multi_env in _SCENARIOS
]

# Warp runs the entire catalog at num_envs=4, minus the Isaac-only scenarios.
_WARP_PARAMS = [pytest.param(name, marks=_ISAAC_ONLY_SKIP) if name in _ISAAC_ONLY else name for name, _ in _SCENARIOS]


@pytest.mark.mujoco_classic
@pytest.mark.parametrize("scenario", _CLASSIC_PARAMS)
def test_behavior_classic_cpu(scenario):
    run_harness(
        _HARNESS,
        "--scenario",
        scenario,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        label=f"mujoco-classic/{scenario}",
        timeout=400,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="WarpBackend behavioral tests require CUDA")
@pytest.mark.parametrize("scenario", _WARP_PARAMS)
def test_behavior_warp_multi_env(scenario):
    run_harness(
        _HARNESS,
        "--scenario",
        scenario,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        label=f"mjwarp/{scenario} (num_envs=4)",
        timeout=400,
    )
