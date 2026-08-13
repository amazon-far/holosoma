"""Behavioral physics tests on the IsaacGym backend via the behavior_assert.py harness.

Each test runs one behavioral scenario in its own subprocess (IsaacGym segfaults on a second
gymapi sim per process) at num_envs=4 and asserts the harness exited 0 — the harness asserts a
physical OUTCOME in EVERY env. IsaacGym loads URDF only; every behavioral preset ships a urdf, so
all scenarios spawn here. Unmarked (conftest's directory rule applies ``isaacgym``; the CI job
selects ``-m isaacgym``) + importorskip("isaacgym")/CUDA-gated so it skips cleanly elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "behavior_assert.py"

# The behavioral catalog runs on IsaacGym at num_envs=4. dr-damping-governs is statically skipped:
# IsaacGym exposes rigid-body damping only at asset import (AssetOptions), with no runtime setter,
# so runtime damping DR is genuinely unsupported here (MuJoCo + IsaacSim cover it).
_SCENARIOS = [
    "collide-into-fixed",
    "momentum-transfer",
    "static-support",
    "friction-slide",
    "dr-friction-governs",
    pytest.param(
        "dr-damping-governs",
        marks=pytest.mark.skip(reason="IsaacGym has no runtime body-damping setter (import-time AssetOptions only)"),
    ),
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
    # link_physics robot/object parity twins, on the minimal one-link `onelink-box` robot. IsaacGym
    # cannot run it: the bare 0-DOF form crashes ("Cannot get mass matrix for actors without DOFs"),
    # and the dummy-joint form the Isaac articulation needs then mismatches the minimal RobotConfig's
    # body/dof counts (IsaacGym asserts num_dof==len(dof_names) and num_bodies==len(body_names) in
    # load_assets). These twins run on MuJoCo (classic + warp) and IsaacSim, where the link_physics
    # material bind applies.
    pytest.param(
        "galileo-twin",
        marks=pytest.mark.skip(reason="IsaacGym: one-link robot not representable (0-DOF crash / dof-count mismatch)"),
    ),
    pytest.param(
        "restitution-twin",
        marks=pytest.mark.skip(reason="IsaacGym: one-link robot not representable (0-DOF crash / dof-count mismatch)"),
    ),
    pytest.param(
        "damped-fall-twin",
        marks=pytest.mark.skip(reason="IsaacGym: one-link robot not representable (0-DOF crash / dof-count mismatch)"),
    ),
    # combine-mode-collision (pure objects) needs a per-pair friction combine-mode field IsaacGym's
    # RigidShapeProperties lacks; robot-combine-mode additionally needs the 0-DOF robot. Both skip.
    pytest.param(
        "combine-mode-collision",
        marks=pytest.mark.skip(reason="IsaacGym has no per-pair friction-combine-mode field"),
    ),
    pytest.param(
        "robot-combine-mode",
        marks=pytest.mark.skip(reason="IsaacGym: 0-DOF robot + no per-pair friction-combine-mode field"),
    ),
]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_behavior_isaacgym(scenario):
    run_harness(
        _HARNESS,
        "--scenario",
        scenario,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        label=f"isaacgym/{scenario} (num_envs=4)",
        timeout=600,
    )
