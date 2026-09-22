"""Live MuJoCo ``contact_forces_substep`` contract via the shared harness.

Routes both MuJoCo backends through the SAME contact_substep_assert.py harness the Isaac backends
use, so the buffer's contract is proven identically on all four; that harness documents the
properties it asserts. ClassicBackend runs on CPU — the only backend where this contract is
checkable without a GPU; the Warp path is CUDA-gated and multi-env.

STABLE and ORDERED are this backend's regression guards: the rotation used to live in
``refresh_sim_tensors``, which runs a variable number of times per control step (the reset path
calls it a second time, duplicating a frame for every env whenever any env reset).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "contact_substep_assert.py"


@pytest.mark.mujoco_classic
def test_contact_substep_classic(tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--result-file",
        str(result_file),
        label="mujoco/contact-substep (classic)",
        timeout=600,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp multi-env requires a CUDA device")
@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_contact_substep_warp(num_envs, tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"mjwarp/contact-substep (num_envs={num_envs})",
        timeout=600,
        result_file=result_file,
    )
