"""Live MuJoCo Warp (GPU, multi-env) non-contiguous env_ids subset write via the shared harness.

Routes the Warp backend through subset_write_assert.py: relocate a static pillar in ONLY envs
[0, 2] of 4 to per-env-distinct world targets, assert written envs land at their own target and
unwritten envs [1, 3] are untouched. Exercises the strided-subset index encode/decode
(get_object_indices / resolve_indices / write_object_states) under set_static_body_pose on the
WarpBackend's per-world field-expansion path, which the contiguous full-range writes in
test_static_move_warp.py and behavior per-env-relocation never reach.

MuJoCo ClassicBackend (CPU) cannot do >1 env, so this strided multi-env path is mjwarp-only.
CUDA-gated; one sim per subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp multi-env subset-write requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "subset_write_assert.py"


def test_subset_write():
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        label="mjwarp/subset-write (num_envs=4)",
        timeout=600,
    )
