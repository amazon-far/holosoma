"""Live IsaacSim batched forward kinematics for WBT default-pose transitions (GPU), one sim per subprocess.

Runs wbt_fk_assert.py on a real G1: sweep the joints across 8 frames, write one frame per env, and
check the poses read back are the ones just written -- root pose round-trips, distinct joint
configurations give distinct body poses, and one-frame-per-env agrees with one-frame-at-a-time.

This is the only check that can falsify the assumption the transition rewrite rests on. IsaacLab's
``ArticulationData.body_*_w`` are timestamp-gated lazy caches; if a joint-state write does not
invalidate them, forward kinematics would return the pose from before the write and every transition
frame would carry identical body poses while the joint angles vary. Nothing else reads these buffers
without stepping the sim first, so nothing else would notice.

Marked ``isaacsim`` so the IsaacSim CI job collects it; ``importorskip("isaaclab")``/CUDA-gated.
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

_HARNESS = Path(__file__).resolve().parents[1] / "wbt_fk_assert.py"


def test_wbt_batched_fk_returns_fresh_poses(tmp_path):
    # Judged on the sentinel, not the exit code: IsaacSim teardown swallows a non-zero status, so a
    # crashed harness exits 0 and this would report a pass.
    result_file = tmp_path / "wbt_fk_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "8",
        "--result-file",
        str(result_file),
        label="isaacsim/wbt-batched-fk (num_envs=8)",
        timeout=900,
        result_file=result_file,
    )
