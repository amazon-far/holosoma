"""Live IsaacGym ``contact_forces_substep`` contract (GPU), one sim per subprocess.

Runs the contact_substep_assert.py harness in its own process (IsaacGym segfaults on a second
gymapi sim per process); that harness documents the properties it asserts.

DISTINCT is this backend's regression guard: ``contact_forces`` wraps the gym net-contact tensor,
so without a per-substep ``refresh_net_contact_force_tensor`` every substep records the same stale
frame.

Unmarked + ``importorskip("isaacgym")``/CUDA-gated so the IsaacGym CI job (``-m "not isaacsim"``)
collects it and it skips cleanly elsewhere. The IsaacSim analogue is in
../isaacsim/test_contact_substep_isaacsim.py; the MuJoCo ones are in ../mujoco/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "contact_substep_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_contact_substep(num_envs, tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacgym/contact-substep (num_envs={num_envs})",
        timeout=900,
        result_file=result_file,
    )
