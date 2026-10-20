"""Live IsaacGym fixed-base env checks (GPU), one sim per subprocess.

Runs the fixed_base_env_assert.py harness: the ``test_arm_fixed`` robot (``fix_base_link=True``) goes through
the manager env (reset_all, zero-action steps, a subset reset) and two PPO iterations. Every reset flushes the
spawn pose, which IsaacGym accepts as a same-pose no-op (only a pose change is refused). Unmarked +
``importorskip("isaacgym")``/CUDA-gated like the other IsaacGym harness wrappers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_env_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_fixed_base_env(tmp_path: Path, num_envs: str) -> None:
    result_file = tmp_path / f"fixed_base_env_isaacgym_{num_envs}.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacgym/fixed-base-env (num_envs={num_envs})",
        timeout=900,
        result_file=result_file,
    )
