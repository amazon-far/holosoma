"""Live IsaacSim fixed-base env checks (GPU), one sim per subprocess.

Runs the fixed_base_env_assert.py harness: the ``test_arm_fixed`` robot (``fix_base_link=True``) goes through
the manager env (reset_all, zero-action steps, a subset reset) and two PPO iterations with the welded base at
its spawn pose throughout. Marked ``isaacsim`` and ``importorskip("isaaclab")``/CUDA-gated like the other
IsaacSim harness wrappers; the verdict is the ``--result-file`` sentinel (IsaacSim teardown can corrupt the
exit code).
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

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_env_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_fixed_base_env(tmp_path: Path, num_envs: str) -> None:
    result_file = tmp_path / f"fixed_base_env_isaacsim_{num_envs}.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacsim/fixed-base-env (num_envs={num_envs})",
        timeout=1200,
        result_file=result_file,
    )
