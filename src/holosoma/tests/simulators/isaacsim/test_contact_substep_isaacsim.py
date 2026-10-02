"""Live IsaacSim ``contact_forces_substep`` contract (GPU), one sim per subprocess.

Runs the contact_substep_assert.py harness in its own process (IsaacSim's SimulationContext is a
process singleton); that harness documents the properties it asserts.

SHAPE and FILLED are this backend's regression guards: the buffer used to be sized from a config
knob independent of the decimation, and only its first ``min(decimation, knob)`` slots were ever
written — so raising the knob past the decimation left a tail that read as real zero-force samples
forever.

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it. The verdict is read
from a result-file sentinel because IsaacSim teardown can corrupt the exit code.
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

_HARNESS = Path(__file__).resolve().parents[1] / "contact_substep_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_contact_substep(num_envs, tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacsim/contact-substep (num_envs={num_envs})",
        timeout=900,
        result_file=result_file,
    )
