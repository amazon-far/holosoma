"""A real projectile changes contact partners in two Isaac Sim environments."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "contact_rich_assert.py"


def test_contact_rich_isaacsim(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "2",
        "--result-file",
        str(result),
        label="isaacsim/contact-rich",
        timeout=900,
        result_file=result,
    )
