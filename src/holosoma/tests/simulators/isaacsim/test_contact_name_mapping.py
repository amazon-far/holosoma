"""Native contact indexing follows canonical observations and asset selectors."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402


def test_fixed_base_contact_names_and_canonical_order(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    run_harness(
        Path(__file__).resolve().parents[1] / "isaacsim_contact_mapping_assert.py",
        "--result-file",
        str(result),
        label="isaacsim/contact-name-mapping",
        timeout=900,
        result_file=result,
    )
