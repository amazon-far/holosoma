"""Named contact contracts through native Isaac Sim physics without rendering."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "contact_sensor_assert.py"


@pytest.mark.parametrize("legacy_only", [False, True], ids=["configured", "legacy-only"])
def test_named_contacts_isaacsim(tmp_path: Path, legacy_only: bool) -> None:
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "2",
        "--result-file",
        str(result_file),
        *(["--legacy-only"] if legacy_only else []),
        label="isaacsim/named-contacts",
        timeout=900,
        result_file=result_file,
    )
