"""Live IsaacSim check: identical envs under their own per-env light render alike.

Drives ``multi_env_lighting_assert.py`` (4 envs, one per-env sphere light, no dome or sun). On Isaac Sim 5.1
the cloned envs come out darker or brighter than env 0 (measured 0.75x / 1.05x / 0.80x), although every
light and camera pose matches in USD and Fabric: the renderer lights cloned envs differently. Marked a
strict expected failure, so a renderer or holosoma change that fixes it turns this test red until the mark
is removed. Multi-env frames should not be used as dataset frames while it fails.
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

_HARNESS = Path(__file__).resolve().parents[1] / "multi_env_lighting_assert.py"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="cloned envs are lit differently from env 0 (H3)")
def test_identical_envs_render_alike(tmp_path: Path) -> None:
    result_file = tmp_path / "multi_env_lighting.txt"
    run_harness(
        _HARNESS,
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacsim/multi-env-lighting",
        timeout=900,
        result_file=result_file,
    )
