"""An Isaac Sim renderer fills an MJWarp source's remote camera over the co-sim bridge.

Needs both simulator environments, as in the combined image.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("zmq")

pytestmark = pytest.mark.multi_env
if not torch.cuda.is_available():
    pytest.skip("Co-simulation shares CUDA memory between processes", allow_module_level=True)

from tests.cosim.cosim_pair_roles import run_pair  # noqa: E402


def test_isaacsim_renderer_fills_the_source_remote_camera() -> None:
    sent, rendered = run_pair(2, "--loopback", renderer="isaacsim")

    assert sent["frames"] == rendered["frames"] > 1
    torch.testing.assert_close(rendered["robot_root_states"], sent["robot_root_states"], atol=1e-4, rtol=0)
    torch.testing.assert_close(rendered["dof_pos"], sent["dof_pos"], atol=1e-4, rtol=0)
    for output in ("loopback/rgb", "loopback/depth"):
        assert torch.equal(sent[output], rendered[output])
    assert rendered["loopback/rgb"].float().std() > 0
