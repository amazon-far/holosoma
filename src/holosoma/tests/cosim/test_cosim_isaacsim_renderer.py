"""An Isaac Sim renderer fills an MJWarp head's remote camera over the co-sim bridge.

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


def test_isaacsim_renderer_fills_the_head_remote_camera() -> None:
    sent, rendered = run_pair(2, "--loopback", renderer="isaacsim")

    assert sent["frames"] == rendered["frames"] > 1 and rendered["linked"]
    assert rendered["exports_in_place"] and sent["imports_in_place"]
    torch.testing.assert_close(rendered["robot_root_states"], sent["robot_root_states"], atol=1e-4, rtol=0)
    torch.testing.assert_close(rendered["dof_pos"], sent["dof_pos"], atol=1e-4, rtol=0)
    # Isaac Sim's own kinematics poses every body where the head's does.
    assert rendered["body_names"] == sent["body_names"]
    torch.testing.assert_close(rendered["body_pos"], sent["body_pos"], atol=1e-4, rtol=0)
    assert bool(((rendered["body_rot"] * sent["body_rot"]).sum(-1).abs() > 1 - 1e-6).all())
    for output in ("loopback/rgb", "loopback/depth"):
        assert torch.equal(sent[output], rendered[output])
    # From the first frame on, Isaac's remote view matches MJWarp's local one at the same pose.
    first = sent["first"]
    gap = (first["loopback/depth"] - first["overview/depth"]).abs()
    assert (gap[gap.isfinite()] > 0.05).float().mean() < 0.002
    assert rendered["loopback/rgb"].float().std() > 0
    moved, still = rendered["loopback/depth"], rendered["still/depth"]
    finite = moved.isfinite() & still.isfinite()
    assert (moved - still)[finite].abs().mean() > 0.005  # metres; the head moved the mount 0.2 m
