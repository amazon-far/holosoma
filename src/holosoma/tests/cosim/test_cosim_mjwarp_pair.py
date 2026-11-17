"""An MJWarp render task renders exactly what an MJWarp source publishes over the co-sim bridge."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("zmq")

pytestmark = pytest.mark.mujoco_warp
if not torch.cuda.is_available():
    pytest.skip("Co-simulation shares CUDA memory between processes", allow_module_level=True)

from tests.cosim.cosim_pair_roles import run_pair  # noqa: E402


@pytest.mark.parametrize("num_envs", [1, 2])
def test_mjwarp_receiver_renders_the_published_state(num_envs: int) -> None:
    sent, rendered = run_pair(num_envs)

    assert sent["frames"] == rendered["frames"] > 1
    torch.testing.assert_close(rendered["robot_root_states"], sent["robot_root_states"])
    torch.testing.assert_close(rendered["dof_pos"], sent["dof_pos"])
    assert rendered["overview/rgb"].shape == sent["overview/rgb"].shape
    assert torch.equal(rendered["overview/rgb"], sent["overview/rgb"])


def test_receiver_images_become_the_source_remote_camera_data() -> None:
    sent, rendered = run_pair(2, "--loopback")
    outputs = ("loopback/rgb", "loopback/depth")

    assert sent["frames"] == rendered["frames"] > 1
    for output in outputs:
        assert torch.equal(sent[output], rendered[output])
        assert torch.equal(sent[output], sent[output.replace("loopback", "overview")])
