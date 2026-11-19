"""An MJWarp render task renders exactly what an MJWarp head publishes over the co-sim bridge."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("zmq")

pytestmark = pytest.mark.mujoco_warp
if not torch.cuda.is_available():
    pytest.skip("Co-simulation shares CUDA memory between processes", allow_module_level=True)

from tests.cosim.cosim_pair_roles import run_pair  # noqa: E402


def test_peer_images_become_the_head_remote_camera_data() -> None:
    sent, rendered = run_pair(2, "--loopback")
    outputs = ("loopback/rgb", "loopback/depth")

    assert rendered["frames"] > 1 and rendered["linked"]
    starts = sum(phase == "episode_start" for phase, _ in rendered["episodes"])
    assert sent["exchanges"] == rendered["frames"] + starts  # each renders or starts episodes
    assert rendered["exports_in_place"] and sent["imports_in_place"]
    assert (
        rendered["episodes"]
        == sent["episodes"]
        == [
            ("episode_start", [0, 1]),
            ("episode_end", [1]),
            ("episode_start", [1]),
        ]
    )
    torch.testing.assert_close(rendered["robot_root_states"], sent["robot_root_states"])
    torch.testing.assert_close(rendered["dof_pos"], sent["dof_pos"])
    assert torch.equal(rendered["overview/rgb"], sent["overview/rgb"])  # the same camera rendered on both sides
    for output in outputs:
        assert torch.equal(sent[output], rendered[output])
        assert torch.equal(sent[output], sent[output.replace("loopback", "overview")])
    moved, still = rendered["loopback/depth"], rendered["still/depth"]
    finite = moved.isfinite() & still.isfinite()
    assert (moved - still)[finite].abs().mean() > 0.005  # metres; the head moved the mount 0.2 m
