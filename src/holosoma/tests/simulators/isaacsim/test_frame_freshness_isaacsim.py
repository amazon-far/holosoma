"""Live Isaac Sim regression: a frame egressed at FRAME_END shows the state written for that frame.

The task control frame closes on the control step's LAST substep, on the premise that the scheduled
render has drawn by then, so a camera consumer sees the state that step wrote and not the previous
step's. The unit tests in ``envs/base_task/tests/test_frame_egress.py`` assert that against a fake
renderer — which cannot notice the backend changing underneath it. This measures the premise on the
real renderer instead, through a real ``SimulationTask``.

Method: the robot carries the camera and faces a red panel. One control step writes the pose that
frames the panel, the next writes one 10 m to the side. If the loop's frame boundary is right, the
second frame's capture has already lost the panel; a one-control-step lag would still show it.

Run the harness half directly with:
    python tests/simulators/isaacsim/test_frame_freshness_isaacsim.py [--result-file /tmp/r.txt]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Shadow-fix: keep this file's dir from hiding the real isaacsim package.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    sys.path.pop(0)

SIDESTEP_M = 10.0
"""How far to move the camera-carrying robot so the panel leaves its field of view entirely."""

VISIBLE_FRACTION = 0.02
"""Red coverage that counts as 'panel in frame'; the geometry harness uses the same floor."""


def _red_fraction(image) -> float:
    """Fraction of pixels where red clearly dominates — the panel's silhouette."""
    red, green, blue = (image[..., channel].to(int) for channel in range(3))
    mask = (red > green + 40) & (red > blue + 40)
    return float(mask.sum()) / float(mask.numel())


def _config(device: str):
    """A one-env headless, unpaced IsaacSim task on the panel-target scene with the forward camera."""
    from tests.simulators import _camera_presets
    from tests.simulators._sim_harness import build_run_sim_config

    config = build_run_sim_config(
        "isaacsim", "panel-target", "g1-29dof", "terrain_locomotion_plane", sensors=_camera_presets.front_cam
    )
    return dataclasses.replace(
        config,
        device=device,
        time_scale=0.0,
        training=dataclasses.replace(config.training, num_envs=1, headless=True),
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    config = _config("cuda:0")
    with config.create_task() as task:
        return _check_frames(task, config, check, failures, args.result_file)


def _check_frames(task, config, check, failures: list[str], result_file: str | None) -> int:
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    sim, device = task.simulator, task.device
    camera = next(iter(config.sensor))
    env_ids = torch.arange(1, device=device)
    init = config.robot.init_state
    home = sim.env_origins + torch.tensor(list(init.pos), device=device)
    spawn_dof = sim.dof_pos.clone()

    def write_pose(offset_y: float) -> None:
        """Pin the robot (and so its camera) at the home pose shifted sideways, joints held rigid.

        This is a PRE_STEP write, where a playback plugin puts its state, so the frame under way is
        the one that must carry it.
        """
        states = sim.get_actor_states(["robot"], env_ids).clone()
        states[:, :3] = home + torch.tensor([0.0, offset_y, 0.0], device=device)
        states[:, 3:7] = torch.tensor(list(init.rot), device=device)
        states[:, 7:] = 0.0
        sim.set_actor_states(["robot"], env_ids, states)
        dof_state = torch.zeros(1, sim.num_dof, 2, device=device)
        dof_state[:, :, 0] = spawn_dof
        sim.set_dof_state_tensor_robots(env_ids, dof_state)

    offset = [0.0]
    task.hooks.add(Phase.PRE_STEP, lambda: write_pose(offset[0]), name="test.write_pose")

    def frame(offset_y: float) -> float:
        """Drive one real task control frame; return the red fraction egressed at its FRAME_END."""
        offset[0] = offset_y
        task.step({})
        return _red_fraction(sim.get_camera_data(camera, "rgb")[0])

    task.reset_all()

    # Settle so the first measured frame is not competing with spawn transients.
    for _ in range(6):
        frame(0.0)

    facing = frame(0.0)
    check("panel-visible-when-framed", facing > VISIBLE_FRACTION, f"red fraction {facing:.4f}")

    # The very next frame writes a pose from which the panel cannot be seen. Its own capture must
    # already reflect that; showing the panel here is the one-control-step lag this loop shape fixes.
    moved = frame(SIDESTEP_M)
    check(
        "moved-frame-shows-its-own-pose",
        moved <= VISIBLE_FRACTION,
        f"red fraction {moved:.4f} (previous frame {facing:.4f})",
    )

    back = frame(0.0)
    check("returning-frame-shows-its-own-pose", back > VISIBLE_FRACTION, f"red fraction {back:.4f}")

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    if result_file:
        Path(result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    from tests.simulators._sim_harness import run_and_hard_exit

    run_and_hard_exit(_main)


import pytest  # noqa: E402  (the harness half above must not depend on pytest)

pytestmark = pytest.mark.isaacsim


def test_frame_carries_the_state_written_for_it(tmp_path):
    """Drive the harness above in its own process and require every check to pass."""
    pytest.importorskip("isaaclab")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("IsaacSim requires a CUDA device")

    from tests.simulators._run_harness import run_harness

    result_file = tmp_path / "frame_freshness_isaacsim.txt"
    run_harness(
        Path(__file__).resolve(),
        "--result-file",
        str(result_file),
        label="isaacsim/frame-freshness",
        timeout=600,
        result_file=result_file,
    )
