"""Co-simulation leader harness: camera observations rendered by a running cosim-follower.

Builds a run_sim task with remote sensors and the cosim-leader plugin, connected to a follower
already bound at ``--endpoint``, then checks:

  1. Before any frame the camera buffer exists and is all zeros.
  2. The images are attributed to the follower's backend.
  3. After a few frames the image is a real render (not constant).
  4. Tilting the robot and re-rendering env 0 changes the image without stepping.

Writes ``OK`` to ``--result-file`` once every check passes, then hard-exits.
"""

from __future__ import annotations

import argparse
import math
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.config_registry import parse_config
from tests.simulators._sim_harness import run_and_hard_exit

CAMERA = "head"
FRAMES = 10
TILT_RAD = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--follower-type", required=True, choices=["mujoco", "isaacsim"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # AppLauncher re-parses argv

    import torch

    config = parse_config(
        RunSimConfig,
        args=[
            f"simulator:{args.simulator}",
            "robot:g1-29dof",
            f"sensor.{CAMERA}:g1-head",
            "plugin.cosim:cosim-leader",
            "--plugin.cosim.endpoint",
            args.endpoint,
            "--simulator.config.sim.remote_sensors",
            "True",
            "--simulator.config.bridge.enabled",
            "False",
            "--simulator.config.virtual-gantry.enabled",
            "False",
            "--training.headless",
            "True",
        ],
    )
    with config.create_task() as task:
        sim = task.simulator
        before = sim.get_camera_data(CAMERA, "rgb").clone()
        assert before.abs().sum() == 0, "camera buffer should start zeroed"
        assert sim.sensor_manager is not None
        assert sim.sensor_manager.simulator_type is not None
        assert sim.sensor_manager.simulator_type.value == args.follower_type, sim.sensor_manager.simulator_type

        task.reset_all()
        for _ in range(FRAMES):
            task.step()
        rendered = sim.get_camera_data(CAMERA, "rgb").float()
        assert rendered.std() > 1.0, f"image looks blank (std {rendered.std():.3f})"

        env0 = torch.tensor([0], device=sim.sim_device)
        root = sim.robot_root_states[env0].clone()
        root[:, 3:7] = torch.tensor([0.0, math.sin(TILT_RAD / 2), 0.0, math.cos(TILT_RAD / 2)])
        dofs = torch.stack([sim.dof_pos[env0], torch.zeros_like(sim.dof_pos[env0])], dim=-1)
        sim.set_robot_state(env0, root_states=root, dof_states=dofs)
        sim.rerender_sensors(env0)
        tilted = sim.get_camera_data(CAMERA, "rgb").float()
        change = (tilted[0] - rendered[0]).abs().mean().item()
        assert change > 1.0, f"re-render after a tilt barely changed the image (mean |diff| {change:.3f})"
        print(f"OK: image std {rendered.std():.1f}, re-render change {change:.1f}", flush=True)
        # Inside the session: closing the Isaac Sim app can end the process.
        with open(args.result_file, "w") as f:
            f.write("OK")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
