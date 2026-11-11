# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Verify changed sensor mounts through a full task reset and Isaac's native context reset."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit
from tests.simulators.sensor_mount_pose_assert import _check, _sensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    config = build_run_sim_config(
        "isaacsim", "panel-target", "g1-29dof", "terrain_locomotion_plane", sensors=_sensors()
    )
    config = dataclasses.replace(
        config,
        device="cuda:0",
        time_scale=0.0,
        training=dataclasses.replace(config.training, num_envs=args.num_envs, headless=True),
    )
    with config.create_task() as task:
        import torch

        from holosoma.simulator.isaacsim.isaacsim import IsaacSim
        from holosoma.simulator.isaacsim.sensor_setup import sync_camera_fabric_transforms

        task.reset_envs_idx()
        sim = task.simulator
        assert isinstance(sim, IsaacSim)
        manager = sim.sensor_manager
        offsets = {name: manager.get_mount_pose(name) for name in sim.get_sensor_names()}
        angle = math.radians(30.0)
        for name, pose in offsets.items():
            pose[0, 2] += 0.4
            pose[-1, 3:] = torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)], device=task.device)
            manager.set_mount_pose(name, pose)
        sim.render_sensors()
        _check(sim, offsets, "before episode reset")

        episodes = task.episode_count.clone()
        task.reset_envs_idx()
        torch.testing.assert_close(task.episode_count, episodes + 1)
        sim.sim.reset()
        sync_camera_fabric_transforms(sim)
        sim.forward_kinematics()
        # No pose writes after either reset: inspect what the real renderer and ray caster retained.
        sim.render_sensors()
        _check(sim, offsets, "after episode and native context reset")
        assert sim.sensor_manager is manager
        Path(args.result_file).write_text("OK")
        print(f"PASS: sensor mounts survive IsaacSim task/context resets, {args.num_envs} env(s)")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
