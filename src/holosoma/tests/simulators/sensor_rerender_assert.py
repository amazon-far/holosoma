"""Cross-backend ``sim.num_rerenders_on_reset`` assertion harness.

A downward depth camera and a downward LiDAR ride the robot base. After one capture, the robot is
raised 0.3 m in some envs and ``EPISODE_START`` is emitted for them, which re-renders their sensors:

  1. With ``num_rerenders_on_reset=0`` both sensors keep their frame.
  2. With ``num_rerenders_on_reset=2`` the raised envs see the ground farther away, every other env
     keeps its frame exactly, and the LiDAR is cast once.

Exits 0 PASS / 1 FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from unittest import mock

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.sensor import (
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.hooks import Phase
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit, step

_RAISE = 0.3
# Ahead of the feet, so both sensors see open ground below the robot base.
_SENSORS = {
    "down_cam": CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis", position=[0.6, 0.0, 0.0]),
        width=32,
        height=32,
        vertical_fov=40.0,
        data_types=["depth"],
    ),
    "down_lidar": LidarSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis", position=[0.6, 0.0, 0.0]),
        pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [0.0, 0.2, -1.0]]),
        near=0.05,
        far=10.0,
    ),
}


def _with_rerenders(config, count: int):
    sim_engine = dataclasses.replace(config.sim, num_rerenders_on_reset=count)
    return dataclasses.replace(config, sim=sim_engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(sim_arg, "empty", args.robot, args.terrain, sensors=_SENSORS)
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)
    n = 1 if args.simulator == "mujoco" else 3
    device = "cpu" if args.simulator == "mujoco" else "cuda:0"
    config = dataclasses.replace(
        config, device=device, training=dataclasses.replace(config.training, num_envs=n, headless=True)
    )

    sim, device, _app = launch_simulator(config, device=device)
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    env_ids = torch.arange(n, device=device)
    env_origins = torch.zeros(n, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(init.root_state_list(), device=device, dtype=torch.float32)
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()
    step(sim, 2)
    sim.render_sensors()

    if args.simulator == "isaacsim":
        from holosoma.simulator.isaacsim import sensor_setup
    else:
        from holosoma.simulator.mujoco import sensor_setup  # type: ignore[no-redef]
    finalize = sensor_setup.finalize_lidar_returns
    with mock.patch.object(sensor_setup, "finalize_lidar_returns", wraps=finalize) as casts:

        def _frames():
            return (
                sim.get_camera_data("down_cam", "depth")[..., 0].clone(),
                sim.get_lidar_data("down_lidar", "ranges").clone(),
            )

        depth0, ranges0 = _frames()
        moved = torch.tensor([0] if n == 1 else [1, 2], device=device)
        states = sim.get_actor_states(["robot"], moved).clone()
        states[:, 2] += _RAISE
        sim.set_actor_states(["robot"], moved, states)
        sim.forward_kinematics()

        fails: list[str] = []
        sim.simulator_config = _with_rerenders(sim.simulator_config, 0)
        sim.hooks.emit(Phase.EPISODE_START, moved)
        depth, ranges = _frames()
        if not torch.equal(depth, depth0) or not torch.equal(ranges, ranges0):
            fails.append("num_rerenders_on_reset=0 changed a sensor frame")
        if casts.call_count:
            fails.append(f"num_rerenders_on_reset=0 cast the LiDAR {casts.call_count} time(s)")

        sim.simulator_config = _with_rerenders(sim.simulator_config, 2)
        sim.hooks.emit(Phase.EPISODE_START, moved)
        depth, ranges = _frames()
        if casts.call_count != 1:
            fails.append(f"num_rerenders_on_reset=2 cast the LiDAR {casts.call_count} times, want 1")
        for env in moved.tolist():
            visible = torch.isfinite(depth0[env]) & torch.isfinite(depth[env])
            growth = float((depth[env] - depth0[env])[visible].median()) if bool(visible.any()) else float("nan")
            if not abs(growth - _RAISE) < 0.1:
                fails.append(f"env{env}: camera ground depth grew {growth:.3f} m, want ~{_RAISE}")
            lidar_growth = float((ranges[env] - ranges0[env]).min())
            if not lidar_growth > 0.5 * _RAISE:
                fails.append(f"env{env}: LiDAR ranges grew at least {lidar_growth:.3f} m, want ~{_RAISE}")
        kept = torch.tensor([e for e in env_ids.tolist() if e not in moved.tolist()], device=device, dtype=torch.long)
        if not torch.equal(depth[kept], depth0[kept]) or not torch.equal(ranges[kept], ranges0[kept]):
            fails.append("re-render changed envs outside the requested subset")

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    for fail in fails:
        print(f"[{args.simulator}] FAIL: {fail}")
    if fails:
        return 1
    print(f"[{args.simulator}] PASS: camera and LiDAR re-render only the reset envs, LiDAR cast once")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
