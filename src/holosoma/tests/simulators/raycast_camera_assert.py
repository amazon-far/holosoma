"""Isaac Sim ray-cast depth camera assertion harness (``isaacsim.renderer="raycast"``).

Uses the ``raycast_cam`` camera, which casts against the terrain only. ``--check frame`` asserts the
camera produces a depth frame without an RTX ``TiledCamera`` or any RTX sensor. ``--check rerender``
raises the robot in two envs without stepping, re-renders those envs, and asserts their ground depth
grows while the third env's frame is unchanged.

Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step, steps_for_seconds

_NUM_ENVS = 3


def _check_frame(sim):
    import torch

    fails = []
    if sim.tiled_cameras:
        fails.append(f"RTX TiledCameras were built: {sorted(sim.tiled_cameras)}")
    if sim.sim.has_rtx_sensors():
        fails.append("RTX sensors are active with only a ray-cast camera configured")
    depth = sim.get_camera_data("raycast", "depth")
    if tuple(depth.shape) != (_NUM_ENVS, 64, 64, 1) or depth.dtype != torch.float32:
        fails.append(f"depth shape/dtype {tuple(depth.shape)} {depth.dtype} != ({_NUM_ENVS}, 64, 64, 1) float32")
    fails.extend(
        f"env{e}: the camera sees almost no ground"
        for e in range(_NUM_ENVS)
        if float(torch.isfinite(depth[e]).float().mean()) < 0.2
    )
    return fails


def _check_rerender(sim):
    import torch

    fails = []
    before = sim.get_camera_data("raycast", "depth")[..., 0].clone()
    env_ids = torch.arange(_NUM_ENVS, device=sim.sim_device)
    moved = torch.tensor([1, 2], device=sim.sim_device)
    states = sim.get_actor_states(["robot"], env_ids).clone()
    states[moved, 2] += 0.3
    sim.set_actor_states(["robot"], moved, states[moved])
    sim.refresh_sim_tensors()
    sim.rerender_sensors(moved)
    after = sim.get_camera_data("raycast", "depth")[..., 0]

    if not torch.equal(after[0], before[0]):
        fails.append("env0: frame changed although env 0 was not re-rendered")
    for e in (1, 2):
        ground = torch.isfinite(before[e]) & torch.isfinite(after[e])
        if not bool(ground.any()):
            fails.append(f"env{e}: no ground visible before and after the move")
        elif float((after[e][ground] > 1.1 * before[e][ground]).float().mean()) < 0.8:
            fails.append(f"env{e}: ground depth did not grow after raising the robot by 0.3 m")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True, choices=["frame", "rerender"])
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        "isaacsim", "empty", "g1-29dof", "terrain_locomotion_plane", sensors=_camera_presets.raycast_cam
    )
    config = dataclasses.replace(
        config, device="cuda:0", training=dataclasses.replace(config.training, num_envs=_NUM_ENVS)
    )
    env, device, _app = setup_simulation_environment(config, device="cuda:0")
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(_NUM_ENVS, torch.zeros(_NUM_ENVS, 3, device=device), base_init)
    sim.prepare_sim()
    sim.install_plugins()

    # Pin each robot upright at its env origin, then capture one frame from that state.
    env_ids = torch.arange(_NUM_ENVS, device=device)
    robot_states = sim.get_actor_states(["robot"], env_ids).clone()
    robot_states[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], env_ids, robot_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.set_actor_states(["robot"], env_ids, robot_states)
    sim.refresh_sim_tensors()
    sim.render_sensors()

    fails = _check_frame(sim) if args.check == "frame" else _check_rerender(sim)
    for failure in fails:
        print(f"[isaacsim] FAIL: {failure}")
    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        return 1
    print(f"[isaacsim] PASS: ray-cast camera {args.check} checks")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
