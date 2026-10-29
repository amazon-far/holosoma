"""Isaac Sim ray-cast depth camera assertion harness (``isaacsim.renderer="raycast"``).

Uses the ``panel-target`` scene and the ``raycast_pair`` cameras: an RTX depth camera and a ray-cast
one at the same mount. The ray-cast camera sees only the terrain and its ``raycast_link_meshes``,
so the panel is excluded from comparisons. ``--check frame`` asserts the ray-cast camera is not an
RTX ``TiledCamera`` and agrees with the RTX camera's ground depth. ``--check rerender`` raises the
robot in two envs without stepping, re-renders those envs, and asserts their ground depth grows
while the third env's frame is unchanged.

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
from tests.simulators.depth_assert import _red_mask

_NUM_ENVS = 3


def _ground(sim, env_id):
    """``[H, W]`` mask of pixels where the RTX camera sees finite ground, not the panel."""
    import torch

    rtx = sim.get_camera_data("rtx", "depth")[env_id, ..., 0]
    return torch.isfinite(rtx) & ~_red_mask(sim.get_camera_data("rtx", "rgb")[env_id])


def _check_frame(sim):
    import torch

    fails = []
    if "raycast" in sim.tiled_cameras:
        fails.append("raycast: built as an RTX TiledCamera")
    rtx = sim.get_camera_data("rtx", "depth")[..., 0]
    raycast = sim.get_camera_data("raycast", "depth")[..., 0]
    for e in range(_NUM_ENVS):
        ground = _ground(sim, e) & torch.isfinite(raycast[e])
        if float(ground.float().mean()) < 0.2:
            fails.append(f"env{e}: only {float(ground.float().mean()):.1%} of pixels are ground in both cameras")
            continue
        # The renderers agree to a few percent on ground; this catches a wrong pose or projection.
        relative = (raycast[e][ground] - rtx[e][ground]).abs() / rtx[e][ground]
        if float(relative.median()) > 0.05 or float((relative < 0.1).float().mean()) < 0.9:
            fails.append(
                f"env{e}: ground depth differs from RTX: median {float(relative.median()):.3%}, "
                f"{float((relative < 0.1).float().mean()):.1%} within 10%"
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
        ground = _ground(sim, e) & torch.isfinite(before[e]) & torch.isfinite(after[e])
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
        "isaacsim", "panel-target", "g1-29dof", "terrain_locomotion_plane", sensors=_camera_presets.raycast_pair
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

    # Pin each robot upright at its env origin, then render once from that state so the RTX frame
    # and the ray-cast poses match.
    env_ids = torch.arange(_NUM_ENVS, device=device)
    robot_states = sim.get_actor_states(["robot"], env_ids).clone()
    robot_states[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], env_ids, robot_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.set_actor_states(["robot"], env_ids, robot_states)
    sim.sim.forward()
    sim.render()
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
