"""Isaac Sim ray-cast depth camera assertion harness (``isaacsim.renderer="raycast"``).

Uses the ``panel-target`` scene and the ``raycast_rig`` cameras. ``--check frame`` asserts:

  1. A ray-cast camera matches a co-located RTX depth camera: same depth on pixels both hit
     (median < 1 cm) and no-return where RTX has no return.
  2. A world-mounted pair framing the robot agrees the same way, so tracked robot meshes sit at
     their rendered poses, while a forward LiDAR sharing those targets still measures the panel.
  3. The default ``mount`` body filter omits the head shell a head-mounted camera sits inside;
     ``target_kind="none"`` sees it.

``--check rerender`` moves the robot in two envs without stepping, re-renders those envs, and
asserts their depth follows the move while the third env's frame is unchanged.

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


def _agreement(raycast, rtx, label, *, exact):
    """Compare two ``[H, W]`` depth frames; return failure strings.

    ``exact`` selects pixels both cameras must place within 1 cm. Elsewhere RTX depth of grazing
    ground is noisier than the ray-mesh distance, so the frame is compared in relative terms.
    """
    import torch

    both = torch.isfinite(raycast) & torch.isfinite(rtx)
    if int(both.sum()) < 0.2 * rtx.numel():
        return [f"{label}: only {int(both.sum())}/{rtx.numel()} pixels hit in both cameras"]
    fails = []
    exact = exact & both
    if not bool(exact.any()):
        fails.append(f"{label}: no pixels to compare exactly")
    elif float((raycast[exact] - rtx[exact]).abs().median()) > 0.01:
        fails.append(
            f"{label}: median |raycast - rtx| {float((raycast[exact] - rtx[exact]).abs().median()):.4f}m > 1 cm"
        )
    relative = (raycast[both] - rtx[both]).abs() / rtx[both]
    if float(relative.median()) > 0.02 or float((relative < 0.05).float().mean()) < 0.9:
        fails.append(
            f"{label}: relative depth difference median {float(relative.median()):.3%}, "
            f"{float((relative < 0.05).float().mean()):.1%} within 5%"
        )
    rtx_miss = torch.isinf(rtx)
    if bool(rtx_miss.any()) and float(torch.isinf(raycast[rtx_miss]).float().mean()) < 0.95:
        fails.append(f"{label}: ray-cast hits where RTX has no return")
    return fails


def _check_frame(sim):
    import torch

    from tests.simulators import _camera_presets

    fails = []
    depth = {
        name: sim.get_camera_data(name, "depth")[..., 0] for name in _camera_presets.raycast_rig if "lidar" not in name
    }
    rgb = sim.get_camera_data("rtx", "rgb")
    side = depth["side_rtx"]
    # The robot stands 1.5 m from the side camera; ground that close is only in the bottom rows.
    side_robot = (side < 1.75) & (torch.arange(side.shape[1], device=side.device) < 0.85 * side.shape[1])[:, None]
    for e in range(_NUM_ENVS):
        panel = _red_mask(rgb[e])
        if not bool(panel.any()):
            fails.append(f"rtx/env{e}: panel not visible")
        fails += _agreement(depth["raycast"][e], depth["rtx"][e], f"forward/env{e}", exact=panel)
        if float(side_robot[e].float().mean()) < 0.05:
            fails.append(f"side_rtx/env{e}: robot not visible")
        fails += _agreement(depth["side_raycast"][e], side[e], f"side/env{e}", exact=side_robot[e])

        head_mount = depth["head_mount"][e]
        head_none = depth["head_none"][e]
        if float((head_none < 0.15).float().mean()) < 0.5:
            fails.append(f"head_none/env{e}: does not see the head shell around the camera")
        if bool((head_mount < 0.15).any()):
            fails.append(f"head_mount/env{e}: sees geometry within 0.15 m despite the mount body filter")

    lidar_range = sim.get_lidar_data("lidar", "ranges")[:, 0]
    expected = _camera_presets._PANEL_DISTANCE - 0.1 - 0.01
    if not bool(((lidar_range - expected).abs() < 0.02).all()):
        fails.append(f"lidar: ranges {lidar_range.tolist()} != panel distance {expected:.3f}m")
    return fails


def _check_rerender(sim):
    import torch

    fails = []
    before = sim.get_camera_data("raycast", "depth")[..., 0].clone()
    env_ids = torch.arange(_NUM_ENVS, device=sim.sim_device)
    states = sim.get_actor_states(["robot"], env_ids).clone()
    shifts = torch.tensor([0.0, -0.2, -0.3], device=sim.sim_device)
    states[:, 0] += shifts
    moved = torch.tensor([1, 2], device=sim.sim_device)
    sim.set_actor_states(["robot"], moved, states[moved])
    sim.rerender_sensors(moved)
    after = sim.get_camera_data("raycast", "depth")[..., 0]

    if not torch.equal(after[0], before[0]):
        fails.append("raycast/env0: frame changed although env 0 was not re-rendered")
    rgb = sim.get_camera_data("rtx", "rgb")
    for e in (1, 2):
        mask = _red_mask(rgb[e]) & torch.isfinite(before[e]) & torch.isfinite(after[e])
        if not bool(mask.any()):
            fails.append(f"raycast/env{e}: panel not visible before and after the move")
            continue
        grew = float((after[e][mask] - before[e][mask]).median())
        if abs(grew + float(shifts[e])) > 0.01:
            fails.append(f"raycast/env{e}: panel depth changed by {grew:.3f}m, expected {-float(shifts[e]):.3f}m")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True, choices=["frame", "rerender"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        "isaacsim", "panel-target", args.robot, args.terrain, sensors=_camera_presets.raycast_rig
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

    env_origins = torch.zeros(_NUM_ENVS, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(_NUM_ENVS, env_origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()

    # Pin each robot upright at its env origin so the forward cameras face that env's panel.
    env_ids = torch.arange(_NUM_ENVS, device=device)
    robot_states = sim.get_actor_states(["robot"], env_ids).clone()
    robot_states[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], env_ids, robot_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    # Re-pin and render once so the RTX frame and the ray-cast poses come from the same state.
    sim.set_actor_states(["robot"], env_ids, robot_states)
    sim.sim.forward()
    sim.render()
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
