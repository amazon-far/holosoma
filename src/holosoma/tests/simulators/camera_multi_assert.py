"""Cross-backend multi-camera attribution assertion harness.

Two cameras on the robot base see different content: ``front_cam`` frames the red panel ahead,
``down_cam`` looks straight down at the ground. Asserts ``get_camera_data(name)`` returns each
camera's own view. Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

SKIP_EXIT_CODE = 77


def _red_fraction(env_rgb) -> float:
    """Fraction of red-panel pixels in one env frame."""
    r, g, b = env_rgb[..., 0].to(int), env_rgb[..., 1].to(int), env_rgb[..., 2].to(int)
    mask = (r > g + 40) & (r > b + 40)
    return float(mask.float().mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    headless = args.headless == "true"
    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.dual_cam)
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)

    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config, device=device, training=dataclasses.replace(config.training, num_envs=args.num_envs)
    )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(headless)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    n = args.num_envs
    env_origins = torch.zeros(n, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    names = sim.get_sensor_names()
    if set(names) != {"front_cam", "down_cam"}:
        print(f"[{args.simulator}] FAIL: expected front_cam+down_cam, got {names}")
        return 1

    # Pin the robot upright so front_cam frames the panel ahead.
    states = sim.get_actor_states(["robot"], torch.arange(n, device=device)).clone()
    states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
    states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], torch.arange(n, device=device), states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.render_sensors()

    fails: list[str] = []
    for e in range(n):
        front_red = _red_fraction(sim.get_camera_data("front_cam", "rgb")[e])
        down_red = _red_fraction(sim.get_camera_data("down_cam", "rgb")[e])
        print(f"[{args.simulator}] env{e}: front_cam red={front_red:.3f}, down_cam red={down_red:.3f}")
        # front_cam should see the panel (substantial red); down_cam should not.
        if front_red < 0.05:
            fails.append(f"{args.simulator}/env{e}: front_cam sees little red ({front_red:.3f}); not framing the panel")
        if down_red > front_red * 0.5:
            fails.append(
                f"{args.simulator}/env{e}: down_cam red {down_red:.3f} not clearly less than front {front_red:.3f} "
                f"(camera buffers may be swapped/mixed)"
            )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: per-camera attribution correct across {n} env(s)")
    return 0


if __name__ == "__main__":
    # IsaacSim teardown deadlocks in carbOnPluginShutdown tearing down the
    # omni.syntheticdata/OmniGraph render-product graph a TiledCamera creates (native
    # py-spy stack), so a normal interpreter exit hangs until the parent's subprocess
    # timeout SIGKILLs it -- turning a PASS (verdict already written to --result-file) into
    # a spurious timeout failure. Hard-exit past the atexit teardown, mirroring
    # behavior_assert / scene_spawn_assert. Rendering itself is fine; only exit hangs.
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
