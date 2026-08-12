"""Cross-backend actor-mount camera assertion harness.

Exercises the ``target_kind="actor"`` mount resolver (a camera mounted on a spawned scene
object rather than a robot link). Mounts a camera on the spawned ``panel`` object and asserts
the frame has the expected shape/dtype and is non-blank. Then moves the panel and asserts the
rendered frame changes, confirming the camera follows the actor. Exits 0 PASS, 1 FAIL, 77 SKIP.
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
    config = build_run_sim_config(sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.actor_cam)
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
    if names != ["panel_cam"]:
        print(f"[{args.simulator}] FAIL: actor-mount camera not created (got {names})")
        return 1

    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.render_sensors()

    fails: list[str] = []
    img = sim.get_camera_data("panel_cam", "rgb")  # [N,H,W,3] uint8
    cam = next(iter(config.sensor.values()))
    if tuple(img.shape) != (n, cam.height, cam.width, 3) or img.dtype != torch.uint8:
        fails.append(f"{args.simulator}: actor-mount frame shape/dtype {tuple(img.shape)} {img.dtype}")
    for e in range(n):
        lum = img[e].float().mean(dim=-1)
        if float(lum.max()) - float(lum.min()) < 1.0:
            fails.append(f"{args.simulator}/env{e}: actor-mount frame ~uniform; renderer saw nothing")

    # Follow: move the panel and confirm the panel-mounted camera's frame changes.
    before = img.clone()
    panel_states = sim.get_actor_states(["panel"], torch.arange(n, device=device)).clone()
    panel_states[:, 2] += 0.5  # lift the panel 0.5 m
    sim.set_actor_states(["panel"], torch.arange(n, device=device), panel_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.render_sensors()
    after = sim.get_camera_data("panel_cam", "rgb")
    if torch.equal(before, after):
        fails.append(f"{args.simulator}: panel-mounted camera frame unchanged after moving the panel (not following)")

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: actor-mounted camera resolves, renders, and follows across {n} env(s)")
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
