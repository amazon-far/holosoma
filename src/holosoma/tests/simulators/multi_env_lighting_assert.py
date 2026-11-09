"""Multi-env lighting equivalence harness: every env must render like env 0 under its own light.

Each env holds the same robot, panel and camera, lit only by one positioned sphere light that the light
rig authors per env (no dome, no sun, so no light crosses envs). Identical envs must then produce the same
image. Prints each env's mean ``front_cam`` luminance and fails when any env differs from env 0 by more
than ``--tolerance`` (fraction). Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.light import SphereLightConfig
from tests.simulators._sim_harness import (
    build_run_sim_config,
    launch_simulator,
    run_and_hard_exit,
    step,
    steps_for_seconds,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--intensity", type=float, default=1.0e7)
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        "isaacsim", "panel-target", "g1-29dof", "terrain_locomotion_plane", sensors=_camera_presets.front_cam
    )
    lamp = SphereLightConfig(radius=0.3, position=(1.0, 0.0, 2.0), intensity=args.intensity)
    config = dataclasses.replace(
        config,
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=args.num_envs, headless=True),
        scene=dataclasses.replace(config.scene, lights={"lamp": lamp}),
    )

    sim, device, _app = launch_simulator(config, device="cuda:0")
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    n = args.num_envs
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, torch.zeros(n, 3, device=device), base_init)
    sim.prepare_sim()
    sim.install_plugins()

    # Pin every robot to the same env-local pose so each camera frames the same view.
    origins = sim.scene.env_origins
    states = sim.get_actor_states(["robot"], torch.arange(n, device=device)).clone()
    states[:, :3] = origins + torch.tensor(list(init.pos), device=device)
    states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], torch.arange(n, device=device), states)
    step(sim, max(10, steps_for_seconds(sim, 0.2)), render=True)  # let the denoiser and light cache settle
    sim.render_sensors()

    rgb = sim.get_camera_data("front_cam", "rgb").float()  # [n, H, W, 3]
    luminance = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean(dim=(1, 2)).tolist()
    print("[isaacsim] mean front_cam luminance per env: " + ", ".join(f"{v:.2f}" for v in luminance))
    fails = []
    if luminance[0] < 5.0:
        fails.append(f"env0 is nearly black ({luminance[0]:.2f}); the light does not reach the view")
    for e in range(1, n):
        ratio = luminance[e] / max(luminance[0], 1e-6)
        if abs(ratio - 1.0) > args.tolerance:
            fails.append(f"env{e} luminance {luminance[e]:.2f} is {ratio:.3f}x env0 ({luminance[0]:.2f})")

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    for f in fails:
        print(f"[isaacsim] FAIL: {f}")
    if fails:
        return 1
    print(f"[isaacsim] PASS: {n} envs render alike within {args.tolerance:.0%}")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
