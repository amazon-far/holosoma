"""Cross-backend camera observation-pipeline assertion harness.

Each step runs render_sensors() then ObservationManager.compute(), the order
BaseTask._post_physics_step uses.

Checks:
  1. Format: an rgb term with a CHW float01 transform yields [N, 3, H, W] float32 in [0,1] in a
     concatenate=False group.
  2. Rate: a camera with update_decimation=3 holds the same frame across a 3-step window while a
     decimation=1 camera in the same group updates every step.

Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

SKIP_EXIT_CODE = 77
_TERMS = "holosoma.managers.observation.terms.cameras"


class _TaskShell:
    """Minimal env exposing what the camera obs terms and ObservationManager read."""

    def __init__(self, sim, device):
        self.simulator = sim
        self.device = device
        self.num_envs = sim.num_envs


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
    # slow-fast-cam: fast_cam (decimation=1) and slow_cam (decimation=3) at the same mount, both rgb.
    config = build_run_sim_config(
        sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.slow_fast_cam
    )
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

    robot_states = sim.get_actor_states(["robot"], torch.arange(n, device=device)).clone()
    robot_states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], torch.arange(n, device=device), robot_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))

    if set(sim.get_sensor_names()) != {"fast_cam", "slow_cam"}:
        print(f"[{args.simulator}] FAIL: expected fast_cam+slow_cam, got {sim.get_sensor_names()}")
        return 1

    _first_cam = next(iter(config.sensor.values()))
    h, w = _first_cam.height, _first_cam.width
    # A dict (concatenate=False) group: fast_cam as CHW float01 rgb, slow_cam as plain rgb.
    rgb_tf = {"layout": "CHW", "scale": "float01"}
    obs_cfg = ObservationManagerCfg(
        groups={
            "image": ObsGroupCfg(
                concatenate=False,
                terms={
                    "fast_rgb": ObsTermCfg(
                        func=f"{_TERMS}:camera_rgb", params={"sensor": "fast_cam", "transform": rgb_tf}
                    ),
                    "slow_rgb": ObsTermCfg(func=f"{_TERMS}:camera_rgb", params={"sensor": "slow_cam"}),
                },
            )
        }
    )
    mgr = ObservationManager(obs_cfg, _TaskShell(sim, device), device)

    fails: list[str] = []

    def _obs():
        """Render the cameras, then compute and return the "image" obs group."""
        sim.render_sensors()
        return mgr.compute()["image"]

    # 1. Format: the transformed rgb term is CHW float32 in [0,1] with the configured H,W.
    first = _obs()
    fast = first["fast_rgb"]
    if tuple(fast.shape) != (n, 3, h, w) or fast.dtype != torch.float32:
        fails.append(f"{args.simulator}: fast_rgb format {tuple(fast.shape)}/{fast.dtype} != ({n},3,{h},{w})/float32")
    elif not (float(fast.min()) >= 0.0 and float(fast.max()) <= 1.0):
        fails.append(
            f"{args.simulator}: fast_rgb out of [0,1] (min={float(fast.min()):.3f} max={float(fast.max()):.3f})"
        )
    slow = first["slow_rgb"]
    if tuple(slow.shape) != (n, h, w, 3) or slow.dtype != torch.uint8:
        fails.append(
            f"{args.simulator}: slow_rgb (untransformed) {tuple(slow.shape)}/{slow.dtype} != ({n},{h},{w},3)/uint8"
        )

    # 2. Rate: step 3 control steps. slow_cam (decimation=3) should hold its first frame for the
    #    whole window; fast_cam (decimation=1) renders each step. Nudge the robot between steps so a
    #    held frame is distinguishable from a freshly rendered one.
    slow0 = first["slow_rgb"].clone()
    fast0 = first["fast_rgb"].clone()
    slow_changed = False
    fast_changed = False
    for _ in range(1, 3):  # steps 2 and 3 of the decimation=3 window
        robot_states[:, 0] += 0.05  # shift the camera so a freshly rendered frame differs
        sim.set_actor_states(["robot"], torch.arange(n, device=device), robot_states)
        step(sim, 1)
        obs = _obs()
        if not torch.equal(obs["slow_rgb"], slow0):
            slow_changed = True
        if not torch.equal(obs["fast_rgb"], fast0):
            fast_changed = True
    print(
        f"[{args.simulator}] over a dec=3 window: slow_cam changed={slow_changed} (want False), "
        f"fast_cam changed={fast_changed} (want True)"
    )
    if slow_changed:
        fails.append(
            f"{args.simulator}: slow_cam (decimation=3) frame changed within its 3-step hold window; gate not honored"
        )
    if not fast_changed:
        fails.append(
            f"{args.simulator}: fast_cam (decimation=1) frame never changed while the camera moved; not re-rendering"
        )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: camera obs pipeline format + decimation rate correct across {n} env(s)")
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
