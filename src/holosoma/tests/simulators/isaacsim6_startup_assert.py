"""Physics-only startup must not load an unused renderer, Kit or physics engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.simulator import IsaacSim6Config
from holosoma.utils.sim_utils import close_simulation_resources
from tests.simulators import _camera_presets
from tests.simulators._native_runtime_assertions import assert_native_runtime
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, step


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaacsim6"], required=True)
    parser.add_argument("--physics", choices=["physx", "newton"], required=True)
    parser.add_argument("--renderer", choices=["isaac_rtx", "newton_warp", "ovrtx"], required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    config = build_run_sim_config(args.simulator, "empty", "g1-29dof", "terrain_locomotion_plane")
    native = IsaacSim6Config(physics_backend=args.physics, renderer=args.renderer)
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(config.simulator.config.sim, isaacsim6=native),
            ),
        ),
        scene=_camera_presets.panel_target,
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=2, headless=True),
        logger=dataclasses.replace(
            config.logger, headless_recording=False, video=dataclasses.replace(config.logger.video, enabled=False)
        ),
    )
    assert not config.sensor
    sim = app = None
    try:
        sim, _device, app = launch_simulator(config, device="cuda:0")
        launched = time.perf_counter()
        sim.set_headless(True)
        sim.setup()
        sim.setup_terrain()
        sim.load_assets()
        import torch
        from isaaclab.utils.string import string_to_callable

        initial = torch.tensor(
            list(config.robot.init_state.pos)
            + list(config.robot.init_state.rot)
            + list(config.robot.init_state.lin_vel)
            + list(config.robot.init_state.ang_vel),
            dtype=torch.float32,
            device="cuda:0",
        )
        sim.create_envs(2, torch.zeros(2, 3, device="cuda:0"), initial)
        sim.prepare_sim()
        ready = time.perf_counter()
        manager = string_to_callable(sim.sim.cfg.physics.class_type)
        if args.physics == "newton":
            from isaaclab_newton.physics import NewtonMJWarpManager

            assert issubclass(manager, NewtonMJWarpManager)
        else:
            assert manager.__module__.startswith("isaaclab_ov.")
        step(sim, 20)
        assert torch.isfinite(sim.dof_pos).all()
        dependencies = assert_native_runtime(physics=args.physics, uses_kit=False)
        assert "isaaclab_ov.renderers.ovrtx_renderer" not in sys.modules
        assert "isaaclab_newton.renderers.newton_warp_renderer" not in sys.modules
        assert not any("ovrtx" in str(path).lower() for path in dependencies["libraries"])
        report = {
            "physics": args.physics,
            "renderer_preference": args.renderer,
            "launch_seconds": launched - started,
            "ready_seconds": ready - started,
            **dependencies,
        }
        Path(args.result_file).with_suffix(".startup.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PASS: physics-only {args.physics}/{args.renderer}; launch={launched - started:.3f}s, "
            f"ready={ready - started:.3f}s, no Kit or renderer",
            flush=True,
        )
    finally:
        close_simulation_resources(sim, app)
    from isaaclab.sim import SimulationContext

    assert SimulationContext.instance() is None
    assert app is not None and not app.is_running()
    Path(args.result_file).write_text("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
