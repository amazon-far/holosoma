"""Cross-backend kinematic-playback harness.

Builds a sim with ``sim.kinematic_playback`` + the ``motion-playback`` plugin, drives the
run_sim phase cycle for a generated clip (root held in the air, joints ramping), and asserts:

- exact clip tracking: root height and joint positions match the clip to tolerance — on a
  true-FK backend, no gravity sag at all; on IsaacGym (bounded one-substep fallback) within
  a small tolerance
- FK propagation: rigid-body positions reflect the written root pose
- the kinematic clock advances while the engine clock does not (strict-FK backends)
- the plugin requests shutdown once the clip ends

Usage:
    python tests/simulators/playback_assert.py --simulator mujoco|mjwarp|isaacgym|isaacsim \
        [--num-envs N] [--result-file /tmp/r.txt]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

# Shadow-fix: keep this file's dir from hiding the real isaacsim package.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    sys.path.pop(0)

from holosoma.utils.safe_torch_import import torch

SKIP_EXIT_CODE = 77

ROOT_Z = 1.5
JOINT_RAMP = 0.4
CLIP_SECONDS = 1.0


def _write_clip(path: Path, num_dof: int, fps: float) -> Path:
    import numpy as np

    frames = int(CLIP_SECONDS * fps) + 1
    mat = np.zeros((frames, 7 + num_dof), dtype=np.float32)  # root(7) + joints
    mat[:, 2] = ROOT_Z
    mat[:, 3] = 1.0  # identity quat (wxyz)
    mat[:, 7:] = np.linspace(0.0, JOINT_RAMP, frames, dtype=np.float32)[:, None]
    file = path / "clip.npy"
    np.save(file, mat)
    return file


def _build(simulator: str, num_envs: int, plugin=None, kinematic: bool = False):
    from holosoma.utils.sim_utils import setup_simulation_environment
    from tests.simulators._sim_harness import build_run_sim_config

    config = build_run_sim_config(simulator, "empty", "g1-29dof", "terrain_locomotion_plane")
    sim_cfg = config.simulator
    if kinematic:
        sim_cfg = dataclasses.replace(
            sim_cfg,
            config=dataclasses.replace(
                sim_cfg.config, sim=dataclasses.replace(sim_cfg.config.sim, kinematic_playback=True)
            ),
        )
    device = "cuda:0" if simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config,
        simulator=sim_cfg,
        device=device,
        plugin=plugin or {},
        training=dataclasses.replace(config.training, num_envs=num_envs),
    )
    env, device, _ = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    base_init = torch.tensor(
        list(config.robot.init_state.pos)
        + list(config.robot.init_state.rot)
        + list(config.robot.init_state.lin_vel)
        + list(config.robot.init_state.ang_vel)
    )
    sim.create_envs(num_envs, torch.zeros(num_envs, 3, device=device), base_init)
    sim.prepare_sim()
    sim.install_plugins()
    return sim


def _frame(sim) -> None:
    from holosoma.simulator.base_simulator.hooks import Phase

    control_decimation = sim.simulator_config.sim.control_decimation_steps
    for substep in range(control_decimation):
        if substep == 0:
            sim.hooks.emit(Phase.FRAME_BEGIN)
        sim.hooks.emit(Phase.PRE_STEP)
        sim.simulate_at_each_physics_step()
        sim.hooks.emit(Phase.POST_STEP)
        if substep == control_decimation - 1:
            # The frame closes on the LAST substep, as run_sim's loop does: a periodic-render backend
            # has drawn by then, so a FRAME_END consumer sees this frame's image, not the last one's.
            sim.refresh_sim_tensors()
            sim.hooks.emit(Phase.FRAME_END)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    if args.simulator == "mujoco" and args.num_envs > 1:
        print("SKIP: MuJoCo ClassicBackend is single-env (use mjwarp)")
        return SKIP_EXIT_CODE

    from holosoma.config_types.plugin import MotionPlaybackPluginConfig

    # The clip needs the sim's num_dof but the plugin config needs the clip path + fps before the
    # sim builds; the clip loads lazily on the first hook call, so build first, write after.
    tmp = Path(tempfile.mkdtemp(prefix="playback_assert_"))
    clip_path = tmp / "clip.npy"
    control_hz = 50.0  # a valid divisor of every backend preset's fps; set on the config below
    cfg = MotionPlaybackPluginConfig(motion_files=[str(clip_path)], fps=control_hz)
    sim = _build(args.simulator, args.num_envs, plugin={"play": cfg}, kinematic=True)

    sim_cfg = sim.simulator_config.sim
    control_hz = sim_cfg.fps / sim_cfg.control_decimation_steps
    cfg = dataclasses.replace(cfg, fps=control_hz)  # sync clip fps to the actual control rate
    # Re-point the plugin's config so its lazy load uses the corrected fps.
    sim.installed_plugins["play"].cfg = cfg
    _write_clip(tmp, sim.num_dof, control_hz)
    frames = int(CLIP_SECONDS * control_hz) + 1

    # IsaacGym's forward_kinematics is one bounded physics substep (no FK-only API); everything
    # else is strictly kinematic. Tolerances reflect that.
    strict_fk = args.simulator != "isaacgym"
    pos_tol = 1e-5 if strict_fk else 5e-3

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    # Engine clock may have accrued time during scene setup; the invariant is that it does not
    # advance during PLAYBACK frames (strict-FK backends never step physics there).
    engine_t0 = sim._physics_time()

    n = frames // 2
    for _ in range(n):
        _frame(sim)

    # 1. Root held at clip height: no gravity accumulation.
    root_z = sim.robot_root_states[:, 2]
    check(
        "root-height-tracks-clip",
        bool(torch.all((root_z - ROOT_Z).abs() < pos_tol)),
        f"z={root_z.tolist()} vs {ROOT_Z} tol={pos_tol}",
    )

    # 2. Joints on the clip ramp (frame n's write sampled clip index n-1).
    expected = JOINT_RAMP * (n - 1) / (frames - 1)
    dof_err = (sim.dof_pos[:] - expected).abs().max().item()
    check("joints-track-clip", dof_err < max(pos_tol, 1e-4), f"max_err={dof_err:.2e} expected={expected:.4f}")

    # 3. FK propagated to rigid-body transforms (pelvis z ~= clip root z).
    body_z = sim._rigid_body_pos[:, 0, 2]
    check(
        "body-transforms-follow-fk",
        bool(torch.all((body_z - ROOT_Z).abs() < max(pos_tol, 1e-3))),
        f"pelvis_z={body_z.tolist()}",
    )

    # 4. Kinematic clock advanced one sim dt per step; engine clock did not (strict-FK only).
    expected_t = n * sim_cfg.control_decimation_steps / sim_cfg.fps
    check("kinematic-clock-advances", abs(sim.time() - expected_t) < 1e-6, f"time={sim.time()} vs {expected_t}")
    if strict_fk:
        drift = sim._physics_time() - engine_t0
        check("engine-clock-frozen", abs(drift) < 1e-9, f"drift={drift} over {n} frames")

    # 5. Play to the end: shutdown requested.
    for _ in range(frames - n + 2):
        _frame(sim)
    check("shutdown-at-clip-end", sim.shutdown_requested)

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    if args.result_file:
        Path(args.result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    from tests.simulators._sim_harness import run_and_hard_exit

    run_and_hard_exit(main)
