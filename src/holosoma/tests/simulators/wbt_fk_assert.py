"""Probe: the URDF forward kinematics behind WBT default-pose transitions agrees with IsaacSim's.

Asserts, on a real G1:
  (a) ``_fk_body_poses`` leaves the sim's root, joint and body state untouched;
  (b) its body poses match the ones IsaacSim computes for the same states;
  (c) center-of-mass positions differ from link origins.

Writes "OK" to ``--result-file`` once every check passes; IsaacSim teardown swallows a non-zero exit.

Usage:
  python wbt_fk_assert.py --simulator isaacsim --result-file /tmp/ok.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import types

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.sim_utils import setup_simulation_environment

NUM_FRAMES = 8
# Enough joint travel that a wrong joint mapping cannot hide inside numerical noise.
JOINT_SWEEP_RAD = 0.4
POS_TOL_M = 1e-4


class _FkProbe:
    """The smallest object ``_fk_body_poses`` will run against, borrowing MotionCommand's real methods."""

    _fk_body_poses = MotionCommand._fk_body_poses
    _body_com_offsets_b = MotionCommand._body_com_offsets_b

    def __init__(self, simulator, robot_config, device):
        self._env = types.SimpleNamespace(simulator=simulator, robot_config=robot_config)
        self.device = device
        # MotionCommand.__init__ declares these; the probe borrows methods, not the constructor.
        self._com_offsets_cache = None
        self._urdf_fk = None


def _build(simulator: str, num_envs: int):
    from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

    argv = [f"simulator:{simulator}", "robot:g1-29dof", "terrain:terrain_locomotion_plane", "scene:empty"]
    config = tyro.cli(RunSimConfig, args=argv)
    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config, bridge=BridgeConfig(enabled=False), virtual_gantry=VirtualGantryCfg(enabled=False)
        ),
    )
    config = dataclasses.replace(
        config,
        simulator=sim_cfg,
        device=("cpu" if simulator == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs),
    )
    import torch

    env, device, _ = setup_simulation_environment(config, device=config.device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    if num_envs > 1:
        origins[:, 0] = torch.arange(num_envs, device=device, dtype=torch.float32) * 8.0
    init = config.robot.init_state
    base_init = torch.tensor(list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device)
    sim.create_envs(num_envs, origins, base_init)
    sim.prepare_sim()
    return sim, config, device


def _sweep_frames(sim, config, device):
    """A joint sweep plus a moving root, i.e. what a real transition segment looks like."""
    import torch

    tau = torch.linspace(0.0, 1.0, NUM_FRAMES, device=device)
    joint_pos = sim.dof_pos[0].clone().unsqueeze(0).repeat(NUM_FRAMES, 1)
    joint_pos += tau.unsqueeze(-1) * JOINT_SWEEP_RAD

    init = config.robot.init_state
    root_pos = torch.tensor(list(init.pos), device=device, dtype=torch.float32).unsqueeze(0).repeat(NUM_FRAMES, 1)
    root_pos[:, 0] += tau * 0.25
    # xyzw identity.
    root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(NUM_FRAMES, 1)
    return joint_pos, root_pos, root_quat


def _sim_state(sim):
    return [
        t.clone()
        for t in (sim.robot_root_states[:], sim.dof_pos, sim.dof_vel, sim.rigid_body_pos_w, sim.rigid_body_quat_w)
    ]


def _isaacsim_fk(sim, joint_pos, root_pos, root_quat):
    """Write frame i into env i and read back the body poses IsaacSim computes for it."""
    import torch

    env_ids = torch.arange(joint_pos.shape[0], device=joint_pos.device)
    origins = sim.scene.env_origins[env_ids]
    sim.robot_root_states[env_ids, :3] = root_pos + origins
    sim.robot_root_states[env_ids, 3:7] = root_quat
    sim.robot_root_states[env_ids, 7:13] = 0.0
    sim.dof_pos[env_ids] = joint_pos
    sim.dof_vel[env_ids] = 0.0
    sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
    sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
    sim.write_state_updates()
    sim.refresh_sim_tensors()
    return sim.rigid_body_pos_w[env_ids] - origins.unsqueeze(1), sim.rigid_body_quat_w[env_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="URDF-vs-IsaacSim FK probe for WBT transitions.")
    parser.add_argument("--simulator", required=True, choices=["isaacsim"])
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    import torch

    sim, config, device = _build(args.simulator, NUM_FRAMES)
    joint_pos, root_pos, root_quat = _sweep_frames(sim, config, device)

    before = _sim_state(sim)
    body_pos, body_quat, body_com_pos = _FkProbe(sim, config.robot, device)._fk_body_poses(  # type: ignore[misc]
        joint_pos, root_pos, root_quat
    )
    # (a) exact equality: nothing may have been written.
    if not all(torch.equal(b, a) for b, a in zip(before, _sim_state(sim), strict=True)):
        print("FAIL: _fk_body_poses changed the simulator state")
        return 1

    if body_pos.shape != (NUM_FRAMES, sim.rigid_body_pos_w.shape[1], 3):
        print(f"FAIL: unexpected body_pos shape {tuple(body_pos.shape)}")
        return 1

    # (b) a URDF/USD import difference (merged fixed joints, reordered bodies) diverges here.
    sim_pos, sim_quat = _isaacsim_fk(sim, joint_pos, root_pos, root_quat)
    pos_gap = float((sim_pos - body_pos).abs().max())
    # q and -q are the same rotation, so compare by angle rather than by component.
    quat_gap = float((1.0 - (sim_quat * body_quat).sum(dim=-1).abs()).max())
    if pos_gap > POS_TOL_M or quat_gap > 1e-6:
        worst = int((sim_pos - body_pos).abs().amax(dim=(0, 2)).argmax())
        print(f"FAIL: URDF and IsaacSim FK disagree (pos {pos_gap:.6f} m, quat {quat_gap:.2e}, worst body {worst})")
        return 1

    # (c) G1 links all have mass, so at least one CoM offset must be nonzero.
    com_offset = float((body_com_pos - body_pos).norm(dim=-1).max())
    if com_offset < 1e-4:
        print(f"FAIL: every body CoM sits on its link origin (max offset {com_offset:.2e} m)")
        return 1

    print(
        f"[{args.simulator}] FK ok: {NUM_FRAMES} frames, {body_pos.shape[1]} bodies, "
        f"URDF-vs-sim gap {pos_gap:.2e} m / {quat_gap:.2e}, max CoM offset {com_offset:.4f} m"
    )
    if args.result_file:
        with open(args.result_file, "w") as handle:
            handle.write("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
