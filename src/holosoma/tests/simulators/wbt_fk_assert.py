"""Probe: the batched forward kinematics behind WBT default-pose transitions returns fresh poses.

``MotionCommand._fk_body_poses`` writes one interpolated frame per environment, flushes
once, and reads the rigid-body pose buffers back. That is only correct if the write actually
invalidates those buffers. IsaacLab's ``ArticulationData.body_*_w`` are timestamp-gated lazy
caches, so a read that does not see the write would hand back the pose from before it -- and the
failure mode is silent: every transition frame would carry the same body pose while the joint
angles vary, i.e. a reference trajectory whose limbs never move. Nothing else in the codebase
depends on this, so nothing else would catch it.

Asserts, on a real G1:
  (a) the root pose round-trips -- the poses we read back are the state we just wrote;
  (b) distinct joint configurations produce distinct body poses (buffers are not frozen);
  (c) writing N frames across N envs matches writing them one env at a time, so the batching is
      not itself hiding a staleness bug;
  (d) the centre-of-mass positions differ from the link origins, so the caller's velocity
      differencing has a real offset to work with (a zero offset would silently be link-origin).

Kinematic consistency of the emitted trajectory is covered on CPU against a closed-form chain in
``managers/command/tests/test_wbt_transition_fields.py``; this harness only checks the parts that
need a live articulation.

Writes "OK" to ``--result-file`` once every check passes. Success is judged on that sentinel, not
the exit code: IsaacSim's teardown swallows a non-zero status, so a crashed run exits 0 and would
otherwise be reported as a pass.

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
# Enough joint travel that a frozen-buffer bug cannot hide inside numerical noise.
JOINT_SWEEP_RAD = 0.4


class _FkProbe:
    """The smallest object ``_fk_body_poses`` will run against.

    Borrows the real methods rather than reimplementing them, so this probes shipped code. Building
    a full MotionCommand would need a task, a config tree and a motion file, none of which this
    check is about.
    """

    _fk_body_poses = MotionCommand._fk_body_poses
    _body_com_offsets_b = MotionCommand._body_com_offsets_b
    _check_fk_poses_are_fresh = MotionCommand._check_fk_poses_are_fresh

    def __init__(self, simulator, device, num_envs: int):
        self._env = types.SimpleNamespace(simulator=simulator)
        self.device = device
        self.num_envs = num_envs
        # MotionCommand.__init__ declares this; the probe borrows methods, not the constructor.
        self._com_offsets_cache = None


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
    # xyzw identity; the sweep is what this probe is about, not the base orientation.
    root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(NUM_FRAMES, 1)
    return joint_pos, root_pos, root_quat


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched FK freshness probe for WBT transitions.")
    parser.add_argument("--simulator", required=True, choices=["isaacsim"])
    parser.add_argument("--num-envs", type=int, default=NUM_FRAMES)
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    import torch

    sim, config, device = _build(args.simulator, args.num_envs)
    joint_pos, root_pos, root_quat = _sweep_frames(sim, config, device)

    # (a) is checked inside _check_fk_poses_are_fresh, which runs on every chunk below.
    batched = _FkProbe(sim, device, args.num_envs)
    body_pos, body_quat, body_com_pos = batched._fk_body_poses(joint_pos, root_pos, root_quat)  # type: ignore[misc]

    if body_pos.shape != (NUM_FRAMES, sim._rigid_body_pos.shape[1], 3):
        print(f"FAIL: unexpected body_pos shape {tuple(body_pos.shape)}")
        return 1
    if not all(torch.isfinite(t).all() for t in (body_pos, body_quat, body_com_pos)):
        print("FAIL: forward kinematics produced non-finite poses")
        return 1

    # (b) distinct joint configurations must produce distinct body poses.
    spread = (body_pos - body_pos[0]).abs().amax(dim=(1, 2))
    if float(spread[1:].min()) < 1e-4:
        print(f"FAIL: body poses barely move across a {JOINT_SWEEP_RAD} rad joint sweep (spread={spread.tolist()})")
        return 1

    # (c) one frame per env must agree with one frame at a time. This is the real staleness check:
    # the chunked path re-reads the buffers per chunk, so if a write is not seen it diverges here.
    chunked = _FkProbe(sim, device, 1)
    chunk_pos, chunk_quat, _ = chunked._fk_body_poses(joint_pos, root_pos, root_quat)  # type: ignore[misc]
    pos_gap = float((chunk_pos - body_pos).abs().max())
    # q and -q are the same rotation, so compare by angle rather than by component.
    quat_gap = float((1.0 - (chunk_quat * body_quat).sum(dim=-1).abs()).max())
    if pos_gap > 1e-4 or quat_gap > 1e-4:
        print(f"FAIL: batched and per-frame FK disagree (pos {pos_gap:.6f}, quat {quat_gap:.6f})")
        return 1

    # (d) the CoM offsets are real, so differencing CoM positions is not secretly the same as
    # differencing link origins. G1 links all have mass, so at least one offset must be nonzero.
    com_offset = float((body_com_pos - body_pos).norm(dim=-1).max())
    if com_offset < 1e-4:
        print(f"FAIL: every body CoM sits on its link origin (max offset {com_offset:.2e} m)")
        return 1

    print(
        f"[{args.simulator}] FK ok: {NUM_FRAMES} frames, {body_pos.shape[1]} bodies, "
        f"max spread {float(spread.max()):.4f} m, batched-vs-chunked gap {pos_gap:.2e}, "
        f"max CoM offset {com_offset:.4f} m"
    )
    if args.result_file:
        with open(args.result_file, "w") as handle:
            handle.write("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
