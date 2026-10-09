"""Live Isaac Gym regression for subset-shaped robot DOF writes."""

# Isaac Gym must load before torch in this subprocess.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import dataclasses
import sys

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment

from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit

NUM_ENVS = 4


def _build(device: str):
    config = build_run_sim_config("isaacgym", "empty", "g1-29dof", "terrain_locomotion_plane")
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=NUM_ENVS),
    )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    env_origins = torch.zeros(NUM_ENVS, 3, device=sim.device)
    env_origins[:, 0] = torch.arange(NUM_ENVS, device=sim.device, dtype=torch.float32) * 5.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(NUM_ENVS, env_origins, base_init)
    sim.prepare_sim()
    sim.simulate_at_each_physics_step()
    sim.refresh_sim_tensors()
    return sim


def _fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    from isaacgym import gymapi
    from holosoma.utils.safe_torch_import import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    args = parser.parse_args()

    torch.manual_seed(0)
    sim = _build(args.device)

    robot_indices = torch.tensor(
        [
            [
                sim.gym.get_actor_dof_index(
                    sim.envs[env_id],
                    sim.robot_handles[env_id],
                    dof_id,
                    gymapi.DOMAIN_SIM,
                )
                for dof_id in range(sim.num_dof)
            ]
            for env_id in range(NUM_ENVS)
        ],
        device=sim.device,
        dtype=torch.long,
    )
    expected_rows = NUM_ENVS * sim.num_dof
    if sim.dof_state.shape != (expected_rows, 2):
        return _fail(f"global DOF state shape {tuple(sim.dof_state.shape)} != {(expected_rows, 2)}")
    before = sim.dof_state.clone()

    selected_envs = torch.tensor([0, 2], device=sim.device, dtype=torch.long)
    unselected_envs = torch.tensor([1, 3], device=sim.device, dtype=torch.long)
    lower = sim.hard_dof_pos_limits[:, 0]
    upper = sim.hard_dof_pos_limits[:, 1]
    midpoint = (lower + upper) * 0.5
    offset = (upper - lower) * 0.1
    target = torch.empty(len(selected_envs), sim.num_dof, 2, device=sim.device)
    target[0, :, 0] = midpoint - offset
    target[1, :, 0] = midpoint + offset
    target[0, :, 1] = -0.15
    target[1, :, 1] = 0.15

    sim.set_dof_state_tensor_robots(selected_envs, target.reshape(-1, 2))
    sim.refresh_sim_tensors()
    after = sim.dof_state.clone()

    selected_rows = robot_indices[selected_envs]
    unselected_rows = robot_indices[unselected_envs]
    if not torch.allclose(after[selected_rows], target, atol=1e-6, rtol=0.0):
        return _fail("selected robot environments did not receive their distinct target states")
    if not torch.equal(after[unselected_rows], before[unselected_rows]):
        return _fail("robot DOF state changed outside the selected environment subset")
    if not torch.allclose(sim.dof_pos[selected_envs], target[..., 0], atol=1e-6, rtol=0.0):
        return _fail("robot position view does not reflect the simulator write")
    if not torch.allclose(sim.dof_vel[selected_envs], target[..., 1], atol=1e-6, rtol=0.0):
        return _fail("robot velocity view does not reflect the simulator write")

    print("PASS: subset-shaped robot write reached Isaac Gym and preserved unselected environments")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
