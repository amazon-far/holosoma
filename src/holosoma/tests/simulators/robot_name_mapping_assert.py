"""Exercise configured order against real backend assets with a different native order."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from types import SimpleNamespace

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _robot_presets
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit

# An independently stated physical correspondence, used for native API reads below.
JOINTS = ["slide_joint", "shoulder_joint", "elbow_joint"]
BODIES = ["slider", "base_link", "link2"]


def mapped_robot():
    robot = _robot_presets.test_arm_fixed
    return dataclasses.replace(
        robot,
        dof_names=["slide_joint", "shoulder_joint", "elbow_joint"],
        body_names=["slider", "base_link", "link2"],
        joint_tags={},
        body_tags={"#base": ["base_link"]},
        dof_pos_limits=dict(
            zip(["slide_joint", "shoulder_joint", "elbow_joint"], zip([0.0, -1.5, -1.5], [0.05, 1.5, 1.5]))
        ),
        dof_vel_limits={"slide_joint": 1.0, "shoulder_joint": 9.0, "elbow_joint": 8.0},
        dof_effort_limits={"slide_joint": 40.0, "shoulder_joint": 18.0, "elbow_joint": 16.0},
        dof_armature={"slide_joint": 0.003, "shoulder_joint": 0.02, "elbow_joint": 0.03},
        dof_joint_friction={"slide_joint": 0.0, "shoulder_joint": 0.0, "elbow_joint": 0.0},
        init_state=dataclasses.replace(
            robot.init_state, default_joint_positions={"slide_joint": 0.015, "shoulder_joint": 0.1, "elbow_joint": -0.2}
        ),
    )


def build(simulator):
    _robot_presets.register()
    config = build_run_sim_config(simulator, "empty", "test-arm-fixed", "terrain_locomotion_plane")
    device = "cpu" if simulator == "mujoco" else "cuda:0"
    num_envs = 1 if simulator == "mujoco" else 4
    config = dataclasses.replace(
        config,
        robot=mapped_robot(),
        device=device,
        training=dataclasses.replace(config.training, num_envs=num_envs, headless=True),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(
                    config.simulator.config.sim,
                    physx=dataclasses.replace(config.simulator.config.sim.physx, enable_dof_force_sensors=True),
                ),
            ),
        ),
    )
    env, device, app = setup_simulation_environment(config, device=device)
    import torch

    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    origins[:, 0] = torch.arange(num_envs, device=device) * 5.0
    sim.create_envs(num_envs, origins, torch.tensor(config.robot.init_state.root_state_list(), device=device))
    check_body_randomization(sim, simulator)
    sim.prepare_sim()
    return sim, app


def native_masses(sim, backend):
    """Include the omitted helper body so an unintended write cannot escape the comparison."""
    import torch

    names = ["base_link", "link1", "link2", "slider"]
    if backend in ("mujoco", "mjwarp"):
        import mujoco

        ids = [mujoco.mj_name2id(sim.root_model, mujoco.mjtObj.mjOBJ_BODY, "robot_" + name) for name in names]
        if backend == "mujoco":
            return torch.tensor(sim.root_model.body_mass[ids], dtype=torch.float32)[None]
        import warp as wp

        return wp.to_torch(sim.backend.mjw_model.body_mass)[:, ids].cpu().clone()
    if backend == "isaacgym":
        ids = [sim.gym.get_asset_rigid_body_names(sim.robot_asset).index(name) for name in names]
        return torch.tensor(
            [
                [props[i].mass for i in ids]
                for env, handle in zip(sim.envs, sim.robot_handles)
                for props in [sim.gym.get_actor_rigid_body_properties(env, handle)]
            ]
        )
    ids = [sim._robot.body_names.index(name) for name in names]
    return sim._robot.root_physx_view.get_masses()[:, ids].cpu().clone()


def check_body_randomization(sim, backend):
    import torch

    from holosoma.managers.randomization.terms.locomotion import randomize_mass_startup
    from holosoma.utils.robot_name_resolver import RobotNameResolver
    from holosoma.utils.sampler import STAGE_SETUP, TermSampler

    if backend == "mjwarp":
        sim.prepare_randomization_fields(["body_mass", "body_inertia"])
    env = SimpleNamespace(
        simulator=sim,
        num_envs=sim.num_envs,
        device=sim.sim_device,
        robot_names=RobotNameResolver(sim.robot_config, sim.sim_device),
    )
    selected = torch.arange(0, sim.num_envs, 2, device=sim.sim_device)
    expected = native_masses(sim, backend)
    # "slider" is canonical column 0 / native slider; "#base" is canonical column 1 / native base_link.
    expected[selected.cpu(), 3] *= 1.5
    expected[selected.cpu(), 0] += 0.25
    randomize_mass_startup(
        env,
        selected,
        sampler=TermSampler.bind(0, "name_mapping", STAGE_SETUP),
        link_bodies=["slider"],
        link_mass_range=(1.5, 1.5),
        base_body="#base",
        added_mass_range=(0.25, 0.25),
    )
    torch.testing.assert_close(native_masses(sim, backend), expected)


def native_state(sim, backend):
    """Read by original asset names through engine APIs, independently of Holosoma's maps."""
    import torch

    if backend in ("mujoco", "mjwarp"):
        import mujoco

        model = sim.root_model
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_" + name) for name in JOINTS]
        bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_" + name) for name in BODIES]
        pos_ids, vel_ids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        if backend == "mujoco":
            return (
                torch.tensor(sim.root_data.qpos[pos_ids], dtype=torch.float32)[None],
                torch.tensor(sim.root_data.qvel[vel_ids], dtype=torch.float32)[None],
                torch.tensor(sim.root_data.xpos[bodies], dtype=torch.float32)[None],
            )
        import warp as wp

        return (
            sim.backend.qpos_t[:, pos_ids.tolist()].clone(),
            sim.backend.qvel_t[:, vel_ids.tolist()].clone(),
            wp.to_torch(sim.backend.mjw_data.xpos)[:, bodies].clone(),
        )
    if backend == "isaacgym":
        from isaacgym import gymtorch

        joint_ids = [sim.gym.get_asset_dof_names(sim.robot_asset).index(name) for name in JOINTS]
        body_ids = [sim.gym.get_asset_rigid_body_names(sim.robot_asset).index(name) for name in BODIES]
        dofs = gymtorch.wrap_tensor(sim.gym.acquire_dof_state_tensor(sim.sim)).view(sim.num_envs, 3, 2)
        bodies = gymtorch.wrap_tensor(sim.gym.acquire_rigid_body_state_tensor(sim.sim)).view(sim.num_envs, -1, 13)
        return dofs[:, joint_ids, 0].clone(), dofs[:, joint_ids, 1].clone(), bodies[:, body_ids, :3].clone()
    robot = sim._robot
    joint_ids = [robot.joint_names.index(name) for name in JOINTS]
    body_ids = [robot.body_names.index(name) for name in BODIES]
    return (
        robot.data.joint_pos[:, joint_ids].clone(),
        robot.data.joint_vel[:, joint_ids].clone(),
        robot.data.body_link_pos_w[:, body_ids].clone(),
    )


def check_dynamics_views(sim, backend):
    import torch

    if backend in ("mujoco", "mjwarp"):
        import mujoco

        model = sim.root_model
        motors = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot_" + name) for name in JOINTS]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_" + name) for name in JOINTS]
        if backend == "mujoco":
            forces = torch.tensor(sim.root_data.actuator_force[motors], dtype=torch.float32)[None]
            accelerations = torch.tensor(sim.root_data.qacc[model.jnt_dofadr[joints]], dtype=torch.float32)[None]
        else:
            import warp as wp

            forces = wp.to_torch(sim.backend.mjw_data.actuator_force)[:, motors]
            accelerations = wp.to_torch(sim.backend.mjw_data.qacc)[:, model.jnt_dofadr[joints].tolist()]
        torch.testing.assert_close(sim.dof_acc[:], accelerations)
    elif backend == "isaacgym":
        from isaacgym import gymtorch

        joints = [sim.gym.get_asset_dof_names(sim.robot_asset).index(name) for name in JOINTS]
        bodies = [sim.gym.get_asset_rigid_body_names(sim.robot_asset).index(name) for name in BODIES]
        forces = gymtorch.wrap_tensor(sim.gym.acquire_dof_force_tensor(sim.sim)).view(sim.num_envs, -1)[:, joints]
        actor_name = sim.robot_config.asset.robot_type
        native_jacobian = gymtorch.wrap_tensor(sim.gym.acquire_jacobian_tensor(sim.sim, actor_name))
        native_mass = gymtorch.wrap_tensor(sim.gym.acquire_mass_matrix_tensor(sim.sim, actor_name))
        # The fixed root has no Jacobian row; every remaining row and column uses config order.
        torch.testing.assert_close(sim.jacobian, native_jacobian[:, [b - 1 for b in bodies if b != 0]][..., joints])
        torch.testing.assert_close(sim.massmatrix, native_mass[:, joints][:, :, joints])
    else:
        joints = [sim._robot.joint_names.index(name) for name in JOINTS]
        forces = sim._robot.data.applied_torque[:, joints]
    assert torch.any(forces != 0), "Force readback must exercise a nonzero signal."
    for env_id in range(sim.num_envs):
        torch.testing.assert_close(sim.get_dof_forces(env_id), forces[env_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    args = parser.parse_args()
    sim, app = build(args.simulator)  # keep the SimulationApp alive through the checks
    import torch

    device = sim.sim_device
    assert sim.dof_names == ["slide_joint", "shoulder_joint", "elbow_joint"]
    # link1 remains in the asset, outside this observation list.
    assert sim.body_names == ["slider", "base_link", "link2"]
    if args.simulator == "isaacsim":
        torch.testing.assert_close(sim._height_scanner.data.pos_w, sim._robot.data.root_pos_w)
    limits, velocity, effort = sim.get_dof_limits_properties()
    torch.testing.assert_close(limits, torch.tensor([[0, 0.05], [-1.5, 1.5], [-1.5, 1.5]], device=device))
    torch.testing.assert_close(velocity, torch.tensor([1.0, 9.0, 8.0], device=device))
    torch.testing.assert_close(effort, torch.tensor([40.0, 18.0, 16.0], device=device))

    all_envs = torch.arange(sim.num_envs, device=device)
    selected = all_envs[::2]
    untouched = all_envs[1::2]
    # Two distinct writes catch stale advanced-index copies. Partial resets must preserve other envs.
    for iteration in range(2):
        before_pos, before_vel, _ = native_state(sim, args.simulator)
        q = torch.tensor([0.01 + 0.01 * iteration, 0.2, -0.3], device=device).repeat(len(selected), 1)
        v = torch.tensor([0.01, 0.02, -0.03], device=device).repeat(len(selected), 1)
        q += torch.arange(len(selected), device=device)[:, None] * 0.001
        sim.dof_pos[selected] = q
        sim.dof_vel[selected] = v
        sim.set_dof_state_tensor_robots(selected, sim.dof_state)
        sim.refresh_sim_tensors()
        native_pos, native_vel, _ = native_state(sim, args.simulator)
        torch.testing.assert_close(native_pos[selected], q)
        torch.testing.assert_close(native_vel[selected], v)
        torch.testing.assert_close(native_pos[untouched], before_pos[untouched])
        torch.testing.assert_close(native_vel[untouched], before_vel[untouched])
        torch.testing.assert_close(sim.dof_pos[:], native_pos)
        torch.testing.assert_close(sim.dof_vel[:], native_vel)

    # Pin the two hinges and drive only the prismatic joint in canonical column 0.
    # This checks the physical effect, in addition to comparing the live engine arrays.
    initial_extension = sim.dof_pos[:, 0].clone()
    for _ in range(60):
        q, v = sim.dof_pos[:], sim.dof_vel[:]
        torque = torch.zeros(sim.num_envs, 3, device=device)
        torque[:, 0] = 3.0
        torque[:, 1:] = -20.0 * q[:, 1:] - 2.0 * v[:, 1:]
        sim.apply_torques_at_dof(torque)
        sim.apply_torques_at_dof(torch.tensor([4.0], device=device), dof_indices=[0])
        sim.simulate_at_each_physics_step()
        sim.refresh_sim_tensors()
        native_pos, native_vel, native_bodies = native_state(sim, args.simulator)
        torch.testing.assert_close(sim.dof_pos[:], native_pos)
        torch.testing.assert_close(sim.dof_vel[:], native_vel)
        torch.testing.assert_close(sim.rigid_body_pos_w, native_bodies, atol=1e-5, rtol=1e-5)
    assert torch.all(sim.dof_pos[:, 0] > initial_extension + 0.01), sim.dof_pos[:]
    check_dynamics_views(sim, args.simulator)
    print(
        f"PASS: {args.simulator}: aliases, reordered joints/bodies, limits, "
        "body randomization, partial resets, reads and torques"
    )
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
