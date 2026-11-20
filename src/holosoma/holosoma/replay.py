from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType

if TYPE_CHECKING:
    from holosoma.envs.policy_task import PolicyTask
    from holosoma.managers.command.terms.wbt import MotionCommand


def replay(tyro_config: ExperimentConfig) -> None:
    with tyro_config.create_task() as env:
        done = False
        while not done:
            env.simulator.sim.step()  # type: ignore[attr-defined]
            done = step_visualize_motion(env)


def step_visualize_motion(env: PolicyTask) -> bool:
    """Set the robot (and object) to the motion's next frame; return whether the motion has ended."""
    motion_command = cast("MotionCommand", env.command_manager.get_state("motion_command"))
    dt = 1.0 / float(motion_command.motion.fps)
    motion_command.step()
    print("time_steps: ", motion_command.time_steps[0].item())
    if env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
        _draw_debug_vis_isaacsim(motion_command)

    # set root_states_from_motion_command
    root_pos = motion_command.root_pos_w.clone()
    root_ori = motion_command.root_quat_w.clone()  # wxyz
    root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

    joint_pos = motion_command.joint_pos.clone()
    joint_vel = motion_command.joint_vel.clone()

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.simulator.dof_pos[env_ids] = joint_pos
    env.simulator.dof_vel[env_ids] = joint_vel

    env.simulator.robot_root_states[env_ids, :3] = root_pos
    env.simulator.robot_root_states[env_ids, 3:7] = root_ori
    env.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
    env.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

    if motion_command.motion.has_object:
        # set object root_states from motion command
        object_pos = motion_command.object_pos_w.clone()
        object_ori = motion_command.object_quat_w.clone()
        object_lin_vel = motion_command.object_lin_vel_w.clone()

        object_states = torch.zeros(len(env_ids), 13, device=env.device)
        object_states[:, :3] = object_pos[:]
        object_states[:, 3:7] = object_ori[:]
        object_states[:, 7:10] = object_lin_vel[:]
        object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
        object_name = motion_command.object_name
        env.simulator.set_actor_states([object_name], env_ids, object_states)

    env.simulator.forward_kinematics()
    env.simulator.sim.render()  # type: ignore[attr-defined]

    time.sleep(dt)

    result: bool = motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
    return result


def _draw_debug_vis_isaacsim(motion_command: MotionCommand) -> None:
    # torso link
    real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
    real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
    real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
    motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

    motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
    motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
    motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
    motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

    for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
        motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
        motion_command.visualization_markers[f"motion_{body_names}"].visualize(motion_robot_body_pos_xyz.unsqueeze(0))

    # object
    if motion_command.motion.has_object:
        real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
        real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
        real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_object"].visualize(real_object_pos_xyz, real_object_quat_wxyz)

        motion_object_pos_xyz = motion_command.object_pos_w.clone()
        motion_object_quat_xyzw = motion_command.object_quat_w.clone()
        motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_object"].visualize(motion_object_pos_xyz, motion_object_quat_wxyz)


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
