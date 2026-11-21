from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from holosoma.envs.policy_task import PolicyTask
from holosoma.utils.simulator_config import SimulatorType

if TYPE_CHECKING:
    from holosoma.managers.command.terms.wbt import MotionCommand
    from holosoma.managers.randomization.terms.locomotion import PushRandomizerState


class WholeBodyTrackingManager(PolicyTask):
    _in_post_physics_step: bool = False

    def _init_buffers(self) -> None:
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        self._init_domain_rand_buffers()

    @property
    def base_quat(self) -> torch.Tensor:
        return self.simulator.base_quat

    def _update_log_dict(self) -> None:
        super()._update_log_dict()
        motion_command = cast("MotionCommand", self.command_manager.get_state("motion_command"))
        motion_command.update_metrics()
        self.log_dict.update(motion_command.metrics)

    def _post_physics_step(self) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        # MotionCommand.step() refreshes the targets after this frame's resets, before anything reads them.
        self._in_post_physics_step = True
        try:
            return super()._post_physics_step()
        finally:
            self._in_post_physics_step = False

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        # A reset outside step() is not followed by MotionCommand.step(), so refresh the targets
        # the observations and terminations read.
        if not self._in_post_physics_step:
            motion_command = cast("MotionCommand", self.command_manager.get_state("motion_command"))
            motion_command.update_relative_body_targets()
        super()._after_reset_synchronization(env_ids)

    def _reset_robot_states_callback(
        self, env_ids: torch.Tensor, target_states: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        # TODO(jchen): Now,reset robot/object states is implemented in command/terms/wbt.MotionCommand.reset
        # discuss whether to move to here in the future.
        pass

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self) -> None:
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids: torch.Tensor) -> None:
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = cast("PushRandomizerState | None", self.randomization_manager.get_state("push_randomizer_state"))
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        # Additive push to match BeyondMimic/IsaacLab's push_by_setting_velocity.
        self.simulator.robot_root_states[env_ids, 7:13] += self.push_robot_vel_buf[env_ids]
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self) -> None:
        motion_command = cast("MotionCommand", self.command_manager.get_state("motion_command"))
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
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

        # object
        if motion_command.motion.has_object:
            real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
            real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
            real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["real_object"].visualize(real_object_pos_xyz, real_object_quat_wxyz)

            motion_object_pos_xyz = motion_command.object_pos_w.clone()
            motion_object_quat_xyzw = motion_command.object_quat_w.clone()
            motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["motion_object"].visualize(
                motion_object_pos_xyz, motion_object_quat_wxyz
            )

    def _draw_debug_vis(self) -> None:
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()

    def step_visualize_motion(self, actions: torch.Tensor) -> bool:
        motion_command = cast("MotionCommand", self.command_manager.get_state("motion_command"))
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            object_name = motion_command.object_name
            self.simulator.set_actor_states([object_name], env_ids, object_states)

        self.simulator.forward_kinematics()
        self.simulator.sim.render()  # type: ignore[attr-defined]

        time.sleep(dt)

        result: bool = motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
        return result
