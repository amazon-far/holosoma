from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from holosoma.envs.policy_task import PolicyTask
from holosoma.utils.simulator_config import SimulatorType

if TYPE_CHECKING:
    from holosoma.managers.command.terms.wbt import MotionCommand
    from holosoma.managers.randomization.terms.locomotion import PushRandomizerState


class WholeBodyTrackingManager(PolicyTask):
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

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        # A reset does not step, so refresh the targets the observations and terminations read.
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
        if motion_command.motion.has_object:  # type: ignore[has-type]
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
