#!/bin/bash
# Stage-3 + Stage-4 in a single PPO run for a VideoMimic-narrowed student.
#
# Actor obs is xy + yaw command only (drops the step-function pelvis-z and
# roll/pitch from the original motion-mimic obs) so the trained policy can
# be deployed with a 3-DoF gamepad command. Critic still sees the full
# motion target. See `g1_29dof_wbt_observation_videomimic_student_xy_yaw`.
#
# The PPO algo runs a BC→PPO loss schedule (configured in
# `_student_rl_finetune_xy_yaw_algo` and `PPOConfig.bc_*`):
#
#   iter 0–10k       : BC only (against TEACHER_CKPT).
#                      Equivalent to stage-3 distillation.
#   iter 10k–15k     : linear ramp BC→PPO (bc_coef 1→0, ppo_coef 0→1).
#   iter 15k+        : pure PPO motion-tracking finetune (stage-4).
#
# Critic value loss is always on so the critic warms up alongside BC.
#
# Override TEACHER_CKPT='' to skip the BC phase and run plain PPO from
# scratch (equivalent to the previous random-init RL finetune).

source scripts/source_isaacsim_setup.sh

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"
TEACHER_CKPT="${TEACHER_CKPT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260426_g1_ue_multi_terrain_stair_200_future_motion_heightmap_xyz/model_22000.pt}"

TEACHER_ARGS=()
if [ -n "$TEACHER_CKPT" ]; then
    TEACHER_ARGS=(--algo.config.teacher_checkpoint="$TEACHER_CKPT")
fi

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-videomimic-student-rl-finetune-xy-yaw \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=Unreal_Engine \
    --logger.name=g1_unreal_student_RLfinetune_xy_yaw \
    --logger.video.enabled False \
    "${TEACHER_ARGS[@]}" \
    --simulator.config.videomimic-height-map.map-height 11 \
    --simulator.config.videomimic-height-map.map-width 17 \
    --simulator.config.videomimic-height-map.resolution 0.1 \
    --simulator.config.videomimic-height-map.num-channels 3 \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 200 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob 0.3 \
    --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob 0.0 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.dof_pos 0.05 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.root_pos="[0.05, 0.05, 0.05]" \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.root_rot="[0.05, 0.05, 0.1]" \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.root_lin_vel="[0.1, 0.1, 0.1]" \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.root_ang_vel="[0.1, 0.1, 0.1]" \
    --training.num_envs 4096 \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 1000
