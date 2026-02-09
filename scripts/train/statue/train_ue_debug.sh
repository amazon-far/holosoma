#!/bin/bash
# Debug script for statue - reduced num_envs and disabled self_collisions

source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:statue-v0-1-wbt-stiff \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt_debug \
    --logger.name=statue_wbt_stiff_debug \
    --logger.video.enabled False \
    --training.num_envs 128 \
    --robot.asset.enable_self_collisions False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v0_1/whole_body_tracking/walk_chunk_0000_mj.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False
