#!/bin/bash

# Statue whole body tracking training script - COMPLIANT version
# G1-like kp/kd values, action_scale=0.25

source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:statue-v0-1-wbt-compliant \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=statue_wbt_compliant \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v0_1/whole_body_tracking/walk_chunk_0000_mj.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \

