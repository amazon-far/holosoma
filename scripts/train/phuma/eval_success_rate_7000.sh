#!/bin/bash
# Evaluate success rate on full training set with model_07000.pt
source scripts/source_isaacsim_setup.sh
CUDA_VISIBLE_DEVICES=3 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.project holosoma_test \
    --logger.name=eval_sr_7000_025m \
    --logger.mode offline \
    --logger.video.enabled False \
    --training.num_envs 8192 \
    --training.checkpoint="/home/nas4_user/kyungminlee/work/holosoma/logs/holosoma/20260305_g1_29dof_phuma_future_diverse_action_scale/model_07000.pt" \
    --algo.config.num_learning_iterations 7001 \
    --algo.config.eval_interval 1 \
    --algo.config.save_interval 99999 \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_train.txt" \
    --command.setup_terms.motion_command.params.motion_config.pool_size 8192 \
    robot:g1-29dof-diverse-action-scale
