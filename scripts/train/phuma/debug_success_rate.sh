#!/bin/bash
# Debug: check why success_rate is always 0
source scripts/source_isaacsim_setup.sh
CUDA_VISIBLE_DEVICES=3 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma \
    logger:wandb \
    --logger.project holosoma_test \
    --logger.name=debug_success_rate \
    --logger.mode offline \
    --logger.video.enabled False \
    --training.num_envs 4 \
    --algo.config.num_learning_iterations 2 \
    --algo.config.eval_interval 1 \
    --algo.config.save_interval 100 \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_test_tiny.txt" \
    --command.setup_terms.motion_command.params.motion_config.pool_size 4
