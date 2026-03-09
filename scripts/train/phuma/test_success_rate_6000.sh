#!/bin/bash
# Test success rate with model_06000.pt checkpoint
# Uses the same config as train_future_diverse_action_scale.sh but with:
#   - 2 iterations only (eval at iteration 1)
#   - checkpoint loaded
#   - small split file for faster eval
source scripts/source_isaacsim_setup.sh
CUDA_VISIBLE_DEVICES=3 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.project holosoma_test \
    --logger.name=test_success_rate_6000 \
    --logger.video.enabled False \
    --training.num_envs 4 \
    --training.checkpoint="/home/nas4_user/kyungminlee/work/holosoma/logs/holosoma/20260305_g1_29dof_phuma_future_diverse_action_scale/model_06000.pt" \
    --algo.config.num_learning_iterations 6002 \
    --algo.config.eval_interval 1 \
    --algo.config.save_interval 99999 \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_test_tiny.txt" \
    --command.setup_terms.motion_command.params.motion_config.pool_size 4 \
    robot:g1-29dof-diverse-action-scale
