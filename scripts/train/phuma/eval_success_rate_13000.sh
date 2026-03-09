#!/bin/bash
# Evaluate success rate on full training set with model_13000.pt
# Runs 0.5m threshold first, then 0.25m threshold, both on GPU 4

cd /home/nas4_user/kyungminlee/work/holosoma

CALLBACK_FILE="src/holosoma/holosoma/agents/callbacks/success_rate_callback.py"
CHECKPOINT="/home/nas4_user/kyungminlee/work/holosoma/logs/holosoma/20260305_g1_29dof_phuma_future_diverse_action_scale/model_13000.pt"

source scripts/source_isaacsim_setup.sh

# ============ 0.5m threshold ============
echo ">>> Setting MOTION_FAR_THRESHOLD = 0.5"
sed -i 's/^MOTION_FAR_THRESHOLD = .*/MOTION_FAR_THRESHOLD = 0.5/' "$CALLBACK_FILE"

CUDA_VISIBLE_DEVICES=4 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.project holosoma_test \
    --logger.name=eval_sr_13000_05m \
    --logger.mode offline \
    --logger.video.enabled False \
    --training.num_envs 8192 \
    --training.checkpoint="$CHECKPOINT" \
    --algo.config.num_learning_iterations 13001 \
    --algo.config.eval_interval 1 \
    --algo.config.save_interval 99999 \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_train.txt" \
    --command.setup_terms.motion_command.params.motion_config.pool_size 8192 \
    robot:g1-29dof-diverse-action-scale

echo ">>> 0.5m eval done"

# ============ 0.25m threshold ============
echo ">>> Setting MOTION_FAR_THRESHOLD = 0.25"
sed -i 's/^MOTION_FAR_THRESHOLD = .*/MOTION_FAR_THRESHOLD = 0.25/' "$CALLBACK_FILE"

CUDA_VISIBLE_DEVICES=4 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.project holosoma_test \
    --logger.name=eval_sr_13000_025m \
    --logger.mode offline \
    --logger.video.enabled False \
    --training.num_envs 8192 \
    --training.checkpoint="$CHECKPOINT" \
    --algo.config.num_learning_iterations 13001 \
    --algo.config.eval_interval 1 \
    --algo.config.save_interval 99999 \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_train.txt" \
    --command.setup_terms.motion_command.params.motion_config.pool_size 8192 \
    robot:g1-29dof-diverse-action-scale

echo ">>> 0.25m eval done"
echo ">>> Both evaluations complete. Check logs/holosoma_test/ for results."
