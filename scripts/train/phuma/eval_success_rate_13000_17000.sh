#!/bin/bash
# Evaluate success rate for model_13000 and model_17000
# Each model: 0.5m threshold once, 0.25m threshold once (4 runs total)
# All on GPU 4, sequential

cd /home/nas4_user/kyungminlee/work/holosoma

CALLBACK_FILE="src/holosoma/holosoma/agents/callbacks/success_rate_callback.py"
CKPT_DIR="/home/nas4_user/kyungminlee/work/holosoma/logs/holosoma/20260305_g1_29dof_phuma_future_diverse_action_scale"
COMMON_ARGS="exp:g1-29dof-phuma-future-motion logger:wandb --logger.mode offline --logger.video.enabled False --training.num_envs 8192 --algo.config.eval_interval 1 --algo.config.save_interval 99999 --command.setup_terms.motion_command.params.motion_config.motion_dir=/home/nas4_user/kyungminlee/work/holosoma/g1_npz --command.setup_terms.motion_command.params.motion_config.split_file=/home/nas4_user/kyungminlee/work/holosoma/split/phuma_train.txt --command.setup_terms.motion_command.params.motion_config.pool_size 8192 robot:g1-29dof-diverse-action-scale"

source scripts/source_isaacsim_setup.sh

run_eval() {
    local MODEL_ITER=$1
    local THRESHOLD=$2
    local THRESHOLD_TAG=$3

    echo "============================================"
    echo ">>> model_${MODEL_ITER} / threshold=${THRESHOLD}m"
    echo "============================================"

    sed -i "s/^MOTION_FAR_THRESHOLD = .*/MOTION_FAR_THRESHOLD = ${THRESHOLD}/" "$CALLBACK_FILE"

    CUDA_VISIBLE_DEVICES=4 LOCAL_RANK=2 python src/holosoma/holosoma/train_agent.py \
        $COMMON_ARGS \
        --logger.project holosoma_test \
        --logger.name="eval_sr_${MODEL_ITER}_${THRESHOLD_TAG}" \
        --training.checkpoint="${CKPT_DIR}/model_${MODEL_ITER}.pt" \
        --algo.config.num_learning_iterations $((MODEL_ITER + 1))

    echo ">>> Done: model_${MODEL_ITER} / ${THRESHOLD}m"
}

# --- 1) model_13000, 0.5m ---
run_eval 13000 0.5 05m

# --- 2) model_13000, 0.25m ---
run_eval 13000 0.25 025m

# --- 3) model_17000, 0.5m ---
run_eval 17000 0.5 05m

# --- 4) model_17000, 0.25m ---
run_eval 17000 0.25 025m

# Restore threshold to default
sed -i 's/^MOTION_FAR_THRESHOLD = .*/MOTION_FAR_THRESHOLD = 0.25/' "$CALLBACK_FILE"

echo ">>> All 4 evaluations complete."
