#!/bin/bash
# Evaluate PhUMA checkpoint success rate on validation set using MuJoCo (sim-to-sim).
# Uses the MuJoCo Warp backend for GPU-accelerated multi-env evaluation.
#
# Usage:
#   bash scripts/eval/eval_phuma_val_mujoco.sh

source scripts/source_mujoco_setup.sh

CHECKPOINT="./pretrained/phuma_g1_29dof/model_106000.pt"
MOTION_DIR="./g1_npz"
SPLIT_FILE="./split/phuma_val.txt"
NUM_ENVS=4096

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --num_envs="${NUM_ENVS}" \
    --simulator=mjwarp
