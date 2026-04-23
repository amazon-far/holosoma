#!/bin/bash
# Evaluate checkpoint success rate on the PhUMA validation set.
# Must be run from the kyungminn_phuma branch (which has MultiMotionCommand).
#
# Usage:
#   git stash  # if needed
#   git checkout kyungminn_phuma
#   bash scripts/eval/eval_phuma_val.sh

source scripts/source_isaacsim_setup.sh

CHECKPOINT="./pretrained/phuma_g1_29dof/model_106000.pt"
MOTION_DIR="./g1_npz"
SPLIT_FILE="./split/phuma_val.txt"
NUM_ENVS=4096

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --num_envs="${NUM_ENVS}"
