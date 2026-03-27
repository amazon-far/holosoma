#!/bin/bash
# Re-export ONNX model with dynamic batch size.
#
# The original ONNX was exported with fixed batch=8192 from training,
# causing inference to pad every input to 8192 (extremely slow).
# This re-exports with dynamic_axes so inference runs at batch=1.
#
# Usage:
#   bash scripts/eval/reexport_onnx.sh

source scripts/source_mujoco_setup.sh

CHECKPOINT="./pretrained/phuma_g1_29dof/model_106000.pt"
MOTION_DIR="./g1_npz"
SPLIT_FILE="./split/phuma_val.txt"

python src/holosoma/holosoma/reexport_onnx.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --simulator=mjwarp
