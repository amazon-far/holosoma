#!/bin/bash
set -e

MODEL_PATH="$1"
DATA_PATH="$2"
OUTPUT_PATH="$3"
SCENE_XML="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"

source scripts/source_isaacsim_setup.sh

if [ -d "$DATA_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
    mapfile -t DATA_FILES < <(find "$DATA_PATH" -type f -name "*.npz" | sort)
else
    DATA_FILES=( "$DATA_PATH" )
fi

REF_MOTION="$(dirname "$MODEL_PATH")/motion_data/ref_motion.npy"

for DATA_FILE in "${DATA_FILES[@]}"; do
    if [ -d "$DATA_PATH" ]; then
        MP4_OUT="$OUTPUT_PATH/$(basename "$(dirname "$DATA_FILE")").mp4"
    else
        MP4_OUT="$OUTPUT_PATH"
    fi

    python src/holosoma/holosoma/eval_agent.py \
        --checkpoint="$MODEL_PATH" \
        --terrain.terrain-term.obj-file-path="$DATA_FILE" \
        --terrain.terrain-term.tile-mesh False \
        --command.setup_terms.motion_command.params.motion_config.motion_file="$DATA_FILE" \
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
        --training.num_envs 1

    python visualize.py \
        --input "$REF_MOTION" \
        --output "$MP4_OUT" \
        --scene "$SCENE_XML" \
        --terrain_npz "$DATA_FILE" \
        --input_fps 50
done
