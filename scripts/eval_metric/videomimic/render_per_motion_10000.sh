#!/bin/bash
# Render one mp4 per motion for the videomimic multi-terrain checkpoint.
# Output: videos/model_10000/<motion_name>.mp4  (one file per .npz motion)

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260422_g1_videomimic_multi_terrain_future_motion/model_10000.pt"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"
OUTPUT_DIR="videos/model_10000"

# Full-resolution meshes (obj_max_faces_per_tile=0) + 130 tiles need more
# collision-stack headroom than the ultra-shrunk SR-eval config. Train-level
# buffers are fine with 27 GB free.
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 30))        # 1 GB
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 26))               # 64 MB
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 22))     # 4 M
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 20))   # 1 M
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 19))       # 512 K

python src/holosoma/holosoma/render_per_motion.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$MOTION_DIR" \
    --output_dir="$OUTPUT_DIR"
