#!/bin/bash
# Live (headless=False) IsaacSim playback of the xy_yaw RL-finetune student
# on a single IsaacLab pyramid / inverted-pyramid stair tile. Reuses the
# generator from record_isaaclab_stair.sh — generates the tile + motion
# .npz first, then opens the IsaacSim window so you can watch the policy
# walk on the stair in real time.
#
# Override knobs (env vars):
#   CHECKPOINT     full path to .pt
#   KIND           pyramid | inv_pyramid
#   STEP_HEIGHT, STEP_WIDTH, LIN_VEL_X, MOTION_SECONDS
#   DEVICE         cpu (default — coexists with a train run) | cuda:0

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260503_g1_unreal_student_RLfinetune_xy_yaw/model_15000.pt}"
KIND="${KIND:-inv_pyramid}"
STEP_HEIGHT="${STEP_HEIGHT:-0.15}"
STEP_WIDTH="${STEP_WIDTH:-0.50}"
LIN_VEL_X="${LIN_VEL_X:-0.6}"
TILE_SIZE="${TILE_SIZE:-8.0}"
PLATFORM_WIDTH="${PLATFORM_WIDTH:-3.0}"
MOTION_SECONDS="${MOTION_SECONDS:-8.0}"
DEVICE="${DEVICE:-cpu}"
# eval_agent.py auto-picks "cuda:0 if available else cpu" and doesn't expose
# a --device flag, so we hide the GPU when DEVICE=cpu — same approach
# PHUMA2's play scripts use to coexist with a training run.
if [ "$DEVICE" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

TAG=$(awk -v k="$KIND" -v h="$STEP_HEIGHT" -v w="$STEP_WIDTH" -v v="$LIN_VEL_X" \
       'BEGIN{printf "isaaclab_%s_h%03d_w%03d_v%03d", k, h*100+0.5, w*100+0.5, v*100+0.5}')

WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/eval_${TAG}}"
SOLO_DIR="${WORK_DIR}/solo_motion"
mkdir -p "$WORK_DIR" "$SOLO_DIR"

# --- Phase 0: build the tile + motion .npz (same generator as record_*) ---
GEN_PY="${GEN_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
rm -f "$SOLO_DIR"/*.npz
"$GEN_PY" scripts/eval/unreal_engine/generate_isaaclab_stair_motion.py \
    --out_dir "$SOLO_DIR" \
    --out_basename "$TAG" \
    --kind "$KIND" \
    --step_height "$STEP_HEIGHT" \
    --step_width "$STEP_WIDTH" \
    --tile_size "$TILE_SIZE" \
    --platform_width "$PLATFORM_WIDTH" \
    --lin_vel_x "$LIN_VEL_X" \
    --motion_seconds "$MOTION_SECONDS"

# Sandbox-friendly PhysX buffers so this can coexist with a train run.
export HOLOSOMA_GPU_COLLISION_STACK_SIZE="${HOLOSOMA_GPU_COLLISION_STACK_SIZE:-$((1 << 25))}"
export HOLOSOMA_GPU_HEAP_CAPACITY="${HOLOSOMA_GPU_HEAP_CAPACITY:-$((1 << 24))}"
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT="${HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT:-$((1 << 17))}"
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY="${HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY:-$((1 << 16))}"
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT="${HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT:-$((1 << 15))}"

# --- Phase 1: live IsaacSim eval (headless=False so the window pops up) ---
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint="$CHECKPOINT" \
    --terrain.terrain-term.obj-dir="$SOLO_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.obj-max-tiles 1 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$SOLO_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 1 \
    --command.setup_terms.motion_command.params.motion_config.pool_size 1 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs 1 \
    --training.headless False \
    --simulator.config.scene.env_spacing=0.0
