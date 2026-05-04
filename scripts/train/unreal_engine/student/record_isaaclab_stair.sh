#!/bin/bash
# Headless single-tile inference for the xy_yaw RL-finetune student on a
# pyramid stair tile generated with IsaacLab's MeshPyramidStairsTerrainCfg
# (the same geometry PHUMA2's STAIRS_TERRAINS_CFG builds on). Mirrors
# record_synth_stair.sh but swaps the linear synth stair for an IsaacLab
# pyramid / inverted-pyramid stair tile.
#
# Defaults to model_15000 of the xy_yaw run (= end of BC→PPO ramp, so
# the actor is the freshly-finetuned narrow-command policy).
#
# Override anything via env vars, e.g.:
#   KIND=pyramid STEP_HEIGHT=0.20 bash scripts/train/unreal_engine/student/record_isaaclab_stair.sh

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260503_g1_unreal_student_RLfinetune_xy_yaw/model_15000.pt}"
KIND="${KIND:-inv_pyramid}"          # pyramid (descend) | inv_pyramid (ascend)
STEP_HEIGHT="${STEP_HEIGHT:-0.15}"
STEP_WIDTH="${STEP_WIDTH:-0.50}"
LIN_VEL_X="${LIN_VEL_X:-0.6}"
TILE_SIZE="${TILE_SIZE:-8.0}"
PLATFORM_WIDTH="${PLATFORM_WIDTH:-3.0}"
MOTION_SECONDS="${MOTION_SECONDS:-8.0}"

# Use awk for clean integer-cm formatting in the basename without bash math.
TAG=$(awk -v k="$KIND" -v h="$STEP_HEIGHT" -v w="$STEP_WIDTH" -v v="$LIN_VEL_X" \
       'BEGIN{printf "isaaclab_%s_h%03d_w%03d_v%03d", k, h*100+0.5, w*100+0.5, v*100+0.5}')

RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"

OUTPUT_DIR="${OUTPUT_DIR:-videos/isaaclab_${RUN_DIR}_${CKPT_NAME}}"
WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/isaaclab_${RUN_DIR}_${CKPT_NAME}_${TAG}}"
SOLO_DIR="${WORK_DIR}/solo_motion"
ROLLOUT_DIR="${WORK_DIR}/rollout"
mkdir -p "$WORK_DIR" "$SOLO_DIR" "$ROLLOUT_DIR" "$OUTPUT_DIR"

# --- Phase 0: build the IsaacLab pyramid-stair tile + matching .npz ---
# Run with the hsmujoco env (mujoco + trimesh only; no IsaacSim runtime
# needed because the IsaacLab geometry functions are inlined in the
# generator script).
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

SYNTH_NPZ="$SOLO_DIR/${TAG}.npz"
if [ ! -f "$SYNTH_NPZ" ]; then
    echo "Generator did not produce $SYNTH_NPZ" >&2
    exit 1
fi

# Default to CPU PhysX so this script coexists with a train run that is
# already hogging the GPU. Single tile + 1 env is small enough that CPU
# rollout finishes in a couple of minutes. Override DEVICE=cuda:0 (and
# bump the buffer env vars) if you have a free GPU.
DEVICE="${DEVICE:-cpu}"
export HOLOSOMA_GPU_COLLISION_STACK_SIZE="${HOLOSOMA_GPU_COLLISION_STACK_SIZE:-$((1 << 25))}"     # 32 MB
export HOLOSOMA_GPU_HEAP_CAPACITY="${HOLOSOMA_GPU_HEAP_CAPACITY:-$((1 << 24))}"                  # 16 MB
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT="${HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT:-$((1 << 17))}"
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY="${HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY:-$((1 << 16))}"
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT="${HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT:-$((1 << 15))}"

# --- Phase 1: roll out the single motion (headless) ---
python scripts/eval/unreal_engine/rollout_single_policy.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$SOLO_DIR" \
    --out_dir="$ROLLOUT_DIR" \
    --device="$DEVICE"

POLICY_NPY="$ROLLOUT_DIR/policy_${TAG}.npy"
if [ ! -f "$POLICY_NPY" ]; then
    echo "Rollout npy missing: $POLICY_NPY" >&2
    exit 1
fi

# --- Phase 2: render mp4 (student=main, motion .npz=ghost, terrain from mesh) ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
OUT_MP4="$OUTPUT_DIR/${TAG}.mp4"
"$VIZ_PY" visualize.py \
    --input "$POLICY_NPY" \
    --ref_motion "$SYNTH_NPZ" \
    --terrain_npz "$SYNTH_NPZ" \
    --output "$OUT_MP4" \
    --ghost_alpha 0.45 \
    --fps 30

echo "Done. Saved $OUT_MP4"
