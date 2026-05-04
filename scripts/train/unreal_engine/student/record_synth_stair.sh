#!/bin/bash
# Headless single-stair inference for the RL-finetuned student, saved as an
# mp4. Uses one synthetic stair motion as the command + terrain. The
# resulting video shows the student (gray) on the stair with the synthetic
# motion target as the ghost (blue), so you can see how the student
# responds when the command is "go forward at 0.6 m/s on this stair".
#
# Override SYNTH_NPZ to point at a different single .npz (must be in a
# directory by itself, since rollout_single_policy uses os.walk).
#
#   SYNTH_NPZ=src/.../synth_stairs_h010-030_w030-080/synth_h25_w52_n09_up_i000.npz \
#   bash scripts/train/unreal_engine/student/record_synth_stair.sh

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260501_g1_unreal_student_RLfinetune/model_27000.pt}"
SYNTH_NPZ="${SYNTH_NPZ:-/home/kyungminlee/work/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/synth_stairs_h010-030_w030-080/synth_h10_w71_n12_up_i001.npz}"
RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"
MOTION_NAME="$(basename "$SYNTH_NPZ" .npz)"

OUTPUT_DIR="${OUTPUT_DIR:-videos/synth_record_${RUN_DIR}_${CKPT_NAME}}"
# Derive WORK_DIR from OUTPUT_DIR's basename, not just MOTION_NAME, so two
# parallel runs that happen to point at .npz files with the same filename
# (different content, e.g. standing-init vs walking-init) get separate
# scratch dirs. Otherwise they race to write the same solo_motion/ + rollouts/.
OUTPUT_TAG="$(basename "$OUTPUT_DIR")"
WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/${OUTPUT_TAG}_${MOTION_NAME}}"
SOLO_DIR="${WORK_DIR}/solo_motion"
ROLLOUT_DIR="${WORK_DIR}/rollout"
mkdir -p "$WORK_DIR" "$SOLO_DIR" "$ROLLOUT_DIR" "$OUTPUT_DIR"

# Place a copy of the chosen .npz alone in SOLO_DIR so MotionLibrary picks
# only this one motion (avoids needing pool_size overrides).
rm -f "$SOLO_DIR"/*.npz
cp "$SYNTH_NPZ" "$SOLO_DIR/"

# Train-level GPU buffers (single tile but full-res mesh).
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 30))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 26))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 22))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 20))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 19))

# --- Phase 1: roll out the single motion (headless) ---
python scripts/eval/unreal_engine/rollout_single_policy.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$SOLO_DIR" \
    --out_dir="$ROLLOUT_DIR"

POLICY_NPY="$ROLLOUT_DIR/policy_${MOTION_NAME}.npy"
if [ ! -f "$POLICY_NPY" ]; then
    echo "Rollout npy missing: $POLICY_NPY" >&2
    exit 1
fi

# --- Phase 2: render mp4 (student=main, motion .npz=ghost, terrain from mesh) ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
OUT_MP4="$OUTPUT_DIR/${MOTION_NAME}.mp4"
"$VIZ_PY" visualize.py \
    --input "$POLICY_NPY" \
    --ref_motion "$SYNTH_NPZ" \
    --terrain_npz "$SYNTH_NPZ" \
    --output "$OUT_MP4" \
    --ghost_alpha 0.45 \
    --fps 30

echo "Done. Saved $OUT_MP4"
