#!/bin/bash
# Render 10 success + 10 failure per-motion mp4s for the RL-finetuned student
# at iter 27000 (20260501 run).
#
# Pipeline (same shape as the videomimic eval-and-render driver, adapted
# for the flat .npz layout of Unreal_Engine_stair_subset200_aligned):
#   1. eval_success_rate.py — full 200-motion eval, writes per-motion OK/FAIL.
#   2. Sample 10 OK + 10 FAIL (0.5m threshold), symlink them into a filtered
#      motion dir.
#   3. rollout_single_policy.py — deterministic per-env rollout, saves
#      policy_<motion>.npy per motion.
#   4. visualize.py — policy as main, motion .npz directly as ghost.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260501_g1_unreal_student_RLfinetune/model_27000.pt"
RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"

WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/${RUN_DIR}_${CKPT_NAME}}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_DIR}/filtered_motions}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/${RUN_DIR}_${CKPT_NAME}}"
mkdir -p "$WORK_DIR" "$FILTERED_DIR" "$OUTPUT_DIR"

# Sandbox-friendly PhysX buffers for eval (coexists with a running training
# job on the same GPU).
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 25))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 24))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 17))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 16))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 15))

# Unreal_Engine_stair_subset200_aligned has 200 flat .npz files; the saved
# config uses pool_size=8192, so override max_motions/pool_size to 200.
N_MOTIONS=$(find "$MOTION_DIR" -maxdepth 1 -name "*.npz" | wc -l)
echo "Discovered $N_MOTIONS motions in $MOTION_DIR"

# --- Phase 1: success rate eval ---
if [ ! -f "$WORK_DIR/per_motion.tsv" ]; then
    EVAL_LOG="$WORK_DIR/eval.log"
    echo "Running eval_success_rate.py (log: $EVAL_LOG)"
    python src/holosoma/holosoma/eval_success_rate.py \
        --checkpoint="$CHECKPOINT" \
        --motion_dir="$MOTION_DIR" \
        --max_motions="$N_MOTIONS" \
        --num_envs="$N_MOTIONS" > "$EVAL_LOG" 2>&1

    SR_TXT=$(grep -oE "/[^ ]+/success_rate_iter[0-9]+\.txt" "$EVAL_LOG" | tail -1)
    if [ -z "$SR_TXT" ] || [ ! -f "$SR_TXT" ]; then
        echo "Failed to find success_rate report in eval log:" >&2
        tail -20 "$EVAL_LOG" >&2
        exit 1
    fi
    echo "Eval report: $SR_TXT"

    awk '
        /^--- Per-Motion \[0\.5m\]/ { in_block=1; next }
        /^--- / && !/0\.5m/ { in_block=0 }
        in_block && /^  OK:/ { sub(/^  OK: /, ""); print "OK\t" $0 }
        in_block && /^  FAIL:/ { sub(/^  FAIL: /, ""); print "FAIL\t" $0 }
    ' "$SR_TXT" > "$WORK_DIR/per_motion.tsv"
fi

OK_COUNT=$(grep -c '^OK' "$WORK_DIR/per_motion.tsv")
FAIL_COUNT=$(grep -c '^FAIL' "$WORK_DIR/per_motion.tsv")
echo "Per-motion: $OK_COUNT OK, $FAIL_COUNT FAIL"

# --- Phase 2: sample 10 + 10 and symlink into filtered_motions ---
# Unreal_Engine motions are flat .npz files; rollout_single_policy uses
# os.walk(followlinks=True), and the terrain manager / motion library both
# rglob *.npz which works fine when the symlink target is the file itself.
if [ -z "$(ls -A "$FILTERED_DIR" 2>/dev/null)" ]; then
    grep '^OK' "$WORK_DIR/per_motion.tsv" | shuf -n 10 --random-source="$WORK_DIR/per_motion.tsv" \
        > "$WORK_DIR/selected_ok.tsv"
    grep '^FAIL' "$WORK_DIR/per_motion.tsv" | shuf -n 10 --random-source="$WORK_DIR/per_motion.tsv" \
        > "$WORK_DIR/selected_fail.tsv"

    i=0
    while IFS=$'\t' read -r status path; do
        name=$(basename "$path")
        label=$(echo "$status" | tr '[:upper:]' '[:lower:]')
        ln -sfT "/home/kyungminlee/work/holosoma/$path" "$FILTERED_DIR/$(printf '%02d' $i)_${label}_${name}"
        i=$((i+1))
    done < <(cat "$WORK_DIR/selected_ok.tsv" "$WORK_DIR/selected_fail.tsv")
fi
echo "Filtered dir has $(ls "$FILTERED_DIR" | wc -l) motions"

# --- Phase 3: rollout the 20 motions in one IsaacSim session ---
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 30))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 26))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 22))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 20))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 19))

ROLLOUT_DIR="${WORK_DIR}/rollouts"
mkdir -p "$ROLLOUT_DIR"
python scripts/eval/unreal_engine/rollout_single_policy.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$FILTERED_DIR" \
    --out_dir="$ROLLOUT_DIR"

# --- Phase 4: render with visualize.py — policy as main, motion .npz as ghost ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
for npz_link in "$FILTERED_DIR"/*.npz; do
    base=$(basename "$npz_link" .npz)
    policy_npy="$ROLLOUT_DIR/policy_${base}.npy"
    npz_target=$(readlink -f "$npz_link")
    out_mp4="$OUTPUT_DIR/${base}.mp4"

    if [ ! -f "$policy_npy" ] || [ ! -f "$npz_target" ]; then
        echo "Skipping $base (missing inputs)"
        continue
    fi

    "$VIZ_PY" visualize.py \
        --input "$policy_npy" \
        --ref_motion "$npz_target" \
        --terrain_npz "$npz_target" \
        --output "$out_mp4" \
        --ghost_alpha 0.45 \
        --fps 30
done

echo "Done. mp4s in $OUTPUT_DIR"
