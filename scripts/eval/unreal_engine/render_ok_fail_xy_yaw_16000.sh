#!/bin/bash
# 20 success + 10 fail per-motion mp4s for the xy_yaw RL-finetune student
# at iter 16000. Same pipeline as render_ok_fail_rl_finetune_27000.sh
# (eval over the 200 train motions, sample by 0.5m ASAP threshold,
# rollout the chosen subset, render each as policy=main + motion=ghost),
# but adapted for the new checkpoint and the requested 20/10 split.
#
# Forces CPU PhysX (CUDA_VISIBLE_DEVICES="") so this can coexist with the
# training run that is still hogging the GPU.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

# Hide the GPU from PyTorch / IsaacSim — eval_success_rate.py and
# rollout_single_policy.py both auto-pick "cuda:0 if available else cpu".
export CUDA_VISIBLE_DEVICES=""

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260503_g1_unreal_student_RLfinetune_xy_yaw/model_16000.pt}"
RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"

WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/${RUN_DIR}_${CKPT_NAME}}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_DIR}/filtered_motions}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/${RUN_DIR}_${CKPT_NAME}}"
mkdir -p "$WORK_DIR" "$FILTERED_DIR" "$OUTPUT_DIR"

# Sandbox-friendly PhysX buffers — small, leaves headroom for the train
# run on the same GPU (and CPU PhysX ignores most of these anyway).
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 25))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 24))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 17))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 16))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 15))

N_MOTIONS=$(find "$MOTION_DIR" -maxdepth 1 -name "*.npz" | wc -l)
echo "Discovered $N_MOTIONS motions in $MOTION_DIR"

# --- Phase 1: success rate eval over all 200 motions ---
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

# --- Phase 2: sample 20 OK + 10 FAIL ---
if [ -z "$(ls -A "$FILTERED_DIR" 2>/dev/null)" ]; then
    grep '^OK' "$WORK_DIR/per_motion.tsv" | shuf -n 20 --random-source="$WORK_DIR/per_motion.tsv" \
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

# --- Phase 3: rollout the 30 motions in one IsaacSim session ---
ROLLOUT_DIR="${WORK_DIR}/rollouts"
mkdir -p "$ROLLOUT_DIR"
python scripts/eval/unreal_engine/rollout_single_policy.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$FILTERED_DIR" \
    --out_dir="$ROLLOUT_DIR" \
    --device=cpu

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
