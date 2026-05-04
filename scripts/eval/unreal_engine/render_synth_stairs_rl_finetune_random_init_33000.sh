#!/bin/bash
# Run the RL-finetuned student (model_33000 of 20260502_..._random_init) on the
# 100 synthetic stair tiles at h ∈ [0.10, 0.30] m, w ∈ [0.30, 0.80] m. Same
# pipeline as render_synth_stairs_rl_finetune_27000.sh, just pointed at the
# random-init run / iter 33000 checkpoint.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260502_g1_unreal_student_RLfinetune_random_init/model_33000.pt"
RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/synth_stairs_h010-030_w030-080"

WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/synth_${RUN_DIR}_${CKPT_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${WORK_DIR}/rollouts}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/synth_stairs_${RUN_DIR}_${CKPT_NAME}}"
LABELS_JSON="${WORK_DIR}/reach_labels.json"
mkdir -p "$WORK_DIR" "$ROLLOUT_DIR" "$OUTPUT_DIR"

# rollout_single_policy needs the train-level GPU buffers (full-res mesh +
# 100 tiles).
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 30))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 26))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 22))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 20))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 19))

N_MOTIONS=$(find "$MOTION_DIR" -maxdepth 1 -name "*.npz" | wc -l)
echo "Discovered $N_MOTIONS synthetic stair motions in $MOTION_DIR"

# --- Phase 1: rollout all 100 motions (deterministic env_i ↔ motion_i ↔ tile_i) ---
if [ -z "$(ls -A "$ROLLOUT_DIR" 2>/dev/null | grep -F .npy)" ]; then
    python scripts/eval/unreal_engine/rollout_single_policy.py \
        --checkpoint="$CHECKPOINT" \
        --motion_dir="$MOTION_DIR" \
        --out_dir="$ROLLOUT_DIR"
fi

# --- Phase 2: relabel by reach-target ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
"$VIZ_PY" scripts/eval/unreal_engine/relabel_reach_target.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --motion_dir "$MOTION_DIR" \
    --out "$LABELS_JSON"

N_SUCCESS=$("$VIZ_PY" -c "import json; d=json.load(open('$LABELS_JSON')); print(sum(1 for v in d.values() if v['label']=='success'))")
N_FAIL=$("$VIZ_PY" -c "import json; d=json.load(open('$LABELS_JSON')); print(sum(1 for v in d.values() if v['label']=='fail'))")
echo "Reach labels: $N_SUCCESS success, $N_FAIL fail (of $N_MOTIONS)"

# --- Phase 3: sample 10 success + 10 fail, render ---
"$VIZ_PY" - <<PYEOF > "$WORK_DIR/sample_list.txt"
import json, random
labels = json.load(open("$LABELS_JSON"))
random.seed(0)
success = sorted([k for k, v in labels.items() if v["label"] == "success"])
fail = sorted([k for k, v in labels.items() if v["label"] == "fail"])
random.shuffle(success); random.shuffle(fail)
for i, k in enumerate(success[:10]):
    print(f"success\t{i:02d}\t{k}")
for i, k in enumerate(fail[:10]):
    print(f"fail\t{i+10:02d}\t{k}")
PYEOF

while IFS=$'\t' read -r label idx motion; do
    policy_npy="$ROLLOUT_DIR/policy_${motion}.npy"
    npz_path="$MOTION_DIR/${motion}.npz"
    out_mp4="$OUTPUT_DIR/${idx}_${label}_${motion}.mp4"

    if [ ! -f "$policy_npy" ] || [ ! -f "$npz_path" ]; then
        echo "Skipping $motion (missing inputs)"
        continue
    fi

    "$VIZ_PY" visualize.py \
        --input "$policy_npy" \
        --ref_motion "$npz_path" \
        --terrain_npz "$npz_path" \
        --output "$out_mp4" \
        --ghost_alpha 0.45 \
        --fps 30
done < "$WORK_DIR/sample_list.txt"

# --- Phase 4: analysis plots ---
"$VIZ_PY" scripts/eval/unreal_engine/synth_stairs_analysis_reach.py \
    --labels "$LABELS_JSON" \
    --out_dir "$OUTPUT_DIR/analysis"

echo "Done. mp4s in $OUTPUT_DIR, analysis in $OUTPUT_DIR/analysis"
