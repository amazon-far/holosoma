#!/usr/bin/env bash
# Depth-distillation stair policy. Start run_stair_sim.sh (or an on-robot depth
# server) first: this attaches to the shared-memory block the producer creates.
#
# Point at a checkpoint pair with:
#   CKPT_DIR=/path/to/ckpts RUN=<run-name> demo_scripts/php/run_stair_inference.sh
# expecting <RUN>_depth_backbone.onnx and <RUN>_student.onnx inside CKPT_DIR.
#
# See src/holosoma_inference/docs/workflows/sim-to-sim-depth-distillation.md
set -e

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

source scripts/source_inference_setup.sh

: "${CKPT_DIR:?set CKPT_DIR to the directory holding the checkpoint pair}"
: "${RUN:?set RUN to the checkpoint run name (without the _depth_backbone/_student suffix)}"

BACKBONE="${CKPT_DIR}/${RUN}_depth_backbone.onnx"
STUDENT="${CKPT_DIR}/${RUN}_student.onnx"
for f in "$BACKBONE" "$STUDENT"; do
    [ -f "$f" ] || { echo "missing checkpoint: $f" >&2; exit 1; }
done

# Both paths go in ONE quoted Python list literal: model_path is typed
# `str | list[str]` and the CLI resolves that union to str, so space-separated
# paths arrive as a single token and fail.
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation \
    --task.interface lo \
    --task.model-path "['${BACKBONE}','${STUDENT}']" \
    --task.record-dir stair \
    --task.record-label stair \
    "$@"

# Controls (in THIS terminal):
#   ]        start the policy      o  damping mode (Kp=0, Kd>0)
#   i        re-enter stiff hold   c  toggle motion recording
#   w/s      forward / back        a/d  45deg left / right
#   q/e      90deg left / right    z  stand
#   =        toggle speed mode (LOW <-> HIGH)
#
# Recording writes recorded_motion/stair/stair_duration<X.X>s_motion.npz
# (qpos [N, 36] quat-first, vel_cmd [N, 15] one-hot, fps), relative to cwd.
#
# Run without a depth producer (control-loop smoke test, zero-filled depth):
#   demo_scripts/php/run_stair_inference.sh --task.depth-shm.no-required
