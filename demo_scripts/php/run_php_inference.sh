#!/usr/bin/env bash
# D435i depth-distillation (PHP) policy. Start run_php_sim.sh (or an on-robot depth server) first:
# this attaches to the shared-memory block the producer creates.
#
#   RUN=wandb://<entity>/<project>/<run_id> STEP=model_20000 demo_scripts/php/run_php_inference.sh
#
# A local checkpoint pair works too, pointing at the two files directly:
#   BACKBONE=/path/depth_backbone.onnx STUDENT=/path/student.onnx demo_scripts/php/run_php_inference.sh
#
# See src/holosoma_inference/docs/workflows/sim-to-sim-depth-distillation.md
set -e

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

source scripts/source_inference_setup.sh

RUN="${RUN:-}"
STEP="${STEP:-model_20000}"
BACKBONE="${BACKBONE:-}"
STUDENT="${STUDENT:-}"

if [ -n "$RUN" ]; then
    BACKBONE="${BACKBONE:-${RUN}/${STEP}/depth_backbone.onnx}"
    STUDENT="${STUDENT:-${RUN}/${STEP}/student.onnx}"
fi
if [ -z "$BACKBONE" ] || [ -z "$STUDENT" ]; then
    echo "set RUN=wandb://<entity>/<project>/<run_id> or both BACKBONE and STUDENT" >&2
    exit 2
fi

# Local paths are checked up front; wandb:// URIs are resolved (and cached) by the policy.
case "$BACKBONE" in
    wandb://* | https://*) ;;
    *) for f in "$BACKBONE" "$STUDENT"; do
           [ -f "$f" ] || { echo "missing checkpoint: $f" >&2; exit 1; }
       done ;;
esac

# Both paths go in ONE quoted Python list literal: model_path is typed `str | list[str]` and the CLI
# resolves that union to str, so space-separated paths arrive as a single token and fail (tyro then
# tries to build a parser for robot.unitree_legged_const and dies).
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation-d435i \
    --task.interface lo \
    --task.model-path "['${BACKBONE}','${STUDENT}']" \
    "$@"

# Expect on startup:
#   [DepthShmSensor] attached to 'depth_img_shm' shape=(1, 1, 58, 87)
#
# Controls (in THIS terminal):
#   ]        start the policy      o  damping mode (Kp=0, Kd>0)
#   i        re-enter stiff hold   c  toggle motion recording
#   HOLD w/s forward / back        HOLD a/d  45deg left / right
#   HOLD q/e 90deg left / right    z  stand
#   =        toggle speed mode (LOW <-> HIGH)
#
# Direction keys are MOMENTARY: the robot walks while a key is held and stands the moment you release
# it. Holding two directions stacks them, so releasing the newer one resumes the one still held.
#
# This needs true key-up events (pynput + an X11 DISPLAY). Without a DISPLAY the policy warns and
# falls back to LATCHING, where a press keeps the robot walking until another direction or 'z' —
# releasing the key does not stop it.
#
# Deploy sequence: in the MuJoCo window press 8 (~25x) to lower the gantry, then 9 to release it;
# the policy holds a stiff standing pose until ']' so the robot stays upright meanwhile.
#
# Run without a depth producer (control-loop smoke test, zero-filled depth):
#   demo_scripts/php/run_php_inference.sh --task.depth-shm.no-required
