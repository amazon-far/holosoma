#!/bin/bash
# Launch holosoma-thor-inference with keyboard velocity + keyboard state commands.
# Policy: g1-29dof-loco (blind locomotion).
#
# Usage: ./run_keyboard.sh [extra args for run_policy.py]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm inference \
    inference:g1-29dof-loco \
    --task.use-keyboard \
    --task.interface eth0 \
    "$@"
