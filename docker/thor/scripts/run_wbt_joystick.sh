#!/bin/bash
# Launch holosoma-thor-inference with WBT (whole-body tracking) policy,
# joystick velocity + joystick state.
#
# Policy: g1-29dof-wbt.
#
# Usage: ./run_wbt_joystick.sh [extra args for run_policy.py]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm inference \
    inference:g1-29dof-wbt \
    --task.use-joystick \
    --task.interface eth0 \
    "$@"
