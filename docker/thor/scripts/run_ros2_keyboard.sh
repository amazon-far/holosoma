#!/bin/bash
# Launch holosoma-thor-inference-ros with /cmd_vel from ROS + state from keyboard.
# Policy: g1-29dof-loco.
#
# Usage: ./run_ros2_keyboard.sh [extra args for run_policy.py]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm inference-ros \
    inference:g1-29dof-loco \
    --task.velocity-input ros2 \
    --task.state-input keyboard \
    --task.interface eth0 \
    "$@"
