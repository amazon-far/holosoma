#!/bin/bash
# Launch holosoma-thor-inference-ros with /cmd_vel from ROS + state from joystick.
# Policy: g1-29dof-loco.
#
# Use this when a ROS publisher is driving velocity and the joystick is
# used only for Walking/Standing state transitions.
#
# Usage: ./run_ros2_joystick.sh [extra args for run_policy.py]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm inference-ros \
    inference:g1-29dof-loco \
    --task.velocity-input ros2 \
    --task.state-input interface \
    --task.interface eth0 \
    "$@"
