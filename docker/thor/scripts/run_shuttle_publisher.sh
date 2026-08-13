#!/bin/bash
# Launch the shuttle velocity publisher INSIDE the inference-ros container.
# Publishes /cmd_vel with a 3s-forward, 3s-backward pattern at 0.5 m/s,
# which run_policy (with --task.velocity-input ros2) will subscribe to.
#
# Intended workflow:
#   Terminal 1:  ./run_ros2_joystick.sh           # starts the policy
#   Terminal 2:  ./run_shuttle_publisher.sh       # drives /cmd_vel
#
# Usage: ./run_shuttle_publisher.sh [extra args for ros2_velocity_publisher.py]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm --entrypoint='' inference-ros \
    bash -c "source /opt/ros/jazzy/setup.bash && \
      python3 -u /opt/holosoma-src/demo_scripts/ros2_velocity_publisher.py \
        --pattern shuttle \
        --other-topic holosoma/state_input \
        $*" \
    "$@"
