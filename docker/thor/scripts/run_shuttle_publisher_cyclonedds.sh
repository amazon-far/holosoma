#!/bin/bash
# Same as run_shuttle_publisher.sh but forces CycloneDDS for the publisher
# while the policy container runs rclpy on FastDDS (image default). This
# validates cross-vendor DDS interop for /cmd_vel TwistStamped delivery —
# useful for anyone bringing a CycloneDDS-based ROS publisher to a policy
# container running on FastDDS.
#
# Usage: ./run_shuttle_publisher_cyclonedds.sh [extra args]
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

docker compose -f compose.yaml run --rm --entrypoint='' \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    inference-ros \
    bash -c "source /opt/ros/jazzy/setup.bash && \
      python3 -u /opt/holosoma-src/demo_scripts/ros2_velocity_publisher.py \
        --pattern shuttle \
        --other-topic holosoma/state_input \
        $*" \
    "$@"
