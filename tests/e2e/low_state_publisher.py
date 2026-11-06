"""Publish a standing G1 low state on the loopback interface, the way the simulator bridge does.

The policy E2E needs a robot to talk to. This stands in for one without a simulator: the same
SDK binding the MuJoCo bridge uses publishes a fixed upright state (with its checksum) at a fixed
rate, so run_policy discovers a valid state and its native inference runs on real input.
"""

from __future__ import annotations

import argparse
import json
import time

import unitree_interface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-q", required=True, help="JSON list of motor-ordered joint positions")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--rate", type=float, default=200.0)
    parser.add_argument(
        "--invalid-after",
        type=float,
        default=None,
        help="Seconds after which to publish a zero orientation (a failed-input control)",
    )
    args = parser.parse_args()

    robot = unitree_interface.UnitreeInterface(
        args.interface, unitree_interface.RobotType.G1, unitree_interface.MessageType.HG
    )
    motor_q = json.loads(args.motor_q)
    state = unitree_interface.LowState(robot.get_num_motors())
    q = list(state.motor.q)
    q[: len(motor_q)] = motor_q
    state.motor.q = q
    state.imu.quat = [1.0, 0.0, 0.0, 0.0]
    period = 1.0 / args.rate
    start = time.monotonic()
    print("low-state publisher running", flush=True)
    while True:
        elapsed = time.monotonic() - start
        if args.invalid_after is not None and elapsed >= args.invalid_after:
            state.imu.quat = [0.0, 0.0, 0.0, 0.0]
        state.tick = int(elapsed * 1e3)
        robot.publish_low_state(state)
        time.sleep(period)


if __name__ == "__main__":
    main()
