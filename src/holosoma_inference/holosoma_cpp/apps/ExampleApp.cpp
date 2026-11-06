// Stepping a policy at 50 Hz with no robot attached.
//
//   ./ExampleApp <preset.yaml> [checkpoint.onnx]
//
// The host owns two things: the state it feeds in, and the clock. holosoma reads no clock and
// sleeps nowhere, so the loop below is the whole integration.

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "holosoma_cpp/api.hpp"

using namespace holosoma;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: ExampleApp <preset.yaml> [model.onnx]\n"), 2;

  Policy policy(argv[1], argc > 2 ? argv[2] : "");
  const int n = policy.num_dofs();

  // An actor-only export takes its reference from the host; one with a baked clip refuses this.
  // Your clip or retargeter fills the five future offsets (0, 2, 4, 6, 8 frames) each tick.
  ReferenceWindow window;
  window.q.assign(static_cast<size_t>(kNumOffsets * n), 0.0);
  window.dq.assign(static_cast<size_t>(kNumOffsets * n), 0.0);
  window.quat.assign(kNumOffsets, {1.0, 0.0, 0.0, 0.0});
  try {
    policy.set_reference(&window);
  } catch (const std::runtime_error&) {
    // This preset's policy carries its own motion.
  }

  // Your subscriber fills this; synthetic here so the example needs no robot.
  RobotState state;
  state.base_quat_wxyz = {1.0, 0.0, 0.0, 0.0};
  state.dof_pos.assign(static_cast<size_t>(n), 0.0);
  state.dof_vel.assign(static_cast<size_t>(n), 0.0);

  auto deadline = Clock::now();
  for (int tick = 0; tick < 100; ++tick) {
    const JointCommand& command = policy.step(state);
    if (tick + 1 == 100) {
      std::printf("%zu joints, q[0]=%+.4f kp[0]=%.1f kd[0]=%.2f\n", command.q.size(), command.q[0], command.kp[0],
                  command.kd[0]);
    }
    deadline += std::chrono::microseconds(20000);  // 50 Hz
    std::this_thread::sleep_until(deadline);
  }
  return 0;
}
