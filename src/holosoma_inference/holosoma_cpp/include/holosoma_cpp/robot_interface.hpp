// Hardware abstraction shared by the Unitree SDK2 backend and the test doubles.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace holosoma {

using SteadyClock = std::chrono::steady_clock;

// Robot state in joint order (the Python runtime's flat
// [base_pos, quat, dof_pos, base_lin_vel, base_ang_vel, dof_vel] array).
struct RobotState {
  std::vector<double> base_pos = std::vector<double>(3, 0.0);
  std::vector<double> base_quat_wxyz{1.0, 0.0, 0.0, 0.0};
  std::vector<double> dof_pos;
  std::vector<double> base_lin_vel = std::vector<double>(3, 0.0);
  std::vector<double> base_ang_vel = std::vector<double>(3, 0.0);
  std::vector<double> dof_vel;

  uint32_t tick_ms = 0;             // LowState.tick; the simulator bridge writes sim time here.
  uint64_t sequence = 0;            // Increments for every accepted LowState message.
  SteadyClock::time_point stamp{};  // Local receive time.
};

// Joint-order PD command. Gains are final (already multiplied by kp/kd level).
struct JointCommand {
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<double> tau;
  std::vector<double> kp;
  std::vector<double> kd;
};

// Unitree wireless controller sample.
struct WirelessController {
  float lx = 0.0f;
  float ly = 0.0f;
  float rx = 0.0f;
  float ry = 0.0f;
  uint16_t keys = 0;
};

class RobotInterface {
 public:
  virtual ~RobotInterface() = default;

  // Copies the latest state. Returns false until the first state arrives.
  virtual bool read_state(RobotState& state) = 0;

  // Publishes a command. The backend keeps re-sending the latest command at its own rate.
  virtual void send_command(const JointCommand& command) = 0;

  // Latest wireless controller sample, or nullopt if none has been received.
  virtual std::optional<WirelessController> read_wireless_controller() = 0;

  virtual std::string description() const = 0;
};

}  // namespace holosoma
