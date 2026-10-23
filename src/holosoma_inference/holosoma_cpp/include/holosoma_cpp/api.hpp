// Public, transport-independent inference. Include this and nothing else:
//   Policy policy(preset_path, checkpoint_path);
//   const JointCommand& command = policy.step(state);
#pragma once

#include <cstdint>

#include "holosoma_cpp/general_wbt_policy.hpp"

namespace holosoma {

// Reads nothing, publishes nowhere, keeps what it was given.
class OfflineInterface final : public RobotInterface {
 public:
  bool read_state(RobotState& out) override {
    out = state_;
    return have_state_;
  }
  void send_command(const JointCommand& command) override { command_ = command; }
  std::optional<WirelessController> read_wireless_controller() override { return std::nullopt; }
  std::string description() const override { return "offline"; }
  void set_state(const RobotState& state) {
    state_ = state;
    have_state_ = true;
  }
  const JointCommand& command() const { return command_; }

 private:
  RobotState state_;
  JointCommand command_;
  bool have_state_ = false;
};

class Policy {
 public:
  Policy(const std::string& preset, const std::string& checkpoint = "");
  explicit Policy(InferenceConfig config, bool autostart = true);
  ~Policy();
  Policy(const Policy&) = delete;
  Policy& operator=(const Policy&) = delete;

  // Borrowed until replaced (nullptr clears it). Research policies reject this.
  // Replacing the buffer preserves history and heading anchors. When starting
  // an independent clip with a different heading, call reset() before stepping.
  void set_reference(const ReferenceWindow* window);
  // One tick; the returned command stays valid until the next step/destruction.
  const JointCommand& step(const RobotState& state);
  int num_dofs() const;
  const std::vector<std::string>& dof_names() const;
  const InferenceConfig& config() const;
  BasePolicy& policy();
  const BasePolicy& policy() const;

  // Composition hooks for transports that own engagement, gait and warm-up.
  std::vector<double> step(const std::vector<float>& actor_observation);
  std::vector<double> infer_offsets(const RobotState& state);
  JointCommand step_state(const RobotState& state, bool advance_phase = true);
  void set_state(const RobotState& state);
  void command(StateCommand command);
  void velocity(const VelCmd& command);
  // Times a command asked the operator velocity input to zero. A host that owns
  // that input (the Python runner) zeroes its own provider when this advances.
  std::uint64_t velocity_zero_requests() const;
  void update_phase();
  // Start a new episode: clear history/action feedback and reanchor BOTH robot
  // and reference heading plus motion time on the next step. Keeps the reference.
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace holosoma
