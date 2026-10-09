// Control loop: one or two policies sharing the robot and inputs
// (BasePolicy.run and policies/dual_mode.py DualModePolicy), plus the safety
// layer the C++ runtime adds around it.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "holosoma_cpp/config.hpp"
#include "holosoma_cpp/policy.hpp"

namespace holosoma {

// Creates the policy class the Python runtime would pick for `config`
// (motion_command in actor_obs -> whole-body tracking, otherwise locomotion).
std::unique_ptr<BasePolicy> make_policy(const InferenceConfig& config, PolicyContext& context);

// Owns the primary and optional secondary policy and routes commands to the active one.
class PolicyController {
 public:
  PolicyController(const InferenceConfig& config, PolicyContext& context);

  BasePolicy& active() { return *active_; }
  const BasePolicy& active() const { return *active_; }
  BasePolicy& primary() { return *primary_; }
  BasePolicy* secondary() { return secondary_.get(); }
  bool dual_mode() const { return secondary_ != nullptr; }
  bool active_is_primary() const { return active_ == primary_.get(); }

  void dispatch(StateCommand command);
  void switch_mode();

 private:
  std::unique_ptr<BasePolicy> primary_;
  std::unique_ptr<BasePolicy> secondary_;
  BasePolicy* active_ = nullptr;
};

// Outcome of one loop iteration.
enum class StepResult { kContinue, kKilled, kFault };

class Runner {
 public:
  Runner(PolicyController& controller, PolicyContext& context, const InferenceConfig& config);

  // One iteration of the Python run loop (inputs, commands, phase, control tick). Never sleeps.
  // Faults (stale or invalid state, a failed inference, a non-finite command) latch the damping mode.
  StepResult step();

  // Waits for the first robot state, then loops at task.rl_rate until killed, faulted, or stop() is called.
  // Always finishes by holding the damping command for 0.5 s. Returns the exit code.
  int run();

  // Async-signal-safe stop request (SIGINT/SIGTERM).
  static void request_stop();
  static bool stop_requested();

  bool faulted() const { return faulted_; }
  const std::string& fault_reason() const { return fault_reason_; }
  uint64_t iteration() const { return iteration_; }

  // Damping command: kp = 0, kd = runtime.damping_kd, current position target.
  void send_damping();

 private:
  void fault(const std::string& reason);
  bool wait_for_first_state();

  PolicyController& controller_;
  PolicyContext& context_;
  const InferenceConfig& config_;
  uint64_t iteration_ = 0;
  bool faulted_ = false;
  std::string fault_reason_;
  RobotState state_;
};

}  // namespace holosoma
