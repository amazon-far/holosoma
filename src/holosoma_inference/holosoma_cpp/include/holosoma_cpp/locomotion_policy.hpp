// Velocity-tracking locomotion (holosoma_inference/policies/locomotion.py).
#pragma once

#include "holosoma_cpp/policy.hpp"

namespace holosoma {

class LocomotionPolicy : public BasePolicy {
 public:
  LocomotionPolicy(InferenceConfig config, PolicyContext& context);

  void apply_velocity(const VelCmd& vc) override;
  void dispatch_command(StateCommand command) override;
  void update_phase_time() override;
  void handle_start_policy() override;
  void print_control_status() const override;

  bool walking() const { return stand_command_ == 1.0; }

 protected:
  std::set<std::string> available_terms() const override;
  void compute_terms(const RobotState& state, TermValues& terms) override;

 private:
  void maybe_switch_to_walk_mode(const VelCmd& vc);
  void handle_stand_command();
  void handle_zero_velocity();

  // Re-arms the gait restart when the policy (re)starts, so a command already held at start still gets the
  // lead-foot choice. Only with restart_lead_foot set, so the default keeps the free-running (0, pi) start.
  void arm_restart();

  bool is_standing_ = false;
};

}  // namespace holosoma
