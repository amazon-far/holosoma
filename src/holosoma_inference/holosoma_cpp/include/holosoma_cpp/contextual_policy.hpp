#pragma once

#include "holosoma_cpp/contextual_kernel.hpp"
#include "holosoma_cpp/policy.hpp"

namespace holosoma {
// Depth-conditioned policy using BasePolicy's shared model slots. Transports
// supply one ContextualInput per inference tick and own input freshness.
class ContextualPolicy final : public BasePolicy {
 public:
  ContextualPolicy(InferenceConfig config, PolicyContext& context);
  void set_input(ContextualInput input);
  void clear_input() { input_.reset(); }
  void handle_start_policy() override;
  void apply_velocity(const VelCmd& velocity) override;
  contextual::Policy& core();
  const contextual::Policy& core() const;
  std::vector<double> infer_observation(const std::vector<float>&) override;

 protected:
  void setup_policy(ModelSlot& slot) override;
  void resolve_gains(ModelSlot& slot) const override;
  void on_policy_switched() override;
  std::vector<double> rl_inference(const RobotState& state) override;

 private:
  std::vector<std::unique_ptr<contextual::Policy>> cores_;
  std::optional<ContextualInput> input_;
  bool allow_climb_at_stand_ = false;
  bool allow_climb_continuation_ = false;
};
}  // namespace holosoma
