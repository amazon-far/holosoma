#include "holosoma_cpp/contextual_policy.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace holosoma {
ContextualPolicy::ContextualPolicy(InferenceConfig config, PolicyContext& context)
    : BasePolicy(std::move(config), context) {
  kind_ = "ContextualPolicy";
}

void ContextualPolicy::setup_policy(ModelSlot& slot) {
  const auto& options = config_.contextual.value();
  contextual::Spec spec;
  spec.joints = config_.robot.dof_names;
  spec.defaults = config_.robot.default_dof_angles;
  if (config_.robot.motor_kp) spec.kp = motor_to_joint(config_.robot, *config_.robot.motor_kp);
  if (config_.robot.motor_kd) spec.kd = motor_to_joint(config_.robot, *config_.robot.motor_kd);
  spec.action_scale = config_.task.policy_action_scale;
  spec.command_dim = options.command_dim;
  spec.height = options.depth_height;
  spec.width = options.depth_width;
  spec.upright = config_.task.debug.force_upright_imu;
  spec.zero_omega = config_.task.debug.force_zero_angular_velocity;
  cores_.push_back(std::make_unique<contextual::Policy>(slot.model, std::move(spec), options));
}

void ContextualPolicy::resolve_gains(ModelSlot& slot) const {
  // Contextual metadata is in model joint order; the core resolves that order
  // and keeps explicit runtime gains authoritative.
  slot.kp = cores_.back()->kp();
  slot.kd = cores_.back()->kd();
  slot.gain_source = config_.robot.motor_kp ? "config (override)" : "ONNX metadata";
}

contextual::Policy& ContextualPolicy::core() { return *cores_.at(static_cast<size_t>(active_index_)); }
const contextual::Policy& ContextualPolicy::core() const { return *cores_.at(static_cast<size_t>(active_index_)); }

void ContextualPolicy::on_policy_switched() {
  core().Reset();
  input_.reset();
  allow_climb_at_stand_ = allow_climb_continuation_ = false;
}

void ContextualPolicy::handle_start_policy() {
  // Embedding callers own startup notices. Keep numerical replay stdout clean.
  use_policy_action_ = true;
  get_ready_state_ = false;
}

void ContextualPolicy::apply_velocity(const VelCmd& velocity) {
  core().SetVelocity(velocity.lin_vel_x, velocity.lin_vel_y, velocity.ang_vel);
  BasePolicy::apply_velocity(velocity);
}

void ContextualPolicy::set_input(ContextualInput input) {
  if (!std::isfinite(input.monotonic_time) || input.monotonic_time < 0)
    throw std::invalid_argument("Contextual transition time must be finite and nonnegative");
  if (input.allow_climb_at_stand) allow_climb_at_stand_ = *input.allow_climb_at_stand;
  if (input.allow_climb_continuation) allow_climb_continuation_ = *input.allow_climb_continuation;
  if (input.allow_climb_at_stand || input.allow_climb_continuation)
    core().SetClimbAtStandAllowed(allow_climb_at_stand_, allow_climb_continuation_);
  if (input.speed) core().SetSpeed(*input.speed);
  if (input.fixed_arm) core().SetFixedArm(*input.fixed_arm);
  if (input.dz) core().SetDz(*input.dz);
  if (input.command_index) core().SetCommand(*input.command_index);
  if (input.velocity) core().SetVelocity((*input.velocity)[0], (*input.velocity)[1], (*input.velocity)[2]);
  if (input.key) core().RequestKey(*input.key, input.monotonic_time);
  input_ = std::move(input);
}

std::vector<double> ContextualPolicy::infer_observation(const std::vector<float>&) {
  throw std::invalid_argument("Contextual policy requires RobotState and ContextualInput, not packed observations");
}

std::vector<double> ContextualPolicy::rl_inference(const RobotState& state) {
  auto input = std::exchange(input_, std::nullopt);
  if (!input) throw std::invalid_argument("Supply ContextualInput for each contextual inference tick");
  contextual::State observation;
  observation.q = state.dof_pos;
  observation.dq = state.dof_vel;
  std::copy_n(state.base_quat_wxyz.begin(), 4, observation.quat.begin());
  std::copy_n(state.base_ang_vel.begin(), 3, observation.omega.begin());
  if (state.projected_gravity) {
    observation.gravity.emplace();
    std::copy_n(state.projected_gravity->begin(), 3, observation.gravity->begin());
  }
  auto targets = core().Target(observation, input->depth, input->monotonic_time);
  last_obs_ = core().observation();
  last_policy_action_.assign(core().action().begin(), core().action().end());
  scaled_policy_action_.assign(core().scaled().begin(), core().scaled().end());
  if (config_.task.debug.force_zero_action) {
    std::fill(scaled_policy_action_.begin(), scaled_policy_action_.end(), 0.0);
    return std::vector<double>(targets.size(), 0.0);
  }
  for (size_t i = 0; i < targets.size(); ++i) targets[i] -= default_dof_angles_[i];
  return targets;
}
}  // namespace holosoma
