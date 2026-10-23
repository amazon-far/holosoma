#include "holosoma_cpp/api.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "holosoma_cpp/controller.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace holosoma {
namespace {
class MemoryVelocity final : public VelocityProvider {
 public:
  std::optional<VelCmd> poll_velocity() override { return std::nullopt; }
  // The host owns the real operator input, so count requests for it to mirror.
  void zero() override { ++zero_requests; }
  std::uint64_t zero_requests = 0;
};
class NoCommands final : public CommandProvider {
 public:
  std::vector<StateCommand> poll_commands() override { return {}; }
  void bind(const std::string&, StateCommand) override {}
};

InferenceConfig load_preset(const std::string& preset, const std::string& checkpoint) {
  std::ifstream file(preset);
  if (!file) throw ConfigError("Cannot open preset: " + preset);
  std::ostringstream text;
  text << file.rdbuf();
  auto config = config_from_yaml(text.str(), std::filesystem::absolute(preset).parent_path().string());
  if (!checkpoint.empty()) config.task.model_path = {checkpoint};
  return config;
}
}  // namespace

struct Policy::Impl {
  OfflineInterface robot;
  MemoryVelocity velocity;
  NoCommands commands;
  PolicyContext context;
  std::unique_ptr<BasePolicy> policy;
  bool autostart = false;
  bool pending_engagement = false;
};

Policy::Policy(const std::string& preset, const std::string& checkpoint) : Policy(load_preset(preset, checkpoint)) {}

Policy::Policy(InferenceConfig config, bool autostart) : impl_(std::make_unique<Impl>()) {
  validate_inference_config(config);
  impl_->context.robot = &impl_->robot;
  impl_->context.velocity_input = &impl_->velocity;
  impl_->context.command_provider = &impl_->commands;
  impl_->policy = make_policy(config, impl_->context);
  impl_->autostart = autostart;
  impl_->pending_engagement = autostart;
  if (autostart) {
    impl_->policy->force_policy_active();
    if (config.is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
  }
}
Policy::~Policy() = default;

void Policy::set_reference(const ReferenceWindow* window) {
  auto* tracker = dynamic_cast<GeneralWBTPolicy*>(impl_->policy.get());
  if (!tracker) throw std::runtime_error("this preset's policy carries its own reference");
  tracker->set_reference(window);
}
int Policy::num_dofs() const { return config().robot.num_joints; }
const std::vector<std::string>& Policy::dof_names() const { return config().robot.dof_names; }
const InferenceConfig& Policy::config() const { return impl_->policy->config(); }

std::vector<double> Policy::step(const std::vector<float>& observation) {
  auto offsets = impl_->policy->infer_observation(observation);
  const auto& cfg = impl_->policy->config().robot;
  if (offsets.size() > cfg.default_dof_angles.size()) throw std::runtime_error("Action exceeds robot dimension");
  offsets.insert(offsets.begin(), cfg.default_dof_angles.size() - offsets.size(), 0.0);
  for (size_t i = 0; i < offsets.size(); ++i) {
    offsets[i] += cfg.default_dof_angles[i];
    if (cfg.joint_offsets_deg) offsets[i] += (*cfg.joint_offsets_deg)[i] * 3.14159265358979323846 / 180.0;
    if (!std::isfinite(offsets[i])) throw std::runtime_error("Non-finite joint target");
  }
  return offsets;
}

void Policy::set_state(const RobotState& state) {
  const auto n = static_cast<size_t>(impl_->policy->config().robot.num_joints);
  if (state.base_pos.size() != 3 || state.base_quat_wxyz.size() != 4 || state.base_lin_vel.size() != 3 ||
      state.base_ang_vel.size() != 3 || state.dof_pos.size() != n || state.dof_vel.size() != n)
    throw std::invalid_argument("Robot state dimensions do not match the configuration");
  for (const auto* values : {&state.base_pos, &state.base_quat_wxyz, &state.base_lin_vel, &state.base_ang_vel,
                             &state.dof_pos, &state.dof_vel})
    for (double v : *values)
      if (!std::isfinite(v)) throw std::invalid_argument("Robot state must be finite");
  double norm = 0;
  for (double v : state.base_quat_wxyz) norm += v * v;
  if (norm < 1e-16) throw std::invalid_argument("Robot orientation must be nonzero");
  if (state.projected_gravity) {
    if (state.projected_gravity->size() != 3)
      throw std::invalid_argument("Corrected gravity must have three components");
    for (double v : *state.projected_gravity)
      if (!std::isfinite(v)) throw std::invalid_argument("Corrected gravity must be finite");
  }
  impl_->robot.set_state(state);
}

std::vector<double> Policy::infer_offsets(const RobotState& state) {
  set_state(state);
  return impl_->policy->infer_offsets(state);
}
JointCommand Policy::step_state(const RobotState& state, bool advance_phase) {
  set_state(state);
  if (impl_->pending_engagement) {
    impl_->policy->handle_start_policy();
    if (impl_->policy->config().is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
    impl_->pending_engagement = false;
  }
  if (advance_phase) update_phase();
  impl_->policy->policy_action(state);
  return impl_->robot.command();
}
const JointCommand& Policy::step(const RobotState& state) {
  step_state(state);
  return impl_->robot.command();
}
void Policy::command(StateCommand cmd) {
  if (cmd == StateCommand::START || cmd == StateCommand::STOP || cmd == StateCommand::INIT)
    impl_->pending_engagement = false;
  impl_->policy->dispatch_command(cmd);
}
void Policy::velocity(const VelCmd& cmd) { impl_->policy->apply_velocity(cmd); }
std::uint64_t Policy::velocity_zero_requests() const { return impl_->velocity.zero_requests; }
void Policy::update_phase() {
  if (impl_->policy->use_phase()) impl_->policy->update_phase_time();
}
void Policy::reset() {
  impl_->policy->reset();
  impl_->pending_engagement = impl_->autostart;
  if (impl_->autostart) {
    impl_->policy->force_policy_active();
    if (impl_->policy->config().is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
  }
}
BasePolicy& Policy::policy() { return *impl_->policy; }
const BasePolicy& Policy::policy() const { return *impl_->policy; }
}  // namespace holosoma
