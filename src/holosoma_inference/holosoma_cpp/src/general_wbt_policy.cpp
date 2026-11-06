#include "holosoma_cpp/general_wbt_policy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace holosoma {
namespace {
template <size_t N, class Values>
void copy_finite(const Values& values, std::array<float, N>& out) {
  if (values.size() != N) throw std::invalid_argument("Reference/state dimension mismatch");
  for (size_t i = 0; i < N; ++i) {
    out[i] = static_cast<float>(values[i]);
    if (!std::isfinite(out[i])) throw std::invalid_argument("Reference/state must be finite");
  }
}

void copy_quat(const std::array<double, 4>& values, std::array<float, 4>& out) {
  copy_finite(values, out);
  double norm = 0;
  for (double value : values) norm += value * value;
  if (norm < 1e-12) throw std::invalid_argument("Reference orientation must be nonzero");
}
}  // namespace

InferenceConfig live_policy_config(const std::string& checkpoint, Backend backend, int intra_op_threads) {
  InferenceConfig config;
  config.robot.dof_names.assign(live::kG1BodyJointNames.begin(), live::kG1BodyJointNames.end());
  config.robot.default_dof_angles.assign(live::kActionDim, 0.0);
  config.robot.default_motor_angles = config.robot.default_dof_angles;
  config.robot.joint2motor.resize(live::kActionDim);
  std::iota(config.robot.joint2motor.begin(), config.robot.joint2motor.end(), 0);
  config.robot.motor2joint = config.robot.joint2motor;
  config.observation.obs_dict = {{"actor_obs", {}}, {"actor_targets", {}}};
  config.task.model_path = {checkpoint};
  config.task.use_phase = false;
  config.runtime.backend = backend;
  config.runtime.intra_op_threads = intra_op_threads;
  return config;
}

GeneralWBTPolicy::GeneralWBTPolicy(InferenceConfig config, PolicyContext& context)
    : BasePolicy(std::move(config), context) {
  kind_ = "GeneralWBTPolicy";
}

void GeneralWBTPolicy::setup_policy(ModelSlot& slot) {
  auto core = std::make_unique<live::WbtPolicy>();
  live::WbtHistoryInitialization history;
  if (!live::ParseWbtHistoryInitialization(config_.task.wbt_history_initialization, history))
    throw ConfigError("task.wbt_history_initialization: '" + config_.task.wbt_history_initialization +
                      "' is not repeat_first_valid_obs or zero_pad");
  core->SetHistoryInitialization(history);
  if (!core->Initialize(slot.model)) throw ConfigError(slot.path + ": invalid live WBT checkpoint");
  if (config_.robot.dof_names != core->JointNames())
    throw ConfigError("Live WBT robot.dof_names must match checkpoint metadata order");
  core->SetControlInterval(1.0 / config_.task.rl_rate);
  cores_.push_back(std::move(core));
}

std::set<std::string> GeneralWBTPolicy::available_terms() const {
  // Metadata validation in the core owns the actual layout. Preset terms describe
  // the default export but do not override a different checkpoint's contract.
  const auto terms = obs_builder_.all_terms();
  return {terms.begin(), terms.end()};
}

live::WbtPolicy& GeneralWBTPolicy::core() { return *cores_.at(static_cast<size_t>(active_index_)); }
const live::WbtPolicy& GeneralWBTPolicy::core() const { return *cores_.at(static_cast<size_t>(active_index_)); }
void GeneralWBTPolicy::on_policy_switched() { core().Reset(); }
void GeneralWBTPolicy::handle_start_policy() {
  BasePolicy::handle_start_policy();
  core().ResetHeading();
}
std::vector<double> GeneralWBTPolicy::infer_observation(const std::vector<float>&) {
  throw std::invalid_argument("Live WBT requires step(RobotState) with set_reference(), not packed observations");
}

std::vector<double> GeneralWBTPolicy::rl_inference(const RobotState& state) {
  if (!reference_) throw std::runtime_error("set_reference() before stepping");
  const size_t slots = static_cast<size_t>(std::max(1, core().LookaheadSlots()));
  if (reference_->q.size() != slots * live::kActionDim || reference_->dq.size() != reference_->q.size() ||
      reference_->quat.size() != slots)
    throw std::invalid_argument("Reference window must match the checkpoint's lookahead slots and joint count");
  live::RobotObservation observation;
  copy_finite(state.dof_pos, observation.dof_pos_mj);
  copy_finite(state.dof_vel, observation.dof_vel_mj);
  copy_finite(state.base_quat_wxyz, observation.root_quat_wxyz);
  copy_finite(state.base_ang_vel, observation.base_ang_vel);
  // Same debug overrides and corrected gravity as BasePolicy::compute_terms.
  const DebugConfig& debug = config_.task.debug;
  if (debug.force_zero_angular_velocity) observation.base_ang_vel.fill(0.0f);
  if (debug.force_upright_imu) {
    observation.projected_gravity = std::array<float, 3>{0.0f, 0.0f, -1.0f};
  } else if (state.projected_gravity) {
    std::array<float, 3> gravity{};
    copy_finite(*state.projected_gravity, gravity);
    observation.projected_gravity = gravity;
  }
  live::WbtReferenceSequence reference;
  for (size_t slot = 0; slot < slots; ++slot) {
    live::WbtReferenceFrame frame;
    const size_t start = slot * live::kActionDim;
    copy_finite(std::vector<double>(reference_->q.begin() + start, reference_->q.begin() + start + live::kActionDim),
                frame.dof_pos);
    copy_finite(std::vector<double>(reference_->dq.begin() + start, reference_->dq.begin() + start + live::kActionDim),
                frame.dof_vel);
    copy_quat(reference_->quat[slot], frame.root_quat_wxyz);
    reference.lookahead.push_back(frame);
  }
  reference.current = reference.lookahead.front();
  if (reference_->current) {
    copy_finite(reference_->current->q, reference.current.dof_pos);
    copy_finite(reference_->current->dq, reference.current.dof_vel);
    copy_quat(reference_->current->quat, reference.current.root_quat_wxyz);
  }
  std::array<float, live::kActionDim> offsets{};
  if (!core().ComputeAction(observation, reference, offsets.data()))
    throw std::runtime_error("Live WBT inference failed");
  if (debug.force_zero_action) offsets.fill(0.0f);
  last_obs_ = core().LastObservation();
  if (config_.task.print_observations) print_observations(last_obs_);
  std::vector<double> result(live::kActionDim);
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<double>(offsets[i]) + core().DefaultDofPos()[i] - default_dof_angles_[i];
  return result;
}
}  // namespace holosoma
