#include "holosoma_cpp/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "holosoma_cpp/logging.hpp"
#include "holosoma_cpp/math.hpp"

namespace holosoma {

std::vector<double> motor_to_joint(const RobotConfig& robot, const std::vector<double>& motor_values) {
  std::vector<double> joint(static_cast<size_t>(robot.num_joints));
  for (size_t j = 0; j < joint.size(); ++j) {
    joint[j] = motor_values.at(static_cast<size_t>(robot.joint2motor[j]));
  }
  return joint;
}

std::optional<std::vector<double>> metadata_number_list(const OnnxModel& model, const std::string& key) {
  const auto raw = model.metadata_value(key);
  if (!raw) return std::nullopt;
  try {
    const auto parsed = nlohmann::json::parse(*raw);
    if (parsed.is_null()) return std::nullopt;
    return parsed.get<std::vector<double>>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(model.path() + ": ONNX metadata '" + key + "' is not a JSON number list (" + e.what() +
                             ")");
  }
}

BasePolicy::BasePolicy(InferenceConfig config, PolicyContext& context)
    : config_(std::move(config)),
      context_(context),
      num_dofs_(config_.robot.num_joints),
      default_dof_angles_(config_.robot.default_dof_angles),
      obs_builder_(config_.observation),
      latency_(static_cast<int>(config_.task.rl_rate)) {
  joint_offsets_.assign(static_cast<size_t>(num_dofs_), 0.0);
  if (config_.robot.joint_offsets_deg) {
    for (size_t i = 0; i < joint_offsets_.size(); ++i) {
      joint_offsets_[i] = (*config_.robot.joint_offsets_deg)[i] * math::kPi / 180.0;
    }
  }
  last_policy_action_.assign(static_cast<size_t>(num_dofs_), 0.0);
  scaled_policy_action_.assign(static_cast<size_t>(num_dofs_), 0.0);
  desired_base_height_ = config_.task.desired_base_height;
  base_height_command_ = desired_base_height_;
  const auto n = static_cast<size_t>(num_dofs_);
  command_.q.assign(n, 0.0);
  command_.dq.assign(n, 0.0);
  command_.tau.assign(n, 0.0);
  command_.kp.assign(n, 0.0);
  command_.kd.assign(n, 0.0);
}

BasePolicy::~BasePolicy() = default;

void BasePolicy::initialize() {
  if (!obs_builder_.has_group("actor_obs")) {
    throw std::runtime_error("Observation group 'actor_obs' is not configured for this policy.");
  }
  const std::set<std::string> available = available_terms();
  for (const auto& term : obs_builder_.all_terms()) {
    if (available.count(term) == 0) {
      throw std::runtime_error(kind_ + " cannot produce observation term '" + term + "'");
    }
  }
  for (const auto& path : config_.task.model_path) {
    auto slot = std::make_unique<ModelSlot>();
    slot->path = path;
    slot->model = std::make_unique<OnnxModel>(path);
    setup_policy(*slot);
    resolve_gains(*slot);
    slots_.push_back(std::move(slot));
  }
  activate_policy(0, false);
  init_phase_components();
}

void BasePolicy::setup_policy(ModelSlot& slot) {
  OnnxModel& model = *slot.model;
  const auto& inputs = model.inputs();
  if (inputs.size() != 1 || inputs[0].name != "actor_obs") {
    throw std::runtime_error(slot.path + ": expected a single ONNX input named 'actor_obs'");
  }
  const int obs_dim = obs_builder_.group("actor_obs").dim();
  if (inputs[0].element_count() != obs_dim) {
    throw std::runtime_error(slot.path + ": ONNX input 'actor_obs' has " + std::to_string(inputs[0].element_count()) +
                             " values but the observation config produces " + std::to_string(obs_dim));
  }
  if (model.outputs().empty()) {
    throw std::runtime_error(slot.path + ": ONNX model has no outputs");
  }
  const int64_t action_dim = model.outputs()[0].element_count();
  if (action_dim > num_dofs_) {
    throw std::runtime_error(slot.path + ": action has " + std::to_string(action_dim) + " values for " +
                             std::to_string(num_dofs_) + " joints");
  }
}

void BasePolicy::resolve_gains(ModelSlot& slot) const {
  const RobotConfig& robot = config_.robot;
  // A model trained for another joint order would silently get scrambled observations.
  if (const auto raw = slot.model->metadata_value("dof_names")) {
    std::vector<std::string> names;
    try {
      names = nlohmann::json::parse(*raw).get<std::vector<std::string>>();
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(slot.path + ": ONNX metadata 'dof_names' is malformed (" + e.what() + ")");
    }
    if (names != robot.dof_names) {
      throw std::runtime_error(slot.path + ": ONNX metadata dof_names do not match robot.dof_names");
    }
  }

  std::vector<double> kp, kd;
  if (robot.motor_kp && robot.motor_kd) {
    kp = *robot.motor_kp;
    kd = *robot.motor_kd;
    slot.gain_source = "config (override)";
  } else {
    const auto onnx_kp = metadata_number_list(*slot.model, "kp");
    const auto onnx_kd = metadata_number_list(*slot.model, "kd");
    if (!onnx_kp || !onnx_kd) {
      throw std::runtime_error(
          "No KP/KD values found. Either provide them in robot config or ensure ONNX model has metadata attached "
          "during training: " +
          slot.path);
    }
    kp = *onnx_kp;
    kd = *onnx_kd;
    slot.gain_source = "ONNX metadata";
  }
  if (kp.size() != static_cast<size_t>(robot.num_motors)) {
    throw std::runtime_error("KP array length (" + std::to_string(kp.size()) + ") does not match num_motors (" +
                             std::to_string(robot.num_motors) + ")");
  }
  if (kd.size() != static_cast<size_t>(robot.num_motors)) {
    throw std::runtime_error("KD array length (" + std::to_string(kd.size()) + ") does not match num_motors (" +
                             std::to_string(robot.num_motors) + ")");
  }
  for (size_t i = 0; i < kp.size(); ++i) {
    if (!std::isfinite(kp[i]) || !std::isfinite(kd[i]) || kp[i] < 0.0 || kd[i] < 0.0) {
      throw std::runtime_error(slot.path + ": gains must be finite and non-negative");
    }
  }
  slot.kp = motor_to_joint(robot, kp);
  slot.kd = motor_to_joint(robot, kd);
}

const std::vector<double>& BasePolicy::joint_kp() const { return active_slot().kp; }
const std::vector<double>& BasePolicy::joint_kd() const { return active_slot().kd; }

std::string BasePolicy::active_model_name() const {
  return std::filesystem::path(active_slot().path).filename().string();
}

void BasePolicy::activate_policy(int index, bool announce) {
  if (index < 0 || index >= num_models()) return;
  active_index_ = index;
  std::fill(last_policy_action_.begin(), last_policy_action_.end(), 0.0);
  std::fill(scaled_policy_action_.begin(), scaled_policy_action_.end(), 0.0);
  on_policy_switched();
  if (announce && num_models() > 1) {
    log::info(log::cat("Switched to policy [", index + 1, "]: ", active_model_name()), log::Color::kBlue);
  }
}

void BasePolicy::init_phase_components() {
  use_phase_ = config_.task.use_phase;
  if (use_phase_) {
    phase_[0] = 0.0;
    phase_[1] = math::kPi;
    phase_dt_ = 2 * math::kPi / (config_.task.rl_rate * config_.task.gait_period);
  }
}

std::set<std::string> BasePolicy::available_terms() const {
  return {"base_quat", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"};
}

void BasePolicy::compute_terms(const RobotState& state, TermValues& terms) {
  terms["base_quat"] = state.base_quat_wxyz;
  if (config_.task.debug.force_zero_angular_velocity) {
    terms["base_ang_vel"] = {0.0, 0.0, 0.0};
  } else {
    terms["base_ang_vel"] = state.base_ang_vel;
  }
  auto& dof_pos = terms["dof_pos"];
  dof_pos.resize(static_cast<size_t>(num_dofs_));
  for (size_t i = 0; i < dof_pos.size(); ++i) dof_pos[i] = state.dof_pos[i] - default_dof_angles_[i];
  terms["dof_vel"] = state.dof_vel;
  if (config_.task.debug.force_upright_imu) {
    terms["projected_gravity"] = {0.0, 0.0, -1.0};
  } else {
    const math::Quat q{state.base_quat_wxyz[0], state.base_quat_wxyz[1], state.base_quat_wxyz[2],
                       state.base_quat_wxyz[3]};
    const math::Vec3 g = math::quat_rotate_inverse(q, {0.0, 0.0, -1.0});
    terms["projected_gravity"] = {g[0], g[1], g[2]};
  }
}

const std::vector<float>& BasePolicy::prepare_actor_obs(const RobotState& state) {
  TermValues terms;
  compute_terms(state, terms);
  ObservationGroup& group = obs_builder_.group("actor_obs");
  last_obs_.resize(static_cast<size_t>(group.dim()));
  group.update(terms, last_obs_.data());
  return last_obs_;
}

std::vector<double> BasePolicy::clip_action(const std::vector<float>& raw) {
  std::vector<double> out(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    out[i] = static_cast<double>(std::clamp(raw[i], -100.0f, 100.0f));
  }
  return out;
}

std::vector<double> BasePolicy::rl_inference(const RobotState& state) {
  const std::vector<float>& obs = prepare_actor_obs(state);
  if (config_.task.print_observations) print_observations(obs);

  OnnxModel& model = *active_slot().model;
  std::vector<float>& input = model.input("actor_obs");
  std::copy(obs.begin(), obs.end(), input.begin());
  model.run();
  const std::vector<double> action = clip_action(model.output(model.outputs()[0].name));

  last_policy_action_ = action;
  scaled_policy_action_.assign(action.size(), 0.0);
  const float scale = static_cast<float>(config_.task.policy_action_scale);
  for (size_t i = 0; i < action.size(); ++i) {
    // float32 action * scale stays float32 in numpy.
    scaled_policy_action_[i] = static_cast<double>(static_cast<float>(action[i]) * scale);
  }
  if (config_.task.debug.force_zero_action) {
    std::fill(scaled_policy_action_.begin(), scaled_policy_action_.end(), 0.0);
  }
  return scaled_policy_action_;
}

std::optional<ManualCommand> BasePolicy::manual_command(const RobotState&) { return std::nullopt; }

std::vector<double> BasePolicy::init_target(const RobotState& state) {
  std::vector<double> q(state.dof_pos.begin(), state.dof_pos.begin() + num_dofs_);
  if (get_ready_state_) {
    const double alpha = static_cast<double>(init_count_) / 500.0;
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = state.dof_pos[i] + (default_dof_angles_[i] - state.dof_pos[i]) * alpha;
    }
    ++init_count_;
  }
  return q;
}

void BasePolicy::policy_action(const RobotState& state) {
  const bool use_policy = use_policy_action_;
  const bool get_ready = get_ready_state_;
  inference_ran_ = false;
  if (state.dof_pos.size() != static_cast<size_t>(num_dofs_) ||
      state.dof_vel.size() != static_cast<size_t>(num_dofs_)) {
    throw std::runtime_error("Robot state has the wrong number of joints");
  }

  std::optional<std::vector<double>> kp_override, kd_override;
  std::vector<double> q_target;
  {
    LatencyTracker::Scope scope(latency_, "preprocessing");
    if (get_ready) {
      q_target = init_target(state);
      init_count_ = std::min(init_count_, 500);
      // Stiffer (or softer) gains while ramping to the start pose; the policy keeps its own.
      const double scale = config_.robot.interp_gain_scale;
      if (scale != 1.0) {
        kp_override = active_slot().kp;
        kd_override = active_slot().kd;
        for (auto& v : *kp_override) v *= scale;
        for (auto& v : *kd_override) v *= scale;
      }
    } else if (!use_policy) {
      if (auto manual = manual_command(state)) {
        q_target = std::move(manual->q);
        kp_override = std::move(manual->kp);
        kd_override = std::move(manual->kd);
      } else {
        q_target.assign(state.dof_pos.begin(), state.dof_pos.end());
      }
    }
  }

  if (use_policy && !get_ready) {
    std::vector<double> scaled;
    {
      LatencyTracker::Scope scope(latency_, "inference");
      scaled = rl_inference(state);
    }
    inference_ran_ = true;
    LatencyTracker::Scope scope(latency_, "postprocessing");
    if (scaled.size() != static_cast<size_t>(num_dofs_)) {
      // Policies with fewer outputs drive the trailing joints; the leading ones get zeros.
      std::vector<double> padded(static_cast<size_t>(num_dofs_) - scaled.size(), 0.0);
      padded.insert(padded.end(), scaled.begin(), scaled.end());
      scaled.swap(padded);
    }
    q_target.resize(scaled.size());
    for (size_t i = 0; i < scaled.size(); ++i) q_target[i] = scaled[i] + default_dof_angles_[i];
  }

  const std::vector<double>& kp = kp_override ? *kp_override : active_slot().kp;
  const std::vector<double>& kd = kd_override ? *kd_override : active_slot().kd;
  for (size_t i = 0; i < static_cast<size_t>(num_dofs_); ++i) {
    command_.q[i] = q_target[i] + joint_offsets_[i];
    command_.dq[i] = 0.0;
    command_.tau[i] = 0.0;
    command_.kp[i] = kp[i] * context_.kp_level;
    command_.kd[i] = kd[i] * context_.kd_level;
    if (!std::isfinite(command_.q[i]) || !std::isfinite(command_.kp[i]) || !std::isfinite(command_.kd[i])) {
      throw std::runtime_error("Refusing to send a non-finite command for joint " + config_.robot.dof_names[i]);
    }
  }
  LatencyTracker::Scope scope(latency_, "action_pub");
  context_.robot->send_command(command_);
}

void BasePolicy::apply_velocity(const VelCmd& vc) {
  lin_vel_command_[0] = vc.lin_vel_x;
  lin_vel_command_[1] = vc.lin_vel_y;
  ang_vel_command_ = vc.ang_vel;
}

void BasePolicy::update_phase_time() {
  for (double& p : phase_) {
    p = std::fmod(p + phase_dt_ + math::kPi, 2 * math::kPi) - math::kPi;
  }
}

void BasePolicy::dispatch_command(StateCommand command) {
  switch (command) {
    case StateCommand::START:
      handle_start_policy();
      return;
    case StateCommand::STOP:
      handle_stop_policy();
      return;
    case StateCommand::INIT:
      handle_init_state();
      return;
    case StateCommand::KILL:
      log::info("Killing program via command", log::Color::kRed);
      context_.kill_requested = true;
      return;
    case StateCommand::NEXT_POLICY:
      activate_policy((active_index_ + 1) % num_models(), true);
      return;
    case StateCommand::KP_UP:
      context_.kp_level += 0.1;
      return;
    case StateCommand::KP_DOWN:
      context_.kp_level = std::max(0.0, context_.kp_level - 0.1);
      return;
    case StateCommand::KP_UP_FINE:
      context_.kp_level += 0.01;
      return;
    case StateCommand::KP_DOWN_FINE:
      context_.kp_level = std::max(0.0, context_.kp_level - 0.01);
      return;
    case StateCommand::KP_RESET:
      context_.kp_level = 1.0;
      return;
    default:
      break;
  }
  if (const auto index = switch_policy_index(command)) {
    if (*index != active_index_ && *index < num_models()) activate_policy(*index, true);
  }
}

void BasePolicy::handle_start_policy() {
  use_policy_action_ = true;
  get_ready_state_ = false;
  log::info("Using policy actions", log::Color::kBlue);
  phase_[0] = 0.0;
  phase_[1] = math::kPi;
}

void BasePolicy::handle_stop_policy() {
  use_policy_action_ = false;
  get_ready_state_ = false;
  log::info("Actions set to zero");
}

void BasePolicy::handle_init_state() {
  get_ready_state_ = true;
  init_count_ = 0;
  log::info("Setting to init state");
}

void BasePolicy::print_control_status() const {
  log::info("------------ Control Status ------------");
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f", context_.kp_level);
  log::info(
      log::cat("Active policy [", active_index_ + 1, "/", num_models(), "]: ", active_model_name(), " Kp level ", buf));
}

void BasePolicy::print_observations(const std::vector<float>& obs) const {
  const ObservationGroup& group = obs_builder_.group("actor_obs");
  std::string out = "\r\n========== Observation Vector ==========\r\nactor_obs:\r\n";
  size_t offset = 0;
  char buf[64];
  for (const auto& term : group.sorted_terms()) {
    const size_t dim = static_cast<size_t>(group.term_dim(term) * group.history_length());
    std::snprintf(buf, sizeof(buf), "  %-20s (dim=%2d, hist=%d): [", term.c_str(), group.term_dim(term),
                  group.history_length());
    out += buf;
    for (size_t i = 0; i < dim; ++i) {
      std::snprintf(buf, sizeof(buf), "%s%.3f", i ? " " : "", static_cast<double>(obs[offset + i]));
      out += buf;
    }
    out += "]\r\n";
    offset += dim;
  }
  out += "========================================\r\n";
  std::fputs(out.c_str(), stdout);
  std::fflush(stdout);
}

}  // namespace holosoma
