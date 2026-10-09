#include "holosoma_cpp/wbt_policy.hpp"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

#include "holosoma_cpp/logging.hpp"

namespace holosoma {
namespace {

constexpr const char* kObsInput = "obs";
constexpr const char* kTimeStepInput = "time_step";
constexpr const char* kActionsOutput = "actions";
constexpr const char* kJointPosOutput = "joint_pos";
constexpr const char* kJointVelOutput = "joint_vel";
constexpr const char* kRefQuatOutput = "ref_quat_xyzw";

std::vector<double> as_float32_values(const std::vector<double>& values) {
  std::vector<double> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = static_cast<double>(static_cast<float>(values[i]));
  return out;
}

// _parse_action_scale_metadata: JSON scalar, JSON list, or a comma-separated string.
std::vector<double> parse_action_scale(const std::string& raw, const std::string& where) {
  std::vector<double> values;
  nlohmann::json parsed;
  bool is_json = true;
  try {
    parsed = nlohmann::json::parse(raw);
  } catch (const nlohmann::json::exception&) {
    is_json = false;
  }
  std::string csv;
  if (is_json && parsed.is_number()) {
    values.push_back(parsed.get<double>());
  } else if (is_json && parsed.is_array()) {
    for (const auto& v : parsed) {
      if (!v.is_number()) throw std::runtime_error(where + ": action_scale list must contain numbers");
      values.push_back(v.get<double>());
    }
    if (values.empty()) throw std::runtime_error(where + ": ONNX metadata action_scale is empty.");
  } else {
    csv = is_json && parsed.is_string() ? parsed.get<std::string>() : raw;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
      const auto first = token.find_first_not_of(" \t");
      if (first == std::string::npos) continue;
      try {
        values.push_back(std::stod(token.substr(first)));
      } catch (const std::exception&) {
        throw std::runtime_error(where + ": cannot parse action_scale value '" + token + "'");
      }
    }
    if (values.empty()) throw std::runtime_error(where + ": ONNX metadata action_scale is an empty string.");
  }
  return values;
}

}  // namespace

WholeBodyTrackingPolicy::WholeBodyTrackingPolicy(InferenceConfig config, PolicyContext& context)
    : BasePolicy(std::move(config), context) {
  kind_ = "WholeBodyTrackingPolicy";
  curr_motion_timestep_ = config_.task.motion_start_timestep;
  use_sim_time_ = config_.task.use_sim_time;

  // Motion clock: the simulator bridge writes sim time (ms) into LowState.tick.
  RobotInterface* robot = context_.robot;
  clock_source_ = std::make_unique<ClockSource>([robot]() -> int64_t {
    RobotState state;
    return robot != nullptr && robot->read_state(state) ? static_cast<int64_t>(state.tick_ms) : 0;
  });
  motion_clock_ = std::make_unique<MotionClock>(*clock_source_);
  timestep_util_ =
      std::make_unique<TimestepUtil>(*motion_clock_, 1000.0 / config_.task.rl_rate, config_.task.motion_start_timestep);

  const RobotConfig& robot_cfg = config_.robot;
  stiff_hold_q_ =
      as_float32_values(robot_cfg.stiff_startup_pos ? *robot_cfg.stiff_startup_pos : robot_cfg.default_dof_angles);
  if (!robot_cfg.stiff_startup_kp) {
    throw std::runtime_error("Robot config must specify stiff_startup_kp for WBT policy");
  }
  if (!robot_cfg.stiff_startup_kd) {
    throw std::runtime_error("Robot config must specify stiff_startup_kd for WBT policy");
  }
  stiff_hold_kp_ = as_float32_values(*robot_cfg.stiff_startup_kp);
  stiff_hold_kd_ = as_float32_values(*robot_cfg.stiff_startup_kd);
  if (stiff_hold_q_.size() != static_cast<size_t>(num_dofs_)) {
    throw std::runtime_error("Stiff startup pose dimension mismatch with robot DOFs");
  }
  if (robot_cfg.motion_body_name_ref.empty()) {
    throw std::runtime_error("Robot config must specify motion.body_name_ref for WBT policy");
  }
}

std::set<std::string> WholeBodyTrackingPolicy::available_terms() const {
  return {"motion_command", "motion_ref_ori_b", "base_ang_vel", "dof_pos", "dof_vel", "actions"};
}

void WholeBodyTrackingPolicy::setup_policy(ModelSlot& slot) {
  OnnxModel& model = *slot.model;
  const int obs_dim = obs_builder_.group("actor_obs").dim();
  const TensorInfo* obs = model.find_input(kObsInput);
  const TensorInfo* time_step = model.find_input(kTimeStepInput);
  if (model.inputs().size() != 2 || obs == nullptr || time_step == nullptr) {
    throw std::runtime_error(slot.path + ": WBT models need exactly the inputs 'obs' and 'time_step'");
  }
  if (obs->element_count() != obs_dim) {
    throw std::runtime_error(slot.path + ": ONNX input 'obs' has " + std::to_string(obs->element_count()) +
                             " values but the observation config produces " + std::to_string(obs_dim));
  }
  if (time_step->element_count() != 1) {
    throw std::runtime_error(slot.path + ": ONNX input 'time_step' must hold one value");
  }
  for (const char* name : {kActionsOutput, kJointPosOutput, kJointVelOutput, kRefQuatOutput}) {
    if (model.find_output(name) == nullptr) {
      throw std::runtime_error(slot.path + ": WBT model is missing the output '" + std::string(name) + "'");
    }
  }
  const auto n = static_cast<int64_t>(num_dofs_);
  if (model.find_output(kActionsOutput)->element_count() != n ||
      model.find_output(kJointPosOutput)->element_count() != n ||
      model.find_output(kJointVelOutput)->element_count() != n ||
      model.find_output(kRefQuatOutput)->element_count() != 4) {
    throw std::runtime_error(slot.path + ": WBT outputs must be actions/joint_pos/joint_vel [" + std::to_string(n) +
                             "] and ref_quat_xyzw [4]");
  }
  if (obs_builder_.group("actor_obs").term_dim("motion_command") != 2 * num_dofs_) {
    throw std::runtime_error("observation.obs_dims.motion_command must be 2 * num_joints");
  }

  MotionSlot motion;
  const auto urdf_raw = model.metadata_value("robot_urdf");
  if (!urdf_raw) {
    throw std::runtime_error(slot.path + ": Robot urdf text not found in ONNX metadata");
  }
  std::string urdf;
  try {
    urdf = nlohmann::json::parse(*urdf_raw).get<std::string>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(slot.path + ": ONNX metadata robot_urdf is not a JSON string (" + e.what() + ")");
  }
  motion.kinematics =
      std::make_unique<UrdfChain>(urdf, config_.robot.motion_body_name_ref.front(), config_.robot.dof_names);

  // Reference at the configured start timestep, from a zero observation.
  std::vector<float>& obs_buffer = model.input(kObsInput);
  std::fill(obs_buffer.begin(), obs_buffer.end(), 0.0f);
  model.input(kTimeStepInput)[0] = static_cast<float>(config_.task.motion_start_timestep);
  model.run();
  read_reference_outputs(model, motion.motion_command_0, motion.ref_quat_xyzw_0);
  motion.per_joint_action_scale = configure_action_scale(model, motion.action_scale_source);
  motion_slots_.push_back(std::move(motion));
}

std::optional<std::vector<double>> WholeBodyTrackingPolicy::configure_action_scale(const OnnxModel& model,
                                                                                   std::string& source) const {
  std::vector<double> scales;
  if (const auto raw = model.metadata_value("action_scale")) {
    scales = parse_action_scale(*raw, model.path());
    source = "ONNX metadata";
  } else if (config_.task.action_scales_by_effort_limit_over_p_gain) {
    if (!config_.robot.default_per_joint_action_scale) {
      throw std::runtime_error(
          "task.action_scales_by_effort_limit_over_p_gain=True requires ONNX metadata key 'action_scale' (scalar or "
          "per-joint list) or robot.default_per_joint_action_scale.");
    }
    scales = *config_.robot.default_per_joint_action_scale;
    source = "robot.default_per_joint_action_scale";
    log::warning("ONNX metadata 'action_scale' missing; using robot.default_per_joint_action_scale.");
  } else {
    source = "task.policy_action_scale";
    return std::nullopt;
  }
  if (scales.size() == 1) {
    scales.assign(static_cast<size_t>(num_dofs_), scales.front());
  } else if (scales.size() != static_cast<size_t>(num_dofs_)) {
    throw std::runtime_error("Action scale must contain 1 or " + std::to_string(num_dofs_) + " values, got " +
                             std::to_string(scales.size()) + ".");
  }
  return as_float32_values(scales);
}

void WholeBodyTrackingPolicy::read_reference_outputs(const OnnxModel& model, std::vector<double>& motion_command,
                                                     math::Quat& ref_quat_xyzw) const {
  const auto& joint_pos = model.output(kJointPosOutput);
  const auto& joint_vel = model.output(kJointVelOutput);
  const auto& ref_quat = model.output(kRefQuatOutput);
  motion_command.resize(joint_pos.size() + joint_vel.size());
  std::copy(joint_pos.begin(), joint_pos.end(), motion_command.begin());
  std::copy(joint_vel.begin(), joint_vel.end(), motion_command.begin() + static_cast<long>(joint_pos.size()));
  ref_quat_xyzw = {ref_quat[0], ref_quat[1], ref_quat[2], ref_quat[3]};
}

void WholeBodyTrackingPolicy::on_policy_switched() {
  const MotionSlot& motion = motion_slot();
  motion_command_t_ = motion.motion_command_0;
  ref_quat_xyzw_t_ = motion.ref_quat_xyzw_0;
  motion_clip_progressing_ = false;
  timestep_util_->reset(0);
  curr_motion_timestep_ = timestep_util_->timestep();
  stiff_hold_active_ = true;
  robot_yaw_offset_ = 0.0;
  motion_yaw_offset_ = 0.0;
}

math::Quat WholeBodyTrackingPolicy::ref_body_orientation(const RobotState& state) const {
  const math::Quat base{state.base_quat_wxyz[0], state.base_quat_wxyz[1], state.base_quat_wxyz[2],
                        state.base_quat_wxyz[3]};
  return motion_slot().kinematics->orientation(base, state.dof_pos);
}

std::vector<double> WholeBodyTrackingPolicy::init_target(const RobotState& state) {
  std::vector<double> q(state.dof_pos.begin(), state.dof_pos.end());
  if (get_ready_state_) {
    // Ramp from the current pose to the first frame of the clip.
    const std::vector<double>& target = motion_slot().motion_command_0;
    const double alpha = static_cast<double>(init_count_) / 500.0;
    for (size_t i = 0; i < q.size(); ++i) q[i] = state.dof_pos[i] + (target[i] - state.dof_pos[i]) * alpha;
    ++init_count_;
  }
  return q;
}

void WholeBodyTrackingPolicy::compute_terms(const RobotState& state, TermValues& terms) {
  terms["motion_command"] = motion_command_t_;

  const math::Quat motion_ref = math::remove_yaw_offset(math::xyzw_to_wxyz(ref_quat_xyzw_t_), motion_yaw_offset_);
  const math::Quat robot_ref = math::remove_yaw_offset(ref_body_orientation(state), robot_yaw_offset_);
  const math::Mat3 rel = math::matrix_from_quat(math::subtract_frame_transforms(robot_ref, motion_ref));
  // First two columns of the relative rotation, row-major.
  terms["motion_ref_ori_b"] = {rel[0], rel[1], rel[3], rel[4], rel[6], rel[7]};

  terms["base_ang_vel"] = state.base_ang_vel;
  auto& dof_pos = terms["dof_pos"];
  dof_pos.resize(static_cast<size_t>(num_dofs_));
  for (size_t i = 0; i < dof_pos.size(); ++i) dof_pos[i] = state.dof_pos[i] - default_dof_angles_[i];
  terms["dof_vel"] = state.dof_vel;
  terms["actions"] = last_policy_action_;
}

std::vector<double> WholeBodyTrackingPolicy::rl_inference(const RobotState& state) {
  if (!motion_clip_progressing_) {
    // Keep the motion index pinned at the configured start while waiting for the clip trigger.
    timestep_util_->reset(config_.task.motion_start_timestep);
    curr_motion_timestep_ = timestep_util_->timestep();
  }
  const std::vector<float>& obs = prepare_actor_obs(state);
  if (config_.task.print_observations) print_observations(obs);

  OnnxModel& model = *active_slot().model;
  std::vector<float>& obs_input = model.input(kObsInput);
  std::copy(obs.begin(), obs.end(), obs_input.begin());
  model.input(kTimeStepInput)[0] = static_cast<float>(curr_motion_timestep_);
  model.run();
  const std::vector<double> action = clip_action(model.output(kActionsOutput));
  read_reference_outputs(model, motion_command_t_, ref_quat_xyzw_t_);

  last_policy_action_ = action;
  scaled_policy_action_.assign(action.size(), 0.0);
  const auto& per_joint = motion_slot().per_joint_action_scale;
  const float scalar = static_cast<float>(config_.task.policy_action_scale);
  for (size_t i = 0; i < action.size(); ++i) {
    const float scale = per_joint ? static_cast<float>((*per_joint)[i]) : scalar;
    scaled_policy_action_[i] = static_cast<double>(static_cast<float>(action[i]) * scale);
  }
  set_motion_timestep();
  return scaled_policy_action_;
}

void WholeBodyTrackingPolicy::set_motion_timestep() {
  if (!motion_clip_progressing_) return;
  const int prev = curr_motion_timestep_;
  if (use_sim_time_) {
    bool jumped = false;
    curr_motion_timestep_ = timestep_util_->get_timestep(&jumped);
    if (jumped) log::warning("Motion clock jumped; re-anchoring.");
  } else {
    curr_motion_timestep_ += 1;
  }
  if (curr_motion_timestep_ != prev) {
    log::debug(log::cat("Motion timestep: ", prev, " -> ", curr_motion_timestep_));
  }
  const auto& end = config_.task.motion_end_timestep;
  if (end && *end != 0 && curr_motion_timestep_ >= *end) {
    log::info(log::cat("Reached end timestep ", *end, ", stopping motion clip"), log::Color::kYellow);
    motion_clip_progressing_ = false;
    curr_motion_timestep_ = *end;
  }
}

std::optional<ManualCommand> WholeBodyTrackingPolicy::manual_command(const RobotState&) {
  if (!stiff_hold_active_) return std::nullopt;
  return ManualCommand{stiff_hold_q_, stiff_hold_kp_, stiff_hold_kd_};
}

void WholeBodyTrackingPolicy::handle_start_policy() {
  BasePolicy::handle_start_policy();
  stiff_hold_active_ = false;
  capture_robot_yaw_offset();
  motion_yaw_offset_ = math::quat_yaw(math::xyzw_to_wxyz(motion_slot().ref_quat_xyzw_0));
  log::info(log::cat("Motion yaw offset captured at ", motion_yaw_offset_ * 180.0 / math::kPi, " deg"),
            log::Color::kBlue);
}

void WholeBodyTrackingPolicy::capture_robot_yaw_offset() {
  RobotState state;
  if (!context_.robot->read_state(state)) {
    robot_yaw_offset_ = 0.0;
    log::warning("Unable to capture robot yaw offset - missing robot state.");
    return;
  }
  robot_yaw_offset_ = math::quat_yaw(ref_body_orientation(state));
  log::info(log::cat("Robot yaw offset captured at ", robot_yaw_offset_ * 180.0 / math::kPi, " deg"),
            log::Color::kBlue);
}

void WholeBodyTrackingPolicy::handle_stop_policy() {
  use_policy_action_ = false;
  get_ready_state_ = false;
  stiff_hold_active_ = true;
  log::info("Actions set to stiff startup command");
  motion_clip_progressing_ = false;
  timestep_util_->reset(0);
  curr_motion_timestep_ = timestep_util_->timestep();
  ref_quat_xyzw_t_ = motion_slot().ref_quat_xyzw_0;
  motion_command_t_ = motion_slot().motion_command_0;
  robot_yaw_offset_ = 0.0;
  motion_yaw_offset_ = 0.0;
}

void WholeBodyTrackingPolicy::handle_start_motion_clip() {
  timestep_util_->reset(config_.task.motion_start_timestep);
  curr_motion_timestep_ = timestep_util_->timestep();
  motion_clip_progressing_ = true;
  const auto& task = config_.task;
  if (task.motion_start_timestep > 0 || task.motion_end_timestep) {
    const std::string end =
        task.motion_end_timestep && *task.motion_end_timestep != 0 ? std::to_string(*task.motion_end_timestep) : "end";
    log::info(log::cat("Starting motion clip from timestep ", task.motion_start_timestep, " to ", end),
              log::Color::kBlue);
  } else {
    log::info("Starting motion clip", log::Color::kBlue);
  }
}

void WholeBodyTrackingPolicy::dispatch_command(StateCommand command) {
  if (command == StateCommand::START_MOTION_CLIP) {
    handle_start_motion_clip();
  } else {
    BasePolicy::dispatch_command(command);
  }
}

}  // namespace holosoma
