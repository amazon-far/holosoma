#include "holosoma_cpp/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>

namespace holosoma {
namespace {

namespace fs = std::filesystem;

// Reads one YAML mapping and rejects keys that were neither consumed nor explicitly ignored.
class MapReader {
 public:
  MapReader(const YAML::Node& node, std::string path) : node_(node), path_(std::move(path)) {
    if (!node_.IsMap()) {
      throw ConfigError(path_ + ": expected a mapping");
    }
  }

  template <typename T>
  void required(const char* key, T& out) {
    const YAML::Node value = take(key);
    if (!value || value.IsNull()) {
      throw ConfigError(field(key) + ": required field is missing");
    }
    out = convert<T>(value, field(key));
  }

  // Leaves `out` at its default when the key is absent.
  template <typename T>
  void optional(const char* key, T& out) {
    const YAML::Node value = take(key);
    if (!value) {
      return;
    }
    if (value.IsNull()) {
      throw ConfigError(field(key) + ": null is not allowed");
    }
    out = convert<T>(value, field(key));
  }

  // Nullable field: absent or null -> nullopt.
  template <typename T>
  void nullable(const char* key, std::optional<T>& out) {
    const YAML::Node value = take(key);
    if (!value || value.IsNull()) {
      out.reset();
      return;
    }
    out = convert<T>(value, field(key));
  }

  YAML::Node child(const char* key) { return take(key); }

  // Known Python field without a C++ consumer.
  void ignore(const char* key) { take(key); }

  void finish() const {
    std::vector<std::string> unknown;
    for (const auto& item : node_) {
      const std::string key = item.first.as<std::string>();
      if (consumed_.count(key) == 0) {
        unknown.push_back(key);
      }
    }
    if (!unknown.empty()) {
      std::ostringstream os;
      os << path_ << ": unknown field(s):";
      for (const auto& key : unknown) {
        os << " " << key;
      }
      throw ConfigError(os.str());
    }
  }

  std::string field(const char* key) const { return path_ + "." + key; }

 private:
  YAML::Node take(const char* key) {
    consumed_.insert(key);
    return node_[key];
  }

  template <typename T>
  static T convert(const YAML::Node& value, const std::string& where) {
    try {
      return value.as<T>();
    } catch (const YAML::Exception& e) {
      throw ConfigError(where + ": invalid value (" + e.what() + ")");
    }
  }

  YAML::Node node_;
  std::string path_;
  std::set<std::string> consumed_;
};

std::vector<std::string> parse_model_paths(const YAML::Node& value, const std::string& where) {
  std::vector<std::string> paths;
  if (!value || value.IsNull()) {
    return paths;
  }
  try {
    if (value.IsSequence()) {
      paths = value.as<std::vector<std::string>>();
    } else {
      paths.push_back(value.as<std::string>());
    }
  } catch (const YAML::Exception& e) {
    throw ConfigError(where + ": expected a path or a list of paths (" + e.what() + ")");
  }
  paths.erase(std::remove(paths.begin(), paths.end(), std::string()), paths.end());
  return paths;
}

bool is_remote_path(const std::string& path) {
  return path.rfind("wandb://", 0) == 0 || path.rfind("https://", 0) == 0 || path.rfind("http://", 0) == 0;
}

std::string resolve_path(const std::string& path, const std::string& base_dir) {
  if (path.empty() || is_remote_path(path)) {
    return path;
  }
  fs::path p(path);
  if (p.is_relative() && !base_dir.empty()) {
    p = fs::path(base_dir) / p;
  }
  return p.lexically_normal().string();
}

RobotConfig parse_robot(const YAML::Node& node, const std::string& path) {
  RobotConfig robot;
  MapReader r(node, path);
  r.required("robot_type", robot.robot_type);
  r.required("robot", robot.robot);
  r.required("default_dof_angles", robot.default_dof_angles);
  r.required("default_motor_angles", robot.default_motor_angles);
  r.required("motor2joint", robot.motor2joint);
  r.required("joint2motor", robot.joint2motor);
  r.required("dof_names", robot.dof_names);
  r.nullable("motor_kp", robot.motor_kp);
  r.nullable("motor_kd", robot.motor_kd);
  r.nullable("default_per_joint_action_scale", robot.default_per_joint_action_scale);
  r.optional("interp_gain_scale", robot.interp_gain_scale);
  r.nullable("stiff_startup_pos", robot.stiff_startup_pos);
  r.nullable("stiff_startup_kp", robot.stiff_startup_kp);
  r.nullable("stiff_startup_kd", robot.stiff_startup_kd);
  r.optional("sdk_type", robot.sdk_type);
  r.optional("message_type", robot.message_type);
  r.optional("num_motors", robot.num_motors);
  r.optional("num_joints", robot.num_joints);
  r.optional("torso_link_name", robot.torso_link_name);
  r.nullable("joint_offsets_deg", robot.joint_offsets_deg);

  const YAML::Node motion = r.child("motion");
  if (motion && !motion.IsNull()) {
    MapReader m(motion, r.field("motion"));
    m.optional("body_name_ref", robot.motion_body_name_ref);
    m.finish();
  }

  // Python fields that the C++ runtime does not consume.
  for (const char* key : {"dof_names_upper_body", "dof_names_lower_body", "motor_type", "left_hand_link_name",
                          "right_hand_link_name", "unitree_legged_const", "weak_motor_joint_index",
                          "dof_names_parallel_mech", "use_sensor", "num_upper_body_joints"}) {
    r.ignore(key);
  }
  r.finish();
  return robot;
}

ObservationConfig parse_observation(const YAML::Node& node, const std::string& path) {
  ObservationConfig obs;
  MapReader r(node, path);
  const YAML::Node obs_dict = r.child("obs_dict");
  if (!obs_dict || !obs_dict.IsMap()) {
    throw ConfigError(r.field("obs_dict") + ": required mapping is missing");
  }
  for (const auto& item : obs_dict) {
    obs.obs_dict.emplace_back(item.first.as<std::string>(), item.second.as<std::vector<std::string>>());
  }
  r.required("obs_dims", obs.obs_dims);
  r.required("obs_scales", obs.obs_scales);
  r.required("history_length_dict", obs.history_length_dict);
  r.finish();
  return obs;
}

TaskConfig parse_task(const YAML::Node& node, const std::string& path, const std::string& base_dir) {
  TaskConfig task;
  MapReader r(node, path);
  task.model_path = parse_model_paths(r.child("model_path"), r.field("model_path"));
  for (auto& model : task.model_path) {
    model = resolve_path(model, base_dir);
  }
  r.optional("rl_rate", task.rl_rate);
  r.optional("policy_action_scale", task.policy_action_scale);
  r.optional("action_scales_by_effort_limit_over_p_gain", task.action_scales_by_effort_limit_over_p_gain);
  r.optional("use_phase", task.use_phase);
  r.optional("gait_period", task.gait_period);
  r.optional("skip_stiff_prompt", task.skip_stiff_prompt);
  r.optional("domain_id", task.domain_id);
  r.optional("interface", task.interface);
  r.optional("velocity_input", task.velocity_input);
  r.optional("state_input", task.state_input);
  r.optional("auto_walk_on_vel_cmd", task.auto_walk_on_vel_cmd);
  r.optional("use_sim_time", task.use_sim_time);
  r.optional("desired_base_height", task.desired_base_height);
  r.optional("print_observations", task.print_observations);
  r.optional("motion_start_timestep", task.motion_start_timestep);
  r.nullable("motion_end_timestep", task.motion_end_timestep);

  // TaskConfig.__post_init__: resolve the use_keyboard/use_joystick/use_usb_joystick shortcuts.
  bool use_keyboard = false, use_joystick = false, use_usb_joystick = false;
  r.optional("use_keyboard", use_keyboard);
  r.optional("use_joystick", use_joystick);
  r.optional("use_usb_joystick", use_usb_joystick);
  const int shortcuts = int(use_keyboard) + int(use_joystick) + int(use_usb_joystick);
  if (shortcuts > 1) {
    throw ConfigError(path + ": cannot combine multiple input shortcuts (use_keyboard/use_joystick/use_usb_joystick)");
  }
  if (shortcuts == 1) {
    if (task.velocity_input != "keyboard" || task.state_input != "keyboard") {
      throw ConfigError(path +
                        ": cannot combine an input shortcut with velocity_input/state_input; use one or the other");
    }
    const std::string source = use_usb_joystick ? "joystick" : (use_joystick ? "interface" : "keyboard");
    task.velocity_input = source;
    task.state_input = source;
  }

  const YAML::Node depth = r.child("depth");
  if (depth && !depth.IsNull()) {
    MapReader d(depth, r.field("depth"));
    std::vector<std::string> topics;
    d.optional("topics", topics);
    if (!topics.empty()) {
      throw ConfigError(d.field("topics") + ": depth sensors are not supported by the C++ runtime");
    }
    for (const char* key : {"resized_height", "resized_width", "near_clip", "far_clip", "frame_delay_ms"}) {
      d.ignore(key);
    }
    d.finish();
  }

  const YAML::Node debug = r.child("debug");
  if (debug && !debug.IsNull()) {
    MapReader d(debug, r.field("debug"));
    d.optional("force_upright_imu", task.debug.force_upright_imu);
    d.optional("force_zero_angular_velocity", task.debug.force_zero_angular_velocity);
    d.optional("force_zero_action", task.debug.force_zero_action);
    bool record_rosbag = false;
    d.optional("record_rosbag", record_rosbag);
    if (record_rosbag) {
      throw ConfigError(d.field("record_rosbag") +
                        ": not supported by the C++ runtime; run `ros2 bag record` "
                        "separately");
    }
    d.ignore("rosbag_dir");
    d.ignore("rosbag_cpu");
    d.finish();
  }

  for (const char* key : {"joystick_type", "joystick_device", "ros_cmd_vel_topic", "ros_state_input_topic",
                          "ros_vel_timeout", "residual_upper_body_action"}) {
    r.ignore(key);
  }
  r.finish();
  return task;
}

RuntimeConfig parse_runtime(const YAML::Node& node, const std::string& path) {
  RuntimeConfig runtime;
  if (!node || node.IsNull()) {
    return runtime;
  }
  MapReader r(node, path);
  r.optional("state_timeout_s", runtime.state_timeout_s);
  r.optional("damping_kd", runtime.damping_kd);
  r.finish();
  return runtime;
}

InferenceConfig parse_document(const YAML::Node& node, const std::string& base_dir, const std::string& path,
                               const RuntimeConfig* inherited_runtime) {
  MapReader r(node, path);
  InferenceConfig config;
  config.robot = parse_robot(r.child("robot"), path + ".robot");
  config.observation = parse_observation(r.child("observation"), path + ".observation");
  config.task = parse_task(r.child("task"), path + ".task", base_dir);
  const YAML::Node runtime = r.child("runtime");
  if (inherited_runtime != nullptr) {
    if (runtime && !runtime.IsNull()) {
      throw ConfigError(path + ".runtime: runtime options apply to the whole process; set them on the primary");
    }
    config.runtime = *inherited_runtime;
  } else {
    config.runtime = parse_runtime(runtime, path + ".runtime");
  }
  const YAML::Node secondary = r.child("secondary");
  if (secondary && !secondary.IsNull()) {
    if (inherited_runtime != nullptr) {
      throw ConfigError(path + ".secondary: nested secondary policies are not supported");
    }
    config.secondary =
        std::make_shared<InferenceConfig>(parse_document(secondary, base_dir, path + ".secondary", &config.runtime));
  }
  r.finish();
  return config;
}

void check_size(size_t actual, size_t expected, const std::string& what) {
  if (actual != expected) {
    throw ConfigError(what + ": expected " + std::to_string(expected) + " values, got " + std::to_string(actual));
  }
}

void validate_one(const InferenceConfig& config, const std::string& path) {
  const RobotConfig& robot = config.robot;
  const auto joints = static_cast<size_t>(robot.num_joints);
  const auto motors = static_cast<size_t>(robot.num_motors);
  if (robot.num_joints <= 0 || robot.num_motors <= 0) {
    throw ConfigError(path + ".robot: num_joints and num_motors must be positive");
  }
  check_size(robot.dof_names.size(), joints, path + ".robot.dof_names");
  check_size(robot.default_dof_angles.size(), joints, path + ".robot.default_dof_angles");
  check_size(robot.joint2motor.size(), joints, path + ".robot.joint2motor");
  check_size(robot.motor2joint.size(), motors, path + ".robot.motor2joint");
  std::set<int> motor_ids;
  for (int m : robot.joint2motor) {
    if (m < 0 || m >= robot.num_motors || !motor_ids.insert(m).second) {
      throw ConfigError(path + ".robot.joint2motor: must map each joint to a distinct motor index in [0, num_motors)");
    }
  }
  std::set<std::string> names(robot.dof_names.begin(), robot.dof_names.end());
  if (names.size() != robot.dof_names.size()) {
    throw ConfigError(path + ".robot.dof_names: duplicate joint names");
  }
  if (robot.motor_kp) check_size(robot.motor_kp->size(), motors, path + ".robot.motor_kp");
  if (robot.motor_kd) check_size(robot.motor_kd->size(), motors, path + ".robot.motor_kd");
  if (robot.motor_kp.has_value() != robot.motor_kd.has_value()) {
    throw ConfigError(path + ".robot: motor_kp and motor_kd must be overridden together");
  }
  if (robot.stiff_startup_pos) check_size(robot.stiff_startup_pos->size(), joints, path + ".robot.stiff_startup_pos");
  if (robot.stiff_startup_kp) check_size(robot.stiff_startup_kp->size(), joints, path + ".robot.stiff_startup_kp");
  if (robot.stiff_startup_kd) check_size(robot.stiff_startup_kd->size(), joints, path + ".robot.stiff_startup_kd");
  if (robot.joint_offsets_deg) check_size(robot.joint_offsets_deg->size(), joints, path + ".robot.joint_offsets_deg");
  if (robot.default_per_joint_action_scale && robot.default_per_joint_action_scale->size() != 1) {
    check_size(robot.default_per_joint_action_scale->size(), joints, path + ".robot.default_per_joint_action_scale");
  }

  const ObservationConfig& obs = config.observation;
  for (const auto& [group, terms] : obs.obs_dict) {
    const auto history = obs.history_length_dict.find(group);
    if (history != obs.history_length_dict.end() && history->second < 1) {
      throw ConfigError(path + ".observation.history_length_dict." + group + ": must be >= 1");
    }
    for (const auto& term : terms) {
      if (obs.obs_dims.count(term) == 0) {
        throw ConfigError(path + ".observation.obs_dims: missing dimension for term '" + term + "'");
      }
      if (obs.obs_scales.count(term) == 0) {
        throw ConfigError(path + ".observation.obs_scales: missing scale for term '" + term + "'");
      }
    }
  }

  const TaskConfig& task = config.task;
  if (task.model_path.empty()) {
    throw ConfigError(path + ".task.model_path: at least one model path is required");
  }
  if (task.model_path.size() > 9) {
    throw ConfigError(path + ".task.model_path: at most nine model paths are supported");
  }
  for (const auto& model : task.model_path) {
    if (is_remote_path(model)) {
      throw ConfigError(path + ".task.model_path: '" + model +
                        "' is remote; download it first (the C++ runtime only loads local files)");
    }
  }
  if (!(task.rl_rate > 0.0)) throw ConfigError(path + ".task.rl_rate: must be positive");
  if (!(task.gait_period > 0.0)) throw ConfigError(path + ".task.gait_period: must be positive");
  for (const auto* source : {&task.velocity_input, &task.state_input}) {
    if (*source != "keyboard" && *source != "interface") {
      throw ConfigError(path + ".task: input source '" + *source +
                        "' is not supported by the C++ runtime (supported: keyboard, interface)");
    }
  }
  if (task.motion_start_timestep < 0) throw ConfigError(path + ".task.motion_start_timestep: must be >= 0");
}

}  // namespace

bool InferenceConfig::is_wbt() const {
  for (const auto& [group, terms] : observation.obs_dict) {
    if (group == "actor_obs") {
      return std::find(terms.begin(), terms.end(), "motion_command") != terms.end();
    }
  }
  return false;
}

InferenceConfig parse_inference_config(const YAML::Node& node, const std::string& base_dir) {
  return parse_document(node, base_dir, "config", nullptr);
}

void validate_inference_config(const InferenceConfig& config) {
  validate_one(config, "config");
  if (config.secondary) {
    validate_one(*config.secondary, "config.secondary");
    if (config.secondary->robot.num_joints != config.robot.num_joints ||
        config.secondary->robot.dof_names != config.robot.dof_names ||
        config.secondary->robot.joint2motor != config.robot.joint2motor) {
      throw ConfigError("config.secondary.robot: must describe the same robot as the primary (it shares the hardware)");
    }
  }
}

}  // namespace holosoma
