// Inference configuration (holosoma_inference/config/config_types).
//
// Presets live in config/inference/*.yaml and are exported from the Python
// registry by scripts/export_presets.py. Field names match the Python
// dataclasses one to one; unknown keys are rejected so a renamed or new Python
// field cannot be silently ignored here.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace YAML {
class Node;
}

namespace holosoma {

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RobotConfig {
  std::string robot_type;
  std::string robot;
  std::vector<double> default_dof_angles;
  std::vector<double> default_motor_angles;
  std::vector<int> motor2joint;
  std::vector<int> joint2motor;
  std::vector<std::string> dof_names;
  std::optional<std::vector<double>> motor_kp;
  std::optional<std::vector<double>> motor_kd;
  std::optional<std::vector<double>> default_per_joint_action_scale;
  double interp_gain_scale = 1.0;
  std::optional<std::vector<double>> stiff_startup_pos;
  std::optional<std::vector<double>> stiff_startup_kp;
  std::optional<std::vector<double>> stiff_startup_kd;
  std::string sdk_type = "unitree";
  std::string message_type = "HG";
  int num_motors = 29;
  int num_joints = 29;
  std::string torso_link_name = "torso_link";
  std::vector<std::string> motion_body_name_ref;  // robot.motion["body_name_ref"]
  std::optional<std::vector<double>> joint_offsets_deg;
};

struct ObservationConfig {
  // Group name -> term names in configuration order (groups keep YAML order).
  std::vector<std::pair<std::string, std::vector<std::string>>> obs_dict;
  std::map<std::string, int> obs_dims;
  std::map<std::string, double> obs_scales;
  std::map<std::string, int> history_length_dict;
};

struct DebugConfig {
  bool force_upright_imu = false;
  bool force_zero_angular_velocity = false;
  bool force_zero_action = false;
};

struct TaskConfig {
  std::vector<std::string> model_path;
  double rl_rate = 50.0;
  double policy_action_scale = 0.25;
  bool action_scales_by_effort_limit_over_p_gain = false;
  bool use_phase = true;
  double gait_period = 1.0;
  bool skip_stiff_prompt = false;
  int domain_id = 0;
  std::string interface = "auto";
  std::string velocity_input = "keyboard";
  std::string state_input = "keyboard";
  bool auto_walk_on_vel_cmd = false;
  bool use_sim_time = false;
  double desired_base_height = 0.75;
  bool print_observations = false;
  int motion_start_timestep = 0;
  std::optional<int> motion_end_timestep;
  DebugConfig debug;
};

// Options that only exist in the C++ runtime (YAML section `runtime`, CLI `--runtime.*`).
struct RuntimeConfig {
  // Enter damping if the newest LowState is older than this while commanding (0 disables).
  double state_timeout_s = 0.5;
  // Derivative gain of the damping command (kp = 0) sent on faults and at exit.
  double damping_kd = 5.0;
};

struct InferenceConfig {
  RobotConfig robot;
  ObservationConfig observation;
  TaskConfig task;
  RuntimeConfig runtime;
  std::shared_ptr<InferenceConfig> secondary;

  // True when the actor observations contain motion_command (the Python policy-class heuristic).
  bool is_wbt() const;
};

// Parses a full inference document. Relative model paths resolve against `base_dir`.
InferenceConfig parse_inference_config(const YAML::Node& node, const std::string& base_dir);

// Checks cross-field invariants (lengths, supported inputs, ...). Throws ConfigError.
void validate_inference_config(const InferenceConfig& config);

}  // namespace holosoma
