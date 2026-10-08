// Policy state machine shared by locomotion and whole-body tracking
// (holosoma_inference/policies/base.py BasePolicy).
//
// One control tick (policy_action) picks the joint target from the current
// mode (init ramp, manual hold, or policy), applies joint offsets and the
// operator kp level, and publishes the command. Every model given in
// task.model_path is loaded up front and can be switched at runtime.
#pragma once

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "holosoma_cpp/commands.hpp"
#include "holosoma_cpp/config.hpp"
#include "holosoma_cpp/input.hpp"
#include "holosoma_cpp/observation.hpp"
#include "holosoma_cpp/onnx_model.hpp"
#include "holosoma_cpp/robot_interface.hpp"
#include "holosoma_cpp/timing.hpp"

namespace holosoma {

// Hardware and inputs shared by every policy of one process (the Python
// `_shared_hardware_source` of the primary policy).
struct PolicyContext {
  RobotInterface* robot = nullptr;
  VelocityProvider* velocity_input = nullptr;
  CommandProvider* command_provider = nullptr;
  double kp_level = 1.0;  // operator gain scale (interface.kp_level)
  double kd_level = 1.0;
  bool kill_requested = false;
};

// Target/gain override returned by a manual (non-policy) mode.
struct ManualCommand {
  std::vector<double> q;
  std::optional<std::vector<double>> kp;
  std::optional<std::vector<double>> kd;
};

class BasePolicy {
 public:
  BasePolicy(InferenceConfig config, PolicyContext& context);
  virtual ~BasePolicy();
  BasePolicy(const BasePolicy&) = delete;
  BasePolicy& operator=(const BasePolicy&) = delete;

  // Loads every model and activates the first one. Call once after construction.
  void initialize();

  // ---- loop hooks (BasePolicy.run) ----
  virtual void apply_velocity(const VelCmd& vc);
  virtual void dispatch_command(StateCommand command);
  virtual void update_phase_time();
  bool use_phase() const { return use_phase_; }
  // One control tick: compute the command for `state` and publish it.
  void policy_action(const RobotState& state);
  virtual void print_control_status() const;

  // ---- mode handlers (also driven by dual mode) ----
  virtual void handle_start_policy();
  virtual void handle_stop_policy();
  virtual void handle_init_state();
  void init_phase_components();
  // Non-interactive start: run the policy without the start handler (Python's no-TTY behaviour).
  void force_policy_active() { use_policy_action_ = true; }

  // ---- introspection ----
  const InferenceConfig& config() const { return config_; }
  bool use_policy_action() const { return use_policy_action_; }
  bool get_ready_state() const { return get_ready_state_; }
  int num_models() const { return static_cast<int>(slots_.size()); }
  int active_index() const { return active_index_; }
  std::string active_model_name() const;
  LatencyTracker& latency() { return latency_; }
  const JointCommand& last_command() const { return command_; }
  // Actor observation of the last tick that ran inference (empty before the first one).
  const std::vector<float>& last_observation() const { return last_obs_; }
  bool inference_ran() const { return inference_ran_; }
  const std::vector<double>& joint_kp() const;
  const std::vector<double>& joint_kd() const;
  std::string kind() const { return kind_; }

 protected:
  struct ModelSlot {
    std::string path;
    std::unique_ptr<OnnxModel> model;
    std::vector<double> kp;  // joint order
    std::vector<double> kd;
    std::string gain_source;
  };

  // Per-model setup: validate the graph signature and read metadata. Subclasses extend it.
  virtual void setup_policy(ModelSlot& slot);
  virtual void on_policy_switched() {}
  // Term names this policy can produce (validated against the observation config).
  virtual std::set<std::string> available_terms() const;
  virtual void compute_terms(const RobotState& state, TermValues& terms);
  // Runs the model and returns the scaled action (policy-relative joint targets).
  virtual std::vector<double> rl_inference(const RobotState& state);
  virtual std::optional<ManualCommand> manual_command(const RobotState& state);
  virtual std::vector<double> init_target(const RobotState& state);

  void activate_policy(int index, bool announce);
  // Builds actor_obs for this tick into last_obs_ (updates history).
  const std::vector<float>& prepare_actor_obs(const RobotState& state);
  void print_observations(const std::vector<float>& obs) const;
  void resolve_gains(ModelSlot& slot) const;
  ModelSlot& active_slot() { return *slots_.at(static_cast<size_t>(active_index_)); }
  const ModelSlot& active_slot() const { return *slots_.at(static_cast<size_t>(active_index_)); }
  static std::vector<double> clip_action(const std::vector<float>& raw);

  InferenceConfig config_;
  PolicyContext& context_;
  std::string kind_ = "BasePolicy";
  int num_dofs_ = 0;
  std::vector<double> default_dof_angles_;
  std::vector<double> joint_offsets_;
  ObservationBuilder obs_builder_;

  std::vector<std::unique_ptr<ModelSlot>> slots_;
  int active_index_ = 0;

  // Command state (BasePolicy._init_command_components)
  bool use_policy_action_ = false;
  bool get_ready_state_ = false;
  int init_count_ = 0;
  double desired_base_height_ = 0.75;
  double lin_vel_command_[2] = {0.0, 0.0};
  double ang_vel_command_ = 0.0;
  double stand_command_ = 0.0;
  double base_height_command_ = 0.75;

  // Gait phase
  bool use_phase_ = false;
  double phase_[2] = {0.0, 0.0};
  double phase_dt_ = 0.0;

  // Last raw action (clipped, float32 values) and its scaled version
  std::vector<double> last_policy_action_;
  std::vector<double> scaled_policy_action_;

  LatencyTracker latency_;
  JointCommand command_;
  std::vector<float> last_obs_;
  bool inference_ran_ = false;
};

// Robot-config gains (motor order) converted to joint order.
std::vector<double> motor_to_joint(const RobotConfig& robot, const std::vector<double>& motor_values);

// Parses a JSON array of numbers from ONNX metadata.
std::optional<std::vector<double>> metadata_number_list(const OnnxModel& model, const std::string& key);

}  // namespace holosoma
