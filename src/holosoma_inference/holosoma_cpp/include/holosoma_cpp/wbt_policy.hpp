// Whole-body tracking with the motion clip baked into the ONNX graph
// (holosoma_inference/policies/wbt.py WholeBodyTrackingPolicy).
//
// The graph maps (obs, time_step) to (actions, joint_pos, joint_vel,
// ref_quat_xyzw). The reference outputs of one step become the motion command
// observation of the next step, as in the Python runtime.
#pragma once

#include <memory>

#include "holosoma_cpp/policy.hpp"
#include "holosoma_cpp/urdf_kinematics.hpp"

namespace holosoma {

class WholeBodyTrackingPolicy : public BasePolicy {
 public:
  WholeBodyTrackingPolicy(InferenceConfig config, PolicyContext& context);

  void dispatch_command(StateCommand command) override;
  void handle_start_policy() override;
  void handle_stop_policy() override;

  bool motion_clip_progressing() const { return motion_clip_progressing_; }
  int motion_timestep() const { return curr_motion_timestep_; }
  double robot_yaw_offset() const { return robot_yaw_offset_; }
  double motion_yaw_offset() const { return motion_yaw_offset_; }

 protected:
  void setup_policy(ModelSlot& slot) override;
  void on_policy_switched() override;
  std::set<std::string> available_terms() const override;
  void compute_terms(const RobotState& state, TermValues& terms) override;
  std::vector<double> rl_inference(const RobotState& state) override;
  std::optional<ManualCommand> manual_command(const RobotState& state) override;
  std::vector<double> init_target(const RobotState& state) override;

 private:
  struct MotionSlot {
    std::unique_ptr<UrdfChain> kinematics;
    std::vector<double> motion_command_0;  // joint_pos ++ joint_vel at the start timestep
    math::Quat ref_quat_xyzw_0{0.0, 0.0, 0.0, 1.0};
    std::optional<std::vector<double>> per_joint_action_scale;
    std::string action_scale_source;
  };

  MotionSlot& motion_slot() { return motion_slots_.at(static_cast<size_t>(active_index_)); }
  const MotionSlot& motion_slot() const { return motion_slots_.at(static_cast<size_t>(active_index_)); }
  std::optional<std::vector<double>> configure_action_scale(const OnnxModel& model, std::string& source) const;
  math::Quat ref_body_orientation(const RobotState& state) const;
  void capture_robot_yaw_offset();
  void set_motion_timestep();
  void handle_start_motion_clip();
  void read_reference_outputs(const OnnxModel& model, std::vector<double>& motion_command,
                              math::Quat& ref_quat_xyzw) const;

  std::vector<MotionSlot> motion_slots_;

  bool motion_clip_progressing_ = false;
  int curr_motion_timestep_ = 0;
  std::vector<double> motion_command_t_;
  math::Quat ref_quat_xyzw_t_{0.0, 0.0, 0.0, 1.0};

  std::unique_ptr<ClockSource> clock_source_;
  std::unique_ptr<MotionClock> motion_clock_;
  std::unique_ptr<TimestepUtil> timestep_util_;
  bool use_sim_time_ = false;

  bool stiff_hold_active_ = true;
  std::vector<double> stiff_hold_q_;
  std::vector<double> stiff_hold_kp_;
  std::vector<double> stiff_hold_kd_;
  double robot_yaw_offset_ = 0.0;
  double motion_yaw_offset_ = 0.0;
};

}  // namespace holosoma
