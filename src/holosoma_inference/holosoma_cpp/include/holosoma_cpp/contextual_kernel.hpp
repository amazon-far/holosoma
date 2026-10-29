#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "holosoma_cpp/contextual_types.hpp"
#include "holosoma_cpp/onnx_model.hpp"

namespace holosoma::contextual {

struct State {
  std::vector<double> q, dq;
  // IMU-body orientation. Must be finite and nonzero; it is normalised here.
  std::array<double, 4> quat{1, 0, 0, 0};
  std::array<double, 3> omega{};
  // Optional gravity already projected into the IMU body frame (the same
  // quantity every other policy reads as projected_gravity). It replaces the
  // quaternion-derived vector and then takes the same imu_to_obs_joint_chain
  // rotation, so both sources reach the model in one frame.
  std::optional<std::array<double, 3>> gravity;
};

struct Spec {
  std::vector<std::string> joints;
  std::vector<double> defaults;
  std::optional<std::vector<double>> kp;
  std::optional<std::vector<double>> kd;
  double action_scale = 1;
  int command_dim = 15, height = 58, width = 87;
  bool upright = false, zero_omega = false;

  // Promote before multiplying: positive int dimensions can exceed INT_MAX pixels.
  std::size_t depth_frame_size() const { return static_cast<std::size_t>(height) * static_cast<std::size_t>(width); }
};

using Options = ContextualConfig;

// Owns only inference and temporal policy state. No transport, SDK, ROS clock,
// or implicit CPU fallback. A caller supplies monotonic time for key
// transitions.
class Policy {
 public:
  Policy(std::shared_ptr<OnnxModel> model, Spec spec, Options options = {});
  void Reset();
  void SetSpeed(int speed);
  void SetCommand(int index);
  void SetFixedArm(int side);
  void SetDz(bool enabled);
  // Persistent stand gate, independent of dZ and every new model prediction.
  // `continuation` lets a climb already in progress run on through an
  // unpermitted stand; it never starts one.
  void SetClimbAtStandAllowed(bool allowed, bool continuation = false);
  int SetVelocity(double vx, double vy, double wz, double deadzone = .1);
  void RequestKey(const std::string &key, double now);
  bool UpdateTransition(double now);
  void CancelTransition();
  std::string FinishProfile();
  std::vector<float> Observation(const State &state) const;
  std::vector<double> Target(const State &state, const std::optional<std::vector<float>> &depth, double now);

  const Spec &spec() const { return spec_; }
  const std::vector<double> &kp() const { return kp_; }
  const std::vector<double> &kd() const { return kd_; }
  const std::vector<int> &real_to_model() const { return real_to_model_; }
  const std::vector<int> &model_to_real() const { return model_to_real_; }
  const std::vector<float> &observation() const { return obs_; }
  const std::vector<float> &depth() const { return depth_; }
  const std::vector<int64_t> &depth_shape() const { return depth_shape_; }
  const std::vector<float> &action() const { return action_; }
  const std::vector<float> &scaled() const { return scaled_; }
  bool has_climb_head() const { return climb_output_.has_value(); }
  bool dz() const { return dz_; }
  bool climbing() const { return climbing_; }
  float climb_pred() const { return climb_pred_; }
  int active_command() const { return command_; }
  int effective_command() const { return climbing_ ? 14 : command_; }
  int fixed_arm() const { return fixed_arm_; }
  int speed() const { return speed_; }
  int calls() const { return calls_; }
  const std::string &key() const { return key_; }
  std::optional<int> pending() const { return pending_; }
  double release_time() const { return release_time_; }
  const char *provider() const { return model_->provider(); }

 private:
  void ResolveClimb();
  bool StandBlocked() const;
  bool StandHeld() const;
  bool ProbeClimb() const;
  void ApplyFixedArm();
  int CommandCode(const std::string &name) const;
  std::string VelocityName(double vx, double vy, double wz, double deadzone) const;
  void Run();
  Spec spec_;
  Options options_;
  std::shared_ptr<OnnxModel> model_;
  std::vector<int64_t> depth_shape_;
  std::optional<size_t> climb_output_;
  std::vector<int> real_to_model_, model_to_real_;
  std::vector<double> kp_, kd_, scale_;
  std::vector<float> command_vector_, obs_, depth_, action_, scaled_;
  std::deque<std::vector<float>> frames_;
  int history_ = 0, command_ = 0, speed_ = 0, fixed_arm_ = 0, calls_ = 0;
  bool dz_ = false, climbing_ = false;
  bool climb_at_stand_allowed_ = false, climb_continuation_ = false;
  float climb_pred_ = 0;
  // Whether climb_pred_ was read with column 14 set (gated entry needs it).
  bool pred_under_climb_ = false;
  std::string key_ = "z";
  std::optional<int> pending_;
  double release_time_ = 0;
  // Resolved imu_to_obs_joint_chain: (runtime joint index, quaternion component 1..3).
  std::vector<std::pair<size_t, int>> imu_to_obs_chain_;
};
}  // namespace holosoma::contextual
