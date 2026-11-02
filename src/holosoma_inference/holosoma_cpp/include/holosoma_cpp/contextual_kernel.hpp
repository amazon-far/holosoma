#pragma once

#include <array>
#include <cstddef>
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

// One command interface for a depth-conditioned policy: how operator commands
// become the model's command input, how the observation is assembled, and how
// raw actions become joint targets. Policy selects one from the configured
// discretizer and forwards every call; callers never name a family.
//
// Shared contract for every family: runtime (Spec) joint order on every
// vector a caller reads or writes, model-slot -> runtime index in
// real_to_model(), the shared raw-action ceiling (action_clip.hpp) applied
// before scaling, and the clipped action fed back as the next observation's
// previous action. A setter a family cannot honour throws
// std::invalid_argument rather than being dropped.
class Family {
 public:
  virtual ~Family() = default;
  virtual void Reset() = 0;
  virtual void SetSpeed(int speed) = 0;
  virtual void SetCommand(int index) = 0;
  virtual void SetFixedArm(int side) = 0;
  virtual void SetDz(bool enabled) = 0;
  virtual void SetClimbAtStandAllowed(bool allowed, bool continuation) = 0;
  virtual int SetVelocity(double vx, double vy, double wz, double deadzone) = 0;
  virtual void RequestKey(const std::string &key, double now) = 0;
  virtual bool UpdateTransition(double now) = 0;
  virtual void CancelTransition() = 0;
  virtual std::vector<float> Observation(const State &state) const = 0;
  virtual std::vector<double> Target(const State &state, const std::optional<std::vector<float>> &depth,
                                     double now) = 0;

  // Runtime-order stance the targets are offsets from, and the gains to send.
  virtual const std::vector<double> &default_dof_pos() const = 0;
  virtual const std::vector<double> &kp() const = 0;
  virtual const std::vector<double> &kd() const = 0;
  // True when kp/kd came from the model rather than from Spec.
  virtual bool gains_from_model() const = 0;
  virtual const std::vector<int> &real_to_model() const = 0;
  virtual const std::vector<int> &model_to_real() const = 0;
  virtual const std::vector<float> &observation() const = 0;
  virtual const std::vector<float> &depth() const = 0;
  virtual const std::vector<int64_t> &depth_shape() const = 0;
  virtual const std::vector<float> &action() const = 0;
  virtual const std::vector<float> &scaled() const = 0;
  virtual bool has_climb_head() const = 0;
  virtual bool dz() const = 0;
  virtual bool climbing() const = 0;
  virtual float climb_pred() const = 0;
  // Primitive command index, or kNoPrimitive for a continuous command.
  virtual int active_command() const = 0;
  virtual int effective_command() const = 0;
  virtual int fixed_arm() const = 0;
  virtual int speed() const = 0;
  virtual int calls() const = 0;
  virtual const std::string &key() const = 0;
  virtual std::optional<int> pending() const = 0;
  virtual double release_time() const = 0;
};

// Reported by active_command()/effective_command() when the command is a
// continuous velocity rather than one of the primitive columns.
constexpr int kNoPrimitive = -1;

// Owns only inference and temporal policy state. No transport, SDK, ROS clock,
// or implicit CPU fallback. A caller supplies monotonic time for key
// transitions.
class Policy {
 public:
  Policy(std::shared_ptr<OnnxModel> model, Spec spec, Options options = {});
  void Reset() { family_->Reset(); }
  void SetSpeed(int speed) { family_->SetSpeed(speed); }
  void SetCommand(int index) { family_->SetCommand(index); }
  void SetFixedArm(int side) { family_->SetFixedArm(side); }
  void SetDz(bool enabled) { family_->SetDz(enabled); }
  // Persistent stand gate, independent of dZ and every new model prediction.
  // `continuation` lets a climb already in progress run on through an
  // unpermitted stand; it never starts one.
  void SetClimbAtStandAllowed(bool allowed, bool continuation = false) {
    family_->SetClimbAtStandAllowed(allowed, continuation);
  }
  int SetVelocity(double vx, double vy, double wz, double deadzone = .1) {
    return family_->SetVelocity(vx, vy, wz, deadzone);
  }
  void RequestKey(const std::string &key, double now) { family_->RequestKey(key, now); }
  bool UpdateTransition(double now) { return family_->UpdateTransition(now); }
  void CancelTransition() { family_->CancelTransition(); }
  std::string FinishProfile() { return model_->finish_profile(); }
  std::vector<float> Observation(const State &state) const { return family_->Observation(state); }
  std::vector<double> Target(const State &state, const std::optional<std::vector<float>> &depth, double now) {
    return family_->Target(state, depth, now);
  }

  const Spec &spec() const { return spec_; }
  // True when the configured command interface is a continuous velocity.
  bool continuous() const { return !options_.discretizer.has_value(); }
  const std::vector<double> &default_dof_pos() const { return family_->default_dof_pos(); }
  const std::vector<double> &kp() const { return family_->kp(); }
  const std::vector<double> &kd() const { return family_->kd(); }
  bool gains_from_model() const { return family_->gains_from_model(); }
  const std::vector<int> &real_to_model() const { return family_->real_to_model(); }
  const std::vector<int> &model_to_real() const { return family_->model_to_real(); }
  const std::vector<float> &observation() const { return family_->observation(); }
  const std::vector<float> &depth() const { return family_->depth(); }
  const std::vector<int64_t> &depth_shape() const { return family_->depth_shape(); }
  const std::vector<float> &action() const { return family_->action(); }
  const std::vector<float> &scaled() const { return family_->scaled(); }
  bool has_climb_head() const { return family_->has_climb_head(); }
  bool dz() const { return family_->dz(); }
  bool climbing() const { return family_->climbing(); }
  float climb_pred() const { return family_->climb_pred(); }
  int active_command() const { return family_->active_command(); }
  int effective_command() const { return family_->effective_command(); }
  int fixed_arm() const { return family_->fixed_arm(); }
  int speed() const { return family_->speed(); }
  int calls() const { return family_->calls(); }
  const std::string &key() const { return family_->key(); }
  std::optional<int> pending() const { return family_->pending(); }
  double release_time() const { return family_->release_time(); }
  const char *provider() const { return model_->provider(); }

 private:
  Spec spec_;
  Options options_;
  std::shared_ptr<OnnxModel> model_;
  std::unique_ptr<Family> family_;
};
}  // namespace holosoma::contextual
