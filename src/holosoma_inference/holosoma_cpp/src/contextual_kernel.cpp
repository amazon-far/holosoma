#include "holosoma_cpp/contextual_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include "contextual_family.hpp"
#include "holosoma_cpp/action_clip.hpp"

namespace holosoma::contextual {
namespace detail {
Json Metadata(const OnnxModel &model) {
  Json metadata = Json::object();
  for (const auto &[key, value] : model.metadata()) {
    auto parsed = Json::parse(value, nullptr, false);
    metadata[key] = parsed.is_discarded() ? Json(value) : parsed;
  }
  return metadata;
}
std::vector<double> Numbers(const Json &v) {
  if (!v.is_array()) return {v.get<double>()};
  std::vector<double> values;
  for (const auto &item : v) {
    auto flattened = Numbers(item);
    values.insert(values.end(), flattened.begin(), flattened.end());
  }
  return values;
}
std::vector<std::string> JointNames(const Json &value) {
  if (!value.is_string()) return value.get<std::vector<std::string>>();
  std::vector<std::string> names;
  std::stringstream stream(value.get<std::string>());
  std::string item;
  while (std::getline(stream, item, ','))
    if (!item.empty()) names.push_back(item);
  return names;
}
void JointPermutation(const std::vector<std::string> &names, const std::vector<std::string> &runtime,
                      std::vector<int> &real_to_model, std::vector<int> &model_to_real) {
  const size_t n = runtime.size();
  const std::set<std::string> unique(names.begin(), names.end());
  if (names.size() != n || unique.size() != n || unique != std::set<std::string>(runtime.begin(), runtime.end()))
    throw std::runtime_error("must name every runtime joint exactly once");
  real_to_model.clear();
  model_to_real.assign(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const int real = static_cast<int>(std::find(runtime.begin(), runtime.end(), names[i]) - runtime.begin());
    real_to_model.push_back(real);
    model_to_real[static_cast<size_t>(real)] = static_cast<int>(i);
  }
}
}  // namespace detail

namespace {
using detail::Finite;
using detail::Json;
const std::array<std::map<std::string, int>, 3> kCodes{{
    {{"stand", 0},
     {"forward", 1},
     {"left_45", 2},
     {"turn_left", 3},
     {"right_45", 4},
     {"turn_right", 5},
     {"back", 11},
     {"leftward", 12},
     {"rightward", 13}},
    {{"stand", 0}, {"forward", 6}, {"left_45", 7}, {"left_90", 8}, {"right_45", 9}, {"right_90", 10}, {"back", 11}},
    {{"stand", 0}, {"forward", 12}, {"left_45", 13}, {"left_90", 13}, {"right_45", 14}, {"right_90", 14}, {"back", 11}},
}};
std::string Simple(double forward, double strafe, double turn) {
  if (strafe == 0 && forward == 0) return turn < 0 ? "turn_left" : turn > 0 ? "turn_right" : "stand";
  if (std::abs(forward) >= std::abs(strafe)) {
    if (forward > 0) return turn < 0 ? "left_45" : turn > 0 ? "right_45" : "forward";
    return "back";
  }
  return strafe < 0 ? "leftward" : "rightward";
}
std::array<double, 4> Multiply(const std::array<double, 4> &a, const std::array<double, 4> &b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
// Express world-or-parent vector v in the frame of unit quaternion q (q* v q).
std::array<double, 3> RotateInverse(const std::array<double, 4> &q, const std::array<double, 3> &v) {
  const auto r = Multiply(Multiply({q[0], -q[1], -q[2], -q[3]}, {0, v[0], v[1], v[2]}), q);
  return {r[1], r[2], r[3]};
}
const std::vector<std::pair<std::string, std::string>> kG1WaistChain{
    {"waist_yaw_joint", "z"}, {"waist_roll_joint", "x"}, {"waist_pitch_joint", "y"}};
}  // namespace

namespace detail {
Chain ResolveChain(const std::vector<std::pair<std::string, std::string>> &chain,
                   const std::vector<std::string> &joints) {
  Chain resolved;
  for (const auto &[name, axis] : chain) {
    const auto found = std::find(joints.begin(), joints.end(), name);
    if (found == joints.end())
      throw std::invalid_argument("imu_to_obs_joint_chain names " + name +
                                  ", which is not a runtime joint; set the chain for this robot, or [] for IMU-frame "
                                  "gravity");
    const int component = axis == "x" ? 1 : axis == "y" ? 2 : axis == "z" ? 3 : 0;
    if (!component) throw std::invalid_argument("imu_to_obs_joint_chain axis for " + name + " must be x, y or z");
    resolved.emplace_back(static_cast<size_t>(found - joints.begin()), component);
  }
  return resolved;
}
void CheckState(const State &state, size_t n) {
  if (state.q.size() != n || state.dq.size() != n) throw std::invalid_argument("state joint count mismatch");
  Finite(state.q, "q");
  Finite(state.dq, "dq");
  Finite(state.omega, "omega");
  Finite(state.quat, "orientation");
  if (state.gravity) Finite(*state.gravity, "gravity");
  // An IMU that reports no orientation is a fault, never "upright".
  const double norm = std::sqrt(std::inner_product(state.quat.begin(), state.quat.end(), state.quat.begin(), 0.0));
  if (!(norm >= 1e-8)) throw std::invalid_argument("orientation must be nonzero");
}
std::array<double, 3> ObservationGravity(const State &state, bool upright, const Chain &chain_joints) {
  std::array<double, 3> gravity{0, 0, -1};
  if (upright) return gravity;
  // Gravity in the IMU body frame, from the caller or from the orientation.
  if (state.gravity)
    gravity = *state.gravity;
  else {
    const double norm = std::sqrt(std::inner_product(state.quat.begin(), state.quat.end(), state.quat.begin(), 0.0));
    auto quat = state.quat;
    for (auto &v : quat) v /= norm;
    gravity = RotateInverse(quat, {0, 0, -1});
  }
  // Then from the IMU frame into the observation frame through imu_to_obs_joint_chain.
  std::array<double, 4> chain{1, 0, 0, 0};
  for (const auto &[joint, component] : chain_joints) {
    const double half = state.q[joint] * .5;
    std::array<double, 4> rotation{std::cos(half), 0, 0, 0};
    rotation[component] = std::sin(half);
    chain = Multiply(chain, rotation);
  }
  return RotateInverse(chain, gravity);
}
}  // namespace detail

namespace {
using detail::Chain;

enum class ClimbEnter { kDzOrPred, kDzAndPred, kDzOnly };
enum class ClimbExit { kBothOff, kEitherOff, kPredOff };

ClimbEnter ParseClimbEnter(const std::string &name) {
  if (name == "dz_or_pred") return ClimbEnter::kDzOrPred;
  if (name == "dz_and_pred") return ClimbEnter::kDzAndPred;
  if (name == "dz_only") return ClimbEnter::kDzOnly;
  throw std::invalid_argument("unknown climb_enter");
}
ClimbExit ParseClimbExit(const std::string &name) {
  if (name == "both_off") return ClimbExit::kBothOff;
  if (name == "either_off") return ClimbExit::kEitherOff;
  if (name == "pred_off") return ClimbExit::kPredOff;
  throw std::invalid_argument("unknown climb_exit");
}

// A policy trained on the 15 one-hot primitive commands, reached from a
// velocity through a discretizer, with an optional climb head.
class PrimitiveFamily final : public Family {
 public:
  PrimitiveFamily(std::shared_ptr<OnnxModel> model, Spec spec, Options options);
  void Reset() override;
  void SetSpeed(int speed) override;
  void SetCommand(int index) override;
  void SetFixedArm(int side) override;
  void SetDz(bool enabled) override;
  void SetClimbAtStandAllowed(bool allowed, bool continuation) override;
  int SetVelocity(double vx, double vy, double wz, double deadzone) override;
  void RequestKey(const std::string &key, double now) override;
  bool UpdateTransition(double now) override;
  void CancelTransition() override;
  std::vector<float> Observation(const State &state) const override;
  std::vector<double> Target(const State &state, const std::optional<std::vector<float>> &depth, double now) override;

  const std::vector<double> &default_dof_pos() const override { return spec_.defaults; }
  const std::vector<double> &kp() const override { return kp_; }
  const std::vector<double> &kd() const override { return kd_; }
  bool gains_from_model() const override { return !spec_.kp; }
  const std::vector<int> &real_to_model() const override { return real_to_model_; }
  const std::vector<int> &model_to_real() const override { return model_to_real_; }
  const std::vector<float> &observation() const override { return obs_; }
  const std::vector<float> &depth() const override { return depth_; }
  const std::vector<int64_t> &depth_shape() const override { return depth_shape_; }
  // Primitive exports declare no depth preprocessing; the transport's own applies.
  std::optional<DepthContract> depth_contract() const override { return std::nullopt; }
  const std::vector<float> &action() const override { return action_; }
  const std::vector<float> &scaled() const override { return scaled_; }
  bool has_climb_head() const override { return climb_output_.has_value(); }
  bool dz() const override { return dz_; }
  bool climbing() const override { return climbing_; }
  float climb_pred() const override { return climb_pred_; }
  int active_command() const override { return command_; }
  int effective_command() const override { return climbing_ ? 14 : command_; }
  int fixed_arm() const override { return fixed_arm_; }
  int speed() const override { return speed_; }
  int calls() const override { return calls_; }
  const std::string &key() const override { return key_; }
  std::optional<int> pending() const override { return pending_; }
  double release_time() const override { return release_time_; }

 private:
  void ResolveClimb();
  bool StandBlocked() const;
  bool StandHeld() const;
  bool ProbeClimb() const;
  bool Gated() const { return climb_enter_ != ClimbEnter::kDzOrPred || climb_exit_ != ClimbExit::kBothOff; }
  void ApplyFixedArm();
  int CommandCode(const std::string &name) const;
  std::string VelocityName(double vx, double vy, double wz, double deadzone) const;
  void Run();
  Spec spec_;
  Options options_;
  std::string discretizer_;
  ClimbEnter climb_enter_ = ClimbEnter::kDzOrPred;
  ClimbExit climb_exit_ = ClimbExit::kBothOff;
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
  Chain imu_to_obs_chain_;
};

PrimitiveFamily::PrimitiveFamily(std::shared_ptr<OnnxModel> model, Spec spec, Options options)
    : spec_(std::move(spec)),
      options_(std::move(options)),
      discretizer_(*options_.discretizer),
      model_(std::move(model)) {
  const size_t n = spec_.joints.size();
  if (spec_.command_dim < 15) throw std::invalid_argument("invalid contextual policy spec");
  if (discretizer_ != "simple" && discretizer_ != "nearest_primitive")
    throw std::invalid_argument("unknown velocity discretizer");
  climb_enter_ = ParseClimbEnter(options_.climb_enter);
  climb_exit_ = ParseClimbExit(options_.climb_exit);
  // Entering must satisfy staying, or the gait flips on every resolve.
  if (climb_enter_ != ClimbEnter::kDzAndPred && climb_exit_ != ClimbExit::kBothOff)
    throw std::invalid_argument("climb_enter " + options_.climb_enter + " needs climb_exit both_off");
  if (discretizer_ != "simple" && (!std::isfinite(options_.yaw_request_gain) || options_.yaw_request_gain <= 0 ||
                                   !std::isfinite(options_.switch_margin) || options_.switch_margin < 0))
    throw std::invalid_argument("invalid nearest-primitive parameters");
  if (discretizer_ == "nearest_primitive") {
    for (const auto *name : {"stand", "forward", "left_45", "right_45", "turn_left", "turn_right"}) {
      const auto found = options_.velocity_primitives.find(name);
      if (found == options_.velocity_primitives.end() || found->second.size() != 2)
        throw std::invalid_argument("nearest_primitive requires all calibrated velocity pairs");
      Finite(found->second, "velocity primitive");
    }
    for (double scale : {options_.forward_distance_scale, options_.yaw_distance_scale})
      if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("invalid velocity distance scale");
    if (options_.in_place_yaw_threshold &&
        (!std::isfinite(*options_.in_place_yaw_threshold) || *options_.in_place_yaw_threshold < 0))
      throw std::invalid_argument("invalid in-place yaw threshold");
  }
  imu_to_obs_chain_ = detail::ResolveChain(options_.imu_to_obs_joint_chain.value_or(kG1WaistChain), spec_.joints);
  Json metadata = detail::Metadata(*model_);
  if (metadata.contains("policy_kind") && metadata["policy_kind"] == "hybrid_velocity")
    throw std::runtime_error(model_->path() +
                             ": a model trained on a continuous velocity command needs discretizer: null");
  std::map<std::string, std::vector<int64_t>> shapes;
  for (const auto &input : model_->inputs()) {
    const auto &name = input.name;
    if (name != "obs" && name != "depth_image" && name != "time_step")
      throw std::runtime_error("unsupported contextual input: " + name);
    shapes[name] = input.shape;
  }
  const auto width = static_cast<int64_t>(6 + n * 3 + spec_.command_dim);
  if (!shapes.count("obs") || shapes.at("obs").size() != 2 || shapes.at("obs").back() != width ||
      (shapes.at("obs")[0] != 1 && shapes.at("obs")[0] != -1))
    throw std::runtime_error("obs shape does not match contextual contract");
  if (!shapes.count("depth_image")) throw std::runtime_error("missing depth_image input");
  depth_shape_ = shapes.at("depth_image");
  if (depth_shape_.size() != 3 && depth_shape_.size() != 4)
    throw std::runtime_error("depth_image must have rank 3 or 4");
  const size_t offset = depth_shape_.size() - 3;
  if (offset && depth_shape_[0] != 1 && depth_shape_[0] != -1)
    throw std::runtime_error("depth_image batch must be one");
  if (offset) depth_shape_[0] = 1;
  history_ = depth_shape_[offset];
  if (history_ <= 0 || depth_shape_[offset + 1] != spec_.height || depth_shape_[offset + 2] != spec_.width)
    throw std::runtime_error("depth_image concrete dimensions do not match spec");
  // Same leading-batch rule as obs: a concrete 1 or a dynamic -1, since Run()
  // always feeds a [1, 1] tensor and the python kernel loads such a model.
  if (shapes.count("time_step") && (shapes.at("time_step").size() != 2 || shapes.at("time_step")[1] != 1 ||
                                    (shapes.at("time_step")[0] != 1 && shapes.at("time_step")[0] != -1)))
    throw std::runtime_error("time_step must be [1, 1] or [-1, 1]");
  bool actions_found = false;
  for (size_t i = 0; i < model_->outputs().size(); ++i) {
    const auto &name = model_->outputs()[i].name;
    if (name == "actions") {
      actions_found = true;
    }
    if (name == "climb_pred") climb_output_ = i;
  }
  // climb_pred is required when something reads it: the caller (require_climb_head), forced climbing, or a
  // climb rule (dz_and_pred, or an exit other than both_off). dz_or_pred and dz_only with both_off climb on
  // the external dz alone, so a model without the head climbs while dz is on.
  const bool rules_read_pred = climb_enter_ == ClimbEnter::kDzAndPred || climb_exit_ != ClimbExit::kBothOff;
  if (!actions_found ||
      ((options_.require_climb_head || options_.force_manual_climbing || rules_read_pred) && !climb_output_))
    throw std::runtime_error("missing actions or required climb_pred output");
  if (model_->find_output("actions")->element_count() != static_cast<int64_t>(n))
    throw std::runtime_error("actions must match joint count");
  if (climb_output_ && model_->find_output("climb_pred")->element_count() != 1)
    throw std::runtime_error("climb_pred must contain one float");
  std::vector<std::string> names = spec_.joints;
  if (metadata.contains("joint_names") && !metadata["joint_names"].is_null()) {
    auto value = metadata["joint_names"];
    if (value.is_string()) {
      names.clear();
      std::stringstream stream(value.get<std::string>());
      std::string item;
      while (std::getline(stream, item, ','))
        if (!item.empty()) names.push_back(item);
    } else
      names = value.get<std::vector<std::string>>();
  }
  if (names.size() != n || std::set<std::string>(names.begin(), names.end()) !=
                               std::set<std::string>(spec_.joints.begin(), spec_.joints.end()))
    throw std::runtime_error("model joint_names must contain runtime joints exactly");
  model_to_real_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    int real = std::find(spec_.joints.begin(), spec_.joints.end(), names[i]) - spec_.joints.begin();
    real_to_model_.push_back(real);
    model_to_real_[real] = i;
  }
  auto gains = [&](const char *name, const std::optional<std::vector<double>> &configured) {
    if (!configured && (!metadata.contains(name) || metadata[name].is_null()))
      throw std::runtime_error(std::string(name) + ": model provides no metadata and no gains were configured");
    auto values = configured ? *configured : detail::Numbers(metadata.at(name));
    if (values.size() != n) throw std::runtime_error(std::string(name) + " must match joint count");
    Finite(values, name);
    if (!configured) {
      auto model = values;
      for (size_t i = 0; i < n; ++i) values[i] = model[model_to_real_[i]];
    }
    return values;
  };
  kp_ = gains("kp", spec_.kp);
  kd_ = gains("kd", spec_.kd);
  scale_ = metadata.contains("action_scale") ? detail::Numbers(metadata["action_scale"])
                                             : std::vector<double>{spec_.action_scale};
  if (scale_.size() != 1 && scale_.size() != n)
    throw std::runtime_error("action_scale must have one or joint-count values");
  Finite(scale_, "action scale");
  if (scale_.size() == 1) scale_.resize(n, scale_[0]);
  command_vector_.resize(spec_.command_dim);
  action_.resize(n);
  scaled_.resize(n);
  Reset();
}

void PrimitiveFamily::Reset() {
  frames_.clear();
  obs_.clear();
  depth_.clear();
  std::fill(action_.begin(), action_.end(), 0);
  std::fill(scaled_.begin(), scaled_.end(), 0);
  CancelTransition();
  fixed_arm_ = 0;
  dz_ = false;
  climb_at_stand_allowed_ = false;
  climb_continuation_ = false;
  climb_pred_ = 0;
  pred_under_climb_ = false;
  SetCommand(0);
}
void PrimitiveFamily::SetSpeed(int speed) {
  if (speed < 0 || speed > 2 || (has_climb_head() && speed == 2))
    throw std::invalid_argument("invalid speed mode or climb-column alias");
  speed_ = speed;
}
void PrimitiveFamily::SetCommand(int index) {
  if (index < 0 || index >= spec_.command_dim) throw std::invalid_argument("invalid command index");
  command_ = index;
  ResolveClimb();
}
void PrimitiveFamily::SetFixedArm(int side) {
  if (side < 0 || side > 2) throw std::invalid_argument("invalid fixed-arm side");
  fixed_arm_ = side;
  ApplyFixedArm();
}
void PrimitiveFamily::SetDz(bool enabled) {
  // A model without a climb head was only loaded under rules that need none (see the constructor).
  dz_ = enabled;
  ResolveClimb();
}
void PrimitiveFamily::SetClimbAtStandAllowed(bool allowed, bool continuation) {
  climb_at_stand_allowed_ = allowed;
  climb_continuation_ = continuation;
  ResolveClimb();
}
bool PrimitiveFamily::StandBlocked() const {
  return command_ == 0 && (options_.force_manual_climbing || !climb_at_stand_allowed_);
}
bool PrimitiveFamily::StandHeld() const {
  return StandBlocked() && climbing_ && climb_continuation_ && !options_.force_manual_climbing;
}
void PrimitiveFamily::ResolveClimb() {
  // Only an explicitly permitted, live navigation pause may climb from stand.
  // Stops/watchdogs revoke permission; new predictions cannot undo that gate.
  // Forced-forward climbing always preserves stand.
  const bool pred = climb_pred_ >= .1;
  // A gated entry never trusts a prediction read under the flat-gait command;
  // the next tick reads it under column 14.
  const bool entry_pred = pred && (!Gated() || pred_under_climb_);
  bool enter = false;
  switch (climb_enter_) {
    case ClimbEnter::kDzOrPred:
      enter = dz_ || entry_pred;
      break;
    case ClimbEnter::kDzAndPred:
      enter = dz_ && entry_pred;
      break;
    case ClimbEnter::kDzOnly:
      enter = dz_;
      break;
  }
  bool stay = false;
  switch (climb_exit_) {
    case ClimbExit::kBothOff:
      stay = dz_ || pred;
      break;
    case ClimbExit::kEitherOff:
      stay = dz_ && pred;
      break;
    case ClimbExit::kPredOff:
      stay = pred;
      break;
  }
  climbing_ = StandHeld()
                  ? stay
                  : !StandBlocked() && (command_ == 14 ||
                                        (options_.force_manual_climbing && command_ == kCodes[speed_].at("forward")) ||
                                        (climbing_ ? stay : enter));
  std::fill(command_vector_.begin(), command_vector_.end(), 0);
  command_vector_[effective_command()] = 1;
  ApplyFixedArm();
}
bool PrimitiveFamily::ProbeClimb() const {
  const bool manual = command_ == 14 || (options_.force_manual_climbing && command_ == kCodes[speed_].at("forward"));
  if (!Gated() || !climb_output_ || manual || (StandBlocked() && !StandHeld())) return false;
  // Only when this tick's decision depends on climb_pred.
  if (climbing_) {
    switch (climb_exit_) {
      case ClimbExit::kBothOff:
        return !dz_;
      case ClimbExit::kEitherOff:
        return dz_;
      case ClimbExit::kPredOff:
        return true;
    }
  }
  switch (climb_enter_) {
    case ClimbEnter::kDzOrPred:
      return !dz_;
    case ClimbEnter::kDzAndPred:
      return dz_;
    case ClimbEnter::kDzOnly:
      return false;
  }
  return false;
}
void PrimitiveFamily::ApplyFixedArm() {
  if (command_ == 6 || command_ == 7) return;
  command_vector_[6] = command_vector_[7] = 0;
  if (fixed_arm_) command_vector_[fixed_arm_ + 5] = 1;
}
int PrimitiveFamily::CommandCode(const std::string &name) const {
  auto found = kCodes[speed_].find(name);
  if (found != kCodes[speed_].end()) return found->second;
  found = kCodes[0].find(name);
  return found == kCodes[0].end() ? 0 : found->second;
}
std::string PrimitiveFamily::VelocityName(double vx, double vy, double wz, double deadzone) const {
  if (discretizer_ == "simple" || speed_ != 0)
    return Simple(std::abs(vx) > deadzone ? vx : 0, std::abs(vy) > deadzone ? -vy : 0,
                  std::abs(wz) > deadzone ? -wz : 0);
  if (std::abs(vy) > deadzone && std::abs(vy) > std::abs(vx)) return vy > 0 ? "leftward" : "rightward";
  if (vx < -deadzone) return "back";
  if (options_.in_place_yaw_threshold && vx <= deadzone)
    return std::abs(wz) <= *options_.in_place_yaw_threshold ? "stand" : wz > 0 ? "turn_left" : "turn_right";
  const auto &nominal = options_.velocity_primitives;
  const std::vector<std::string> candidates =
      vx > deadzone ? std::vector<std::string>{"forward", "left_45", "right_45", "turn_left", "turn_right"}
                    : std::vector<std::string>{"stand", "turn_left", "turn_right"};
  auto distance = [&](const std::string &name) {
    const auto &pair = nominal.at(name);
    return std::pow((vx - pair[0]) / options_.forward_distance_scale, 2) +
           std::pow((wz * options_.yaw_request_gain - pair[1]) / options_.yaw_distance_scale, 2);
  };
  auto best = *std::min_element(candidates.begin(), candidates.end(),
                                [&](auto &a, auto &b) { return distance(a) < distance(b); });
  for (const auto &[name, code] : kCodes[0])
    if (code == command_ && nominal.count(name) && options_.switch_margin > 0 &&
        distance(best) >= distance(name) - options_.switch_margin)
      return name;
  return best;
}
int PrimitiveFamily::SetVelocity(double vx, double vy, double wz, double deadzone) {
  Finite(std::array<double, 3>{vx, vy, wz}, "velocity");
  int index = CommandCode(VelocityName(vx, vy, wz, deadzone));
  if (index != command_) SetCommand(index);
  return index;
}
void PrimitiveFamily::RequestKey(const std::string &key, double now) {
  const std::map<std::string, std::string> names{{"w", "forward"},  {"s", "back"},    {"a", "left_45"},
                                                 {"d", "right_45"}, {"q", "left_90"}, {"e", "right_90"},
                                                 {"z", "stand"}};
  // Validate before any lookup so a stray keypress is refused like every other
  // setter (std::invalid_argument) instead of escaping as std::out_of_range:
  // an unknown key, or a key whose primitive the current speed table does not
  // carry (left_90/right_90 exist only at speeds 1 and 2).
  int index;
  if (key == "j" || key == "l") {
    index = key == "j" ? 12 : 13;
  } else {
    auto name = names.find(key);
    if (name == names.end()) throw std::invalid_argument("unknown movement key");
    auto code = kCodes[speed_].find(name->second);
    if (code == kCodes[speed_].end()) throw std::invalid_argument("movement key unavailable at this speed");
    index = code->second;
  }
  const std::set<std::string> forward{"w", "a", "d"}, maneuver{"s", "q", "e", "j", "l"};
  double delay = 0;
  if ((key_ == "s") != (key == "s") || (forward.count(key_) && maneuver.count(key)) ||
      (maneuver.count(key_) && forward.count(key)))
    delay = 1;
  else if (maneuver.count(key_) && maneuver.count(key) && key != key_)
    delay = .5;
  key_ = key;
  if (delay == 0) {
    pending_.reset();
    SetCommand(index);
  } else {
    SetCommand(0);
    pending_ = index;
    release_time_ = now + delay;
  }
}
bool PrimitiveFamily::UpdateTransition(double now) {
  if (!pending_ || now < release_time_) return false;
  SetCommand(*pending_);
  pending_.reset();
  return true;
}
void PrimitiveFamily::CancelTransition() {
  key_ = "z";
  pending_.reset();
  release_time_ = 0;
}

std::vector<float> PrimitiveFamily::Observation(const State &state) const {
  const auto n = spec_.joints.size();
  detail::CheckState(state, n);
  const auto gravity = detail::ObservationGravity(state, spec_.upright, imu_to_obs_chain_);
  std::vector<float> obs;
  obs.reserve(6 + n * 3 + spec_.command_dim);
  for (auto v : gravity) obs.push_back(v);
  for (auto v : state.omega) obs.push_back(spec_.zero_omega ? 0 : v);
  for (auto real : real_to_model_) obs.push_back(state.q[real] - spec_.defaults[real]);
  for (auto real : real_to_model_) obs.push_back(state.dq[real]);
  obs.insert(obs.end(), action_.begin(), action_.end());
  obs.insert(obs.end(), command_vector_.begin(), command_vector_.end());
  return obs;
}

void PrimitiveFamily::Run() {
  auto &obs = model_->input("obs");
  auto &depth = model_->input("depth_image");
  std::copy(obs_.begin(), obs_.end(), obs.begin());
  std::copy(depth_.begin(), depth_.end(), depth.begin());
  if (model_->find_input("time_step")) model_->input("time_step")[0] = 0;
  ++calls_;
  model_->run();
}
std::vector<double> PrimitiveFamily::Target(const State &state, const std::optional<std::vector<float>> &depth,
                                            double now) {
  UpdateTransition(now);
  obs_ = Observation(state);
  // Branch rather than value_or: value_or evaluates its argument every call,
  // which zero-fills a height*width frame on the control loop even when a
  // depth frame is present.
  const auto frame_size = spec_.depth_frame_size();
  std::vector<float> frame = depth ? *depth : std::vector<float>(frame_size, 0);
  if (frame.size() != frame_size) throw std::invalid_argument("depth frame size mismatch");
  frames_.push_back(std::move(frame));
  if (frames_.size() > static_cast<size_t>(history_)) frames_.pop_front();
  depth_.clear();
  for (int i = frames_.size(); i < history_; ++i)
    depth_.insert(depth_.end(), frames_.front().begin(), frames_.front().end());
  for (const auto &f : frames_) depth_.insert(depth_.end(), f.begin(), f.end());
  Finite(obs_, "observation");
  Finite(depth_, "depth");
  const auto command_start = obs_.size() - spec_.command_dim;
  if (ProbeClimb()) {
    // Read climb_pred under column 14: a head that sees commands 0-13 must not
    // judge a stair with the flat-gait command it is about to replace.
    std::fill(command_vector_.begin(), command_vector_.end(), 0);
    command_vector_[14] = 1;
    ApplyFixedArm();
    std::copy(command_vector_.begin(), command_vector_.end(), obs_.begin() + command_start);
  }
  float old_climb = obs_[command_start + 14];
  calls_ = 0;
  Run();
  if (climb_output_) {
    const auto &climb = model_->output("climb_pred");
    if (climb.size() != 1) throw std::runtime_error("climb_pred must contain one float");
    const float prediction = climb[0];
    if (!std::isfinite(prediction)) throw std::runtime_error("climb_pred is non-finite");
    climb_pred_ = prediction;
    pred_under_climb_ = old_climb == 1;
    ResolveClimb();
  }
  if (command_vector_[14] != old_climb) {
    std::copy(command_vector_.begin(), command_vector_.end(), obs_.begin() + command_start);
    Run();
  }
  const auto &raw = model_->output("actions");
  if (raw.size() != action_.size()) throw std::runtime_error("actions must match joint count");
  std::vector<float> action(action_.size());
  for (size_t i = 0; i < action.size(); ++i) action[i] = clip_policy_action(raw[i]);
  Finite(action, "policy action");
  action_ = std::move(action);
  std::vector<double> target(action_.size());
  for (size_t real = 0; real < target.size(); ++real) {
    int model = model_to_real_[real];
    scaled_[real] = action_[model] * scale_[model];
    target[real] = spec_.defaults[real] + scaled_[real];
  }
  return target;
}
}  // namespace

namespace detail {
std::unique_ptr<Family> MakePrimitiveFamily(std::shared_ptr<OnnxModel> model, const Spec &spec,
                                            const Options &options) {
  return std::make_unique<PrimitiveFamily>(std::move(model), spec, options);
}
}  // namespace detail

Policy::Policy(std::shared_ptr<OnnxModel> model, Spec spec, Options options)
    : spec_(std::move(spec)), options_(std::move(options)), model_(std::move(model)) {
  if (!model_) throw std::invalid_argument("contextual model is required");
  const size_t n = spec_.joints.size();
  if (!n || spec_.defaults.size() != n || spec_.height <= 0 || spec_.width <= 0)
    throw std::invalid_argument("invalid contextual policy spec");
  Finite(spec_.defaults, "default positions");
  family_ = options_.discretizer ? detail::MakePrimitiveFamily(model_, spec_, options_)
                                 : detail::MakeVelocityFamily(model_, spec_, options_);
}
}  // namespace holosoma::contextual
