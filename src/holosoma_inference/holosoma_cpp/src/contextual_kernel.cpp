#include "holosoma_cpp/contextual_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include "holosoma_cpp/action_clip.hpp"

namespace holosoma::contextual {
namespace {
using Json = nlohmann::json;
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
template <class T>
void Finite(const T &values, const char *name) {
  if (!std::all_of(values.begin(), values.end(), [](auto x) { return std::isfinite(x); }))
    throw std::runtime_error(std::string(name) + " contains non-finite values");
}
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
std::vector<double> Vector(const Json &v) {
  if (!v.is_array()) return {v.get<double>()};
  std::vector<double> values;
  for (const auto &item : v) {
    auto flattened = Vector(item);
    values.insert(values.end(), flattened.begin(), flattened.end());
  }
  return values;
}
}  // namespace

Policy::Policy(std::shared_ptr<OnnxModel> model, Spec spec, Options options)
    : spec_(std::move(spec)), options_(std::move(options)), model_(std::move(model)) {
  if (!model_) throw std::invalid_argument("contextual model is required");
  const size_t n = spec_.joints.size();
  if (!n || spec_.defaults.size() != n || spec_.command_dim < 15 || spec_.height <= 0 || spec_.width <= 0)
    throw std::invalid_argument("invalid contextual policy spec");
  Finite(spec_.defaults, "default positions");
  if (options_.discretizer != "simple" && options_.discretizer != "nearest_primitive")
    throw std::invalid_argument("unknown velocity discretizer");
  if (options_.climb_enter != "dz_or_pred" && options_.climb_enter != "dz_and_pred" &&
      options_.climb_enter != "dz_only")
    throw std::invalid_argument("unknown climb_enter");
  if (options_.climb_exit != "both_off" && options_.climb_exit != "either_off" && options_.climb_exit != "pred_off")
    throw std::invalid_argument("unknown climb_exit");
  // Entering must satisfy staying, or the gait flips on every resolve.
  if (options_.climb_enter != "dz_and_pred" && options_.climb_exit != "both_off")
    throw std::invalid_argument("climb_enter " + options_.climb_enter + " needs climb_exit both_off");
  if (options_.discretizer != "simple" &&
      (!std::isfinite(options_.yaw_request_gain) || options_.yaw_request_gain <= 0 ||
       !std::isfinite(options_.switch_margin) || options_.switch_margin < 0))
    throw std::invalid_argument("invalid nearest-primitive parameters");
  if (options_.discretizer == "nearest_primitive") {
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
  for (const auto &[name, axis] : options_.imu_to_obs_joint_chain) {
    const auto found = std::find(spec_.joints.begin(), spec_.joints.end(), name);
    if (found == spec_.joints.end())
      throw std::invalid_argument("imu_to_obs_joint_chain names " + name +
                                  ", which is not a runtime joint; set the chain for this robot, or [] for IMU-frame "
                                  "gravity");
    const int component = axis == "x" ? 1 : axis == "y" ? 2 : axis == "z" ? 3 : 0;
    if (!component) throw std::invalid_argument("imu_to_obs_joint_chain axis for " + name + " must be x, y or z");
    imu_to_obs_chain_.emplace_back(static_cast<size_t>(found - spec_.joints.begin()), component);
  }
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
  if (!actions_found || ((options_.require_climb_head || options_.force_manual_climbing ||
                          options_.climb_enter != "dz_or_pred" || options_.climb_exit != "both_off") &&
                         !climb_output_))
    throw std::runtime_error("missing actions or required climb_pred output");
  if (model_->find_output("actions")->element_count() != static_cast<int64_t>(n))
    throw std::runtime_error("actions must match joint count");
  if (climb_output_ && model_->find_output("climb_pred")->element_count() != 1)
    throw std::runtime_error("climb_pred must contain one float");
  Json metadata = Json::object();
  for (const auto &[key, value] : model_->metadata()) {
    auto parsed = Json::parse(value, nullptr, false);
    metadata[key] = parsed.is_discarded() ? Json(value) : parsed;
  }
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
    auto values = configured ? *configured : Vector(metadata.at(name));
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
  scale_ =
      metadata.contains("action_scale") ? Vector(metadata["action_scale"]) : std::vector<double>{spec_.action_scale};
  if (scale_.size() != 1 && scale_.size() != n)
    throw std::runtime_error("action_scale must have one or joint-count values");
  Finite(scale_, "action scale");
  if (scale_.size() == 1) scale_.resize(n, scale_[0]);
  command_vector_.resize(spec_.command_dim);
  action_.resize(n);
  scaled_.resize(n);
  Reset();
}

void Policy::Reset() {
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
void Policy::SetSpeed(int speed) {
  if (speed < 0 || speed > 2 || (has_climb_head() && speed == 2))
    throw std::invalid_argument("invalid speed mode or climb-column alias");
  speed_ = speed;
}
void Policy::SetCommand(int index) {
  if (index < 0 || index >= spec_.command_dim) throw std::invalid_argument("invalid command index");
  command_ = index;
  ResolveClimb();
}
void Policy::SetFixedArm(int side) {
  if (side < 0 || side > 2) throw std::invalid_argument("invalid fixed-arm side");
  fixed_arm_ = side;
  ApplyFixedArm();
}
void Policy::SetDz(bool enabled) {
  if (enabled && !has_climb_head()) throw std::invalid_argument("dz requires climb_pred");
  dz_ = enabled;
  ResolveClimb();
}
void Policy::SetClimbAtStandAllowed(bool allowed, bool continuation) {
  climb_at_stand_allowed_ = allowed;
  climb_continuation_ = continuation;
  ResolveClimb();
}
bool Policy::StandBlocked() const {
  return command_ == 0 && (options_.force_manual_climbing || !climb_at_stand_allowed_);
}
bool Policy::StandHeld() const {
  return StandBlocked() && climbing_ && climb_continuation_ && !options_.force_manual_climbing;
}
void Policy::ResolveClimb() {
  // Only an explicitly permitted, live navigation pause may climb from stand.
  // Stops/watchdogs revoke permission; new predictions cannot undo that gate.
  // Forced-forward climbing always preserves stand.
  const bool gated = options_.climb_enter != "dz_or_pred" || options_.climb_exit != "both_off";
  const bool pred = climb_pred_ >= .1;
  // A gated entry never trusts a prediction read under the flat-gait command;
  // the next tick reads it under column 14.
  const bool entry_pred = pred && (!gated || pred_under_climb_);
  const bool enter = options_.climb_enter == "dz_only"       ? dz_
                     : options_.climb_enter == "dz_and_pred" ? dz_ && entry_pred
                                                             : dz_ || entry_pred;
  const bool stay = options_.climb_exit == "pred_off"     ? pred
                    : options_.climb_exit == "either_off" ? dz_ && pred
                                                          : dz_ || pred;
  climbing_ = StandHeld()
                  ? stay
                  : !StandBlocked() && (command_ == 14 ||
                                        (options_.force_manual_climbing && command_ == kCodes[speed_].at("forward")) ||
                                        (climbing_ ? stay : enter));
  std::fill(command_vector_.begin(), command_vector_.end(), 0);
  command_vector_[effective_command()] = 1;
  ApplyFixedArm();
}
bool Policy::ProbeClimb() const {
  const bool gated = options_.climb_enter != "dz_or_pred" || options_.climb_exit != "both_off";
  const bool manual = command_ == 14 || (options_.force_manual_climbing && command_ == kCodes[speed_].at("forward"));
  if (!gated || !climb_output_ || manual || (StandBlocked() && !StandHeld())) return false;
  // Only when this tick's decision depends on climb_pred.
  if (climbing_) return options_.climb_exit == "pred_off" || (options_.climb_exit == "both_off" ? !dz_ : dz_);
  return options_.climb_enter == "dz_or_pred" ? !dz_ : options_.climb_enter == "dz_and_pred" ? dz_ : false;
}
void Policy::ApplyFixedArm() {
  if (command_ == 6 || command_ == 7) return;
  command_vector_[6] = command_vector_[7] = 0;
  if (fixed_arm_) command_vector_[fixed_arm_ + 5] = 1;
}
int Policy::CommandCode(const std::string &name) const {
  auto found = kCodes[speed_].find(name);
  if (found != kCodes[speed_].end()) return found->second;
  found = kCodes[0].find(name);
  return found == kCodes[0].end() ? 0 : found->second;
}
std::string Policy::VelocityName(double vx, double vy, double wz, double deadzone) const {
  if (options_.discretizer == "simple" || speed_ != 0)
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
int Policy::SetVelocity(double vx, double vy, double wz, double deadzone) {
  Finite(std::array<double, 3>{vx, vy, wz}, "velocity");
  int index = CommandCode(VelocityName(vx, vy, wz, deadzone));
  if (index != command_) SetCommand(index);
  return index;
}
void Policy::RequestKey(const std::string &key, double now) {
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
bool Policy::UpdateTransition(double now) {
  if (!pending_ || now < release_time_) return false;
  SetCommand(*pending_);
  pending_.reset();
  return true;
}
void Policy::CancelTransition() {
  key_ = "z";
  pending_.reset();
  release_time_ = 0;
}

std::string Policy::FinishProfile() { return model_->finish_profile(); }

std::vector<float> Policy::Observation(const State &state) const {
  const auto n = spec_.joints.size();
  if (state.q.size() != n || state.dq.size() != n) throw std::invalid_argument("state joint count mismatch");
  Finite(state.q, "q");
  Finite(state.dq, "dq");
  Finite(state.omega, "omega");
  Finite(state.quat, "orientation");
  if (state.gravity) Finite(*state.gravity, "gravity");
  // An IMU that reports no orientation is a fault, never "upright".
  const double norm = std::sqrt(std::inner_product(state.quat.begin(), state.quat.end(), state.quat.begin(), 0.0));
  if (!(norm >= 1e-8)) throw std::invalid_argument("orientation must be nonzero");
  std::array<double, 3> gravity{0, 0, -1};
  if (!spec_.upright) {
    // Gravity in the IMU body frame, from the caller or from the orientation.
    if (state.gravity)
      gravity = *state.gravity;
    else {
      auto quat = state.quat;
      for (auto &v : quat) v /= norm;
      gravity = RotateInverse(quat, {0, 0, -1});
    }
    // Then from the IMU frame into the observation frame through imu_to_obs_joint_chain.
    std::array<double, 4> chain{1, 0, 0, 0};
    for (const auto &[joint, component] : imu_to_obs_chain_) {
      const double half = state.q[joint] * .5;
      std::array<double, 4> rotation{std::cos(half), 0, 0, 0};
      rotation[component] = std::sin(half);
      chain = Multiply(chain, rotation);
    }
    gravity = RotateInverse(chain, gravity);
  }
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

void Policy::Run() {
  auto &obs = model_->input("obs");
  auto &depth = model_->input("depth_image");
  std::copy(obs_.begin(), obs_.end(), obs.begin());
  std::copy(depth_.begin(), depth_.end(), depth.begin());
  if (model_->find_input("time_step")) model_->input("time_step")[0] = 0;
  ++calls_;
  model_->run();
}
std::vector<double> Policy::Target(const State &state, const std::optional<std::vector<float>> &depth, double now) {
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
}  // namespace holosoma::contextual
