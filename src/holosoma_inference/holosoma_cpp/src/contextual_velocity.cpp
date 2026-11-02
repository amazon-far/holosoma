// A depth policy trained on a continuous velocity command. Its observation
// contract is declared in ONNX metadata: one flattened input of named
// proprioceptive terms followed by the depth frame, one action output, joint
// order, stance, per-joint action scale, gains and command ranges. Every key
// the contract names is validated here, before the first inference, and a
// present-but-malformed optional key is refused rather than read as absent.

#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include "contextual_family.hpp"
#include "holosoma_cpp/action_clip.hpp"

namespace holosoma::contextual::detail {
namespace {
constexpr const char *kPolicyKind = "hybrid_velocity";
// GMA fixed-arm flag: [left held, right held], never both.
constexpr std::array<std::array<float, 2>, 3> kFixedArmFlags{{{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}}};
constexpr std::array<const char *, 3> kCommandAxes{"lin_vel_x", "lin_vel_y", "ang_vel_yaw"};
// Below this every command axis reads as standing.
constexpr float kMovingThreshold = 1e-3f;

enum class Term { kActions, kBaseAngVel, kDofPos, kDofVel, kFixedArmFlag, kProjectedGravity, kVelocityCommand };
const std::map<std::string, Term> kTerms{
    {"actions", Term::kActions},
    {"base_ang_vel", Term::kBaseAngVel},
    {"dof_pos", Term::kDofPos},
    {"dof_vel", Term::kDofVel},
    {"fixed_arm_flag", Term::kFixedArmFlag},
    {"projected_gravity", Term::kProjectedGravity},
    {"velocity_command", Term::kVelocityCommand},
};

class ContractError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Contract {
 public:
  explicit Contract(const OnnxModel &model) : file_(model.path()), meta_(Metadata(model)) {}

  [[noreturn]] void Fail(const std::string &key, const std::string &why) const {
    throw ContractError(file_ + ": ONNX metadata " + key + " " + why);
  }
  [[noreturn]] void FailModel(const std::string &why) const { throw ContractError(file_ + ": " + why); }
  bool Has(const std::string &key) const { return meta_.contains(key); }
  const Json &Required(const std::string &key) const {
    if (!meta_.contains(key) || meta_.at(key).is_null()) Fail(key, "is missing");
    return meta_.at(key);
  }
  // Present keys must be well formed; null counts as present and malformed.
  const Json *Optional(const std::string &key) const {
    if (!meta_.contains(key)) return nullptr;
    if (meta_.at(key).is_null()) Fail(key, "is null; omit it or give a value");
    return &meta_.at(key);
  }
  std::vector<double> FiniteNumbers(const std::string &key, const Json &value) const {
    std::vector<double> values;
    try {
      values = Numbers(value);
    } catch (const Json::exception &) {
      Fail(key, "must be a number or a list of numbers");
    }
    for (double v : values)
      if (!std::isfinite(v) || std::abs(v) > std::numeric_limits<float>::max())
        Fail(key, "must contain finite float32 values");
    return values;
  }
  long long Integer(const std::string &key, const Json &value) const {
    if (value.is_number_integer()) return value.get<long long>();
    if (value.is_number_float()) {
      const double v = value.get<double>();
      if (std::isfinite(v) && v == std::floor(v) && std::abs(v) < 1e15) return static_cast<long long>(v);
    }
    Fail(key, "must be an integer");
  }
  double Number(const std::string &key, const Json &value) const {
    const auto values = FiniteNumbers(key, value);
    if (values.size() != 1 || value.is_array()) Fail(key, "must be one number");
    return values[0];
  }
  const Json &Object(const std::string &key, const Json &value) const {
    if (!value.is_object()) Fail(key, "must be a JSON object");
    return value;
  }

 private:
  std::string file_;
  Json meta_;
};

class VelocityFamily final : public Family {
 public:
  VelocityFamily(std::shared_ptr<OnnxModel> model, const Spec &spec, const Options &options);

  // Clears the action feedback and the command. The fixed-arm side is a held
  // operator mode and survives a reset, as in the reviewed kernel.
  void Reset() override {
    std::fill(last_action_.begin(), last_action_.end(), 0.f);
    command_ = {};
  }
  void SetSpeed(int) override { Refuse("speed modes"); }
  void SetCommand(int) override { Refuse("primitive command indices"); }
  void SetFixedArm(int side) override {
    if (side < 0 || side > 2) throw std::invalid_argument("invalid fixed-arm side");
    fixed_arm_ = side;
    fixed_arm_flag_ = kFixedArmFlags[static_cast<size_t>(side)];
  }
  void SetDz(bool enabled) override {
    if (enabled) throw std::invalid_argument("dz requires climb_pred; a continuous velocity policy has no climb head");
  }
  // There is no climb head to permit or continue, so the gate has nothing to hold.
  void SetClimbAtStandAllowed(bool, bool) override {}
  // Clamped to the trained ranges; no deadzone, since a small command is in
  // distribution for a continuous policy.
  int SetVelocity(double vx, double vy, double wz, double) override {
    const std::array<double, 3> raw{vx, vy, wz};
    Finite(raw, "velocity");
    for (size_t i = 0; i < 3; ++i) command_[i] = static_cast<float>(std::clamp(raw[i], ranges_[i][0], ranges_[i][1]));
    return kNoPrimitive;
  }
  void RequestKey(const std::string &, double) override { Refuse("movement keys"); }
  bool UpdateTransition(double) override { return false; }
  void CancelTransition() override {}
  std::vector<float> Observation(const State &state) const override;
  std::vector<double> Target(const State &state, const std::optional<std::vector<float>> &depth, double) override;

  const std::vector<double> &default_dof_pos() const override { return default_real_; }
  const std::vector<double> &kp() const override { return kp_; }
  const std::vector<double> &kd() const override { return kd_; }
  bool gains_from_model() const override { return true; }
  const std::vector<int> &real_to_model() const override { return real_to_model_; }
  const std::vector<int> &model_to_real() const override { return model_to_real_; }
  const std::vector<float> &observation() const override { return obs_; }
  const std::vector<float> &depth() const override { return depth_; }
  const std::vector<int64_t> &depth_shape() const override { return depth_shape_; }
  const std::vector<float> &action() const override { return action_; }
  const std::vector<float> &scaled() const override { return scaled_; }
  bool has_climb_head() const override { return false; }
  bool dz() const override { return false; }
  bool climbing() const override { return false; }
  float climb_pred() const override { return 0; }
  int active_command() const override { return kNoPrimitive; }
  int effective_command() const override { return kNoPrimitive; }
  int fixed_arm() const override { return fixed_arm_; }
  int speed() const override { return 0; }
  int calls() const override { return calls_; }
  const std::string &key() const override { return no_key_; }
  std::optional<int> pending() const override { return std::nullopt; }
  double release_time() const override { return 0; }

 private:
  struct Slot {
    Term term;
    std::string name;
    size_t dim;
    float scale;
  };
  [[noreturn]] static void Refuse(const char *what) {
    throw std::invalid_argument(std::string("a continuous velocity policy takes velocity commands, not ") + what);
  }
  size_t Width(Term term) const;
  void ReadCommandContract(const Contract &c, size_t command_start, float command_scale);
  void ReadDepthContract(const Contract &c);

  Spec spec_;
  std::shared_ptr<OnnxModel> model_;
  Chain chain_;
  std::string input_name_, output_name_;
  size_t proprio_dim_ = 0, input_dim_ = 0;
  std::vector<Slot> slots_;
  std::vector<int> real_to_model_, model_to_real_;
  std::vector<double> default_model_, default_real_, scale_model_, kp_, kd_;
  std::array<std::array<double, 2>, 3> ranges_{};
  std::array<float, 3> command_{};
  std::array<float, 2> fixed_arm_flag_{};
  std::vector<float> last_action_, obs_, depth_, action_, scaled_;
  std::vector<int64_t> depth_shape_;
  int fixed_arm_ = 0, calls_ = 0;
  std::string no_key_;
};

size_t VelocityFamily::Width(Term term) const {
  switch (term) {
    case Term::kActions:
    case Term::kDofPos:
    case Term::kDofVel:
      return spec_.joints.size();
    case Term::kFixedArmFlag:
      return 2;
    case Term::kBaseAngVel:
    case Term::kProjectedGravity:
    case Term::kVelocityCommand:
      return 3;
  }
  return 0;
}

VelocityFamily::VelocityFamily(std::shared_ptr<OnnxModel> model, const Spec &spec, const Options &options)
    : spec_(spec), model_(std::move(model)) {
  const size_t n = spec_.joints.size();
  // Options that only mean something for a primitive command or a climb head.
  if (options.require_climb_head)
    throw std::invalid_argument(
        "require_climb_head: a continuous velocity policy has no climb head; set require_climb_head false");
  if (options.force_manual_climbing || options.climb_enter != "dz_or_pred" || options.climb_exit != "both_off")
    throw std::invalid_argument("climb options do not apply to a continuous velocity policy");
  if (!options.velocity_primitives.empty())
    throw std::invalid_argument("velocity_primitives do not apply to a continuous velocity policy");
  if (spec_.kp || spec_.kd)
    throw std::invalid_argument(
        "a continuous velocity policy runs on the gains its model was trained with; remove the configured gains");
  // The contract was reviewed with IMU-frame gravity; an explicit chain overrides it.
  chain_ = ResolveChain(options.imu_to_obs_joint_chain.value_or(decltype(options.imu_to_obs_joint_chain)::value_type{}),
                        spec_.joints);

  const Contract c(*model_);
  try {
    const auto &kind = c.Required("policy_kind");
    if (!kind.is_string() || kind.get<std::string>() != kPolicyKind)
      c.Fail("policy_kind", "must be \"" + std::string(kPolicyKind) +
                                "\" for a continuous velocity command (discretizer: null); a model trained on "
                                "primitive commands needs a discretizer");

    // Graph signature: one [1, width] input, one action output.
    if (model_->inputs().size() != 1 || model_->outputs().size() != 1)
      c.FailModel("a continuous velocity policy has exactly one input and one output");
    const auto &input = model_->inputs()[0];
    const auto &output = model_->outputs()[0];
    input_name_ = input.name;
    output_name_ = output.name;
    auto batch_ok = [](int64_t d) { return d == 1 || d == -1; };
    if (input.shape.size() != 2 || !batch_ok(input.shape[0]) || input.shape[1] <= 0)
      c.FailModel("input " + input_name_ + " must be [1, width] with a concrete width");
    if (output.shape.empty() || output.shape.size() > 2 || (output.shape.size() == 2 && !batch_ok(output.shape[0])) ||
        output.shape.back() != static_cast<int64_t>(n))
      c.FailModel("output " + output_name_ + " must be [1, " + std::to_string(n) + "], one action per joint");

    // Joint order, stance, action scale and gains, all in model order.
    std::vector<std::string> names;
    try {
      names = JointNames(c.Required("dof_names"));
    } catch (const Json::exception &) {
      c.Fail("dof_names", "must be a list of joint names");
    }
    try {
      JointPermutation(names, spec_.joints, real_to_model_, model_to_real_);
    } catch (const std::runtime_error &e) {
      c.Fail("dof_names", e.what());
    }
    auto per_joint = [&](const char *key, bool broadcast) {
      auto values = c.FiniteNumbers(key, c.Required(key));
      if (broadcast && values.size() == 1) values.assign(n, values[0]);
      if (values.size() != n)
        c.Fail(key, "has " + std::to_string(values.size()) + " values for " + std::to_string(n) + " joints");
      return values;
    };
    auto to_real = [&](const std::vector<double> &model_order) {
      std::vector<double> real(n);
      for (size_t m = 0; m < n; ++m) real[real_to_model_[m]] = model_order[m];
      return real;
    };
    default_model_ = per_joint("default_dof_angles", false);
    default_real_ = to_real(default_model_);
    scale_model_ = per_joint("action_scale", true);
    for (const char *key : {"kp", "kd"}) {
      auto gains = per_joint(key, false);
      if (std::any_of(gains.begin(), gains.end(), [](double g) { return g < 0; })) c.Fail(key, "must be nonnegative");
      (std::string(key) == "kp" ? kp_ : kd_) = to_real(gains);
    }

    // Command ranges: every axis the model was trained on, lo <= hi, finite.
    {
      const auto &ranges = c.Object("command_ranges", c.Required("command_ranges"));
      for (const auto &[axis, value] : ranges.items())
        if (std::find_if(kCommandAxes.begin(), kCommandAxes.end(), [&](const char *a) { return axis == a; }) ==
            kCommandAxes.end())
          c.Fail("command_ranges", "names unknown axis " + axis);
      for (size_t i = 0; i < 3; ++i) {
        const std::string key = std::string("command_ranges.") + kCommandAxes[i];
        if (!ranges.contains(kCommandAxes[i])) c.Fail(key, "is missing");
        const auto bounds = c.FiniteNumbers(key, ranges.at(kCommandAxes[i]));
        if (bounds.size() != 2 || !(bounds[0] <= bounds[1])) c.Fail(key, "must be [low, high] with low <= high");
        ranges_[i] = {bounds[0], bounds[1]};
      }
    }

    // Observation terms: known, unique, each declared width equal to what this
    // family assembles for it.
    const auto &terms = c.Required("actor_observation_terms");
    const auto &dims = c.Object("actor_observation_term_dims", c.Required("actor_observation_term_dims"));
    const Json no_scales = Json::object();
    const auto *scales = c.Optional("actor_observation_scales");
    if (scales) c.Object("actor_observation_scales", *scales);
    if (!terms.is_array() || terms.empty()) c.Fail("actor_observation_terms", "must be a nonempty list of term names");
    std::set<std::string> seen;
    size_t command_start = 0;
    float command_scale = 1;
    for (const auto &item : terms) {
      if (!item.is_string()) c.Fail("actor_observation_terms", "must contain only term names");
      const auto name = item.get<std::string>();
      const auto found = kTerms.find(name);
      if (found == kTerms.end()) c.Fail("actor_observation_terms", "names unsupported term " + name);
      if (!seen.insert(name).second) c.Fail("actor_observation_terms", "repeats term " + name);
      const std::string dim_key = "actor_observation_term_dims." + name;
      if (!dims.contains(name)) c.Fail(dim_key, "is missing");
      const auto dim = c.Integer(dim_key, dims.at(name));
      if (dim != static_cast<long long>(Width(found->second)))
        c.Fail(dim_key, "is " + std::to_string(dim) + " but the term has " + std::to_string(Width(found->second)) +
                            " values for this robot");
      float scale = 1;
      if (scales && scales->contains(name)) {
        const double s = c.Number("actor_observation_scales." + name, scales->at(name));
        scale = static_cast<float>(s);
      }
      if (found->second == Term::kVelocityCommand) {
        command_start = proprio_dim_;
        command_scale = scale;
      }
      slots_.push_back({found->second, name, static_cast<size_t>(dim), scale});
      proprio_dim_ += static_cast<size_t>(dim);
    }
    for (const auto &[name, value] : dims.items())
      if (!seen.count(name)) c.Fail("actor_observation_term_dims", "declares " + name + ", which is not a term");
    if (scales)
      for (const auto &[name, value] : scales->items())
        if (!seen.count(name)) c.Fail("actor_observation_scales", "scales " + name + ", which is not a term");
    if (!seen.count("velocity_command")) c.Fail("actor_observation_terms", "must include velocity_command");

    if (const auto *history = c.Optional("actor_observation_history_length"))
      if (c.Integer("actor_observation_history_length", *history) != 1)
        c.Fail("actor_observation_history_length", "must be 1; observation history is not supported");
    if (const auto *dim = c.Optional("actor_observation_dim"))
      if (c.Integer("actor_observation_dim", *dim) != static_cast<long long>(proprio_dim_))
        c.Fail("actor_observation_dim",
               "does not equal the sum of actor_observation_term_dims (" + std::to_string(proprio_dim_) + ")");

    ReadDepthContract(c);
    input_dim_ = proprio_dim_ + spec_.depth_frame_size();
    if (const auto *dim = c.Optional("actor_policy_input_dim"))
      if (c.Integer("actor_policy_input_dim", *dim) != static_cast<long long>(input_dim_))
        c.Fail("actor_policy_input_dim",
               "does not equal observation terms + depth (" + std::to_string(input_dim_) + ")");
    if (input.shape[1] != static_cast<int64_t>(input_dim_))
      c.FailModel("input " + input_name_ + " has " + std::to_string(input.shape[1]) +
                  " values but observation terms + depth are " + std::to_string(input_dim_));
    ReadCommandContract(c, command_start, command_scale);
  } catch (const ContractError &) {
    throw;
  } catch (const Json::exception &e) {
    c.FailModel(std::string("ONNX metadata is malformed (") + e.what() + ")");
  }

  last_action_.assign(n, 0.f);
  action_.assign(n, 0.f);
  scaled_.assign(n, 0.f);
  depth_shape_ = {spec_.height, spec_.width};
  Reset();
}

void VelocityFamily::ReadDepthContract(const Contract &c) {
  if (const auto *hw = c.Optional("depth_policy_shape_hw")) {
    const auto values = c.FiniteNumbers("depth_policy_shape_hw", *hw);
    if (values.size() != 2 || values[0] != spec_.height || values[1] != spec_.width)
      c.Fail("depth_policy_shape_hw", "does not match the configured depth size " + std::to_string(spec_.height) + "x" +
                                          std::to_string(spec_.width));
  }
  if (const auto *dim = c.Optional("depth_observation_dim"))
    if (c.Integer("depth_observation_dim", *dim) != static_cast<long long>(spec_.depth_frame_size()))
      c.Fail("depth_observation_dim", "does not equal depth height x width");
  // The flattened input is the proprioceptive terms, then the depth frame.
  if (const auto *groups = c.Optional("actor_policy_input_groups"))
    if (*groups != Json::array({"actor_obs", "depth_camera"}))
      c.Fail("actor_policy_input_groups", "must be [\"actor_obs\", \"depth_camera\"], the order this family feeds");
}

void VelocityFamily::ReadCommandContract(const Contract &c, size_t command_start, float command_scale) {
  // Commands are fed as given: the caller supplies a heading-frame twist.
  if (const auto *frame = c.Optional("velocity_command_frame"))
    if (!frame->is_string() || frame->get<std::string>() != "heading")
      c.Fail("velocity_command_frame", "must be \"heading\"; commands are fed in the heading frame as given");
  const auto *projection = c.Optional("velocity_command_projection");
  if (!projection) return;
  const std::string key = "velocity_command_projection";
  c.Object(key, *projection);
  auto field = [&](const char *name) -> const Json & {
    if (!projection->contains(name)) c.Fail(key + "." + name, "is missing");
    return projection->at(name);
  };
  // The graph projects the command itself; this family adds no projection.
  if (field("embedded_in_graph") != true)
    c.Fail(key + ".embedded_in_graph", "must be true; this family feeds the raw command and projects nothing");
  if (c.Integer(key + ".observation_start", field("observation_start")) != static_cast<long long>(command_start) ||
      c.Integer(key + ".observation_end", field("observation_end")) != static_cast<long long>(command_start + 3))
    c.Fail(key, "observation span does not match velocity_command at [" + std::to_string(command_start) + ", " +
                    std::to_string(command_start + 3) + ")");
  if (projection->contains("history_length") && c.Integer(key + ".history_length", field("history_length")) != 1)
    c.Fail(key + ".history_length", "must be 1");
  if (projection->contains("observation_scale") &&
      static_cast<float>(c.Number(key + ".observation_scale", field("observation_scale"))) != command_scale)
    c.Fail(key + ".observation_scale", "does not match actor_observation_scales.velocity_command");
  for (const char *deadband : {"linear_deadband", "yaw_deadband", "yaw_rate_min_abs"})
    if (projection->contains(deadband) && c.Number(key + "." + deadband, field(deadband)) != 0)
      c.Fail(key + "." + deadband, "must be 0; this family applies no deadband");
  if (projection->contains("yaw_rate_range")) {
    const auto range = c.FiniteNumbers(key + ".yaw_rate_range", field("yaw_rate_range"));
    if (range.size() != 2 || range[0] != ranges_[2][0] || range[1] != ranges_[2][1])
      c.Fail(key + ".yaw_rate_range", "does not match command_ranges.ang_vel_yaw");
  }
}

std::vector<float> VelocityFamily::Observation(const State &state) const {
  const size_t n = spec_.joints.size();
  CheckState(state, n);
  std::vector<float> out;
  out.reserve(proprio_dim_);
  auto put = [&](double v, float scale) {
    const auto f = static_cast<float>(v);
    out.push_back(scale != 1.f ? f * scale : f);
  };
  for (const auto &slot : slots_) {
    switch (slot.term) {
      case Term::kActions:
        for (float v : last_action_) put(v, slot.scale);
        break;
      case Term::kBaseAngVel:
        for (double v : state.omega) put(spec_.zero_omega ? 0.0 : v, slot.scale);
        break;
      case Term::kDofPos:
        for (size_t m = 0; m < n; ++m) put(state.q[real_to_model_[m]] - default_model_[m], slot.scale);
        break;
      case Term::kDofVel:
        for (size_t m = 0; m < n; ++m) put(state.dq[real_to_model_[m]], slot.scale);
        break;
      case Term::kFixedArmFlag:
        for (float v : fixed_arm_flag_) put(v, slot.scale);
        break;
      case Term::kProjectedGravity:
        for (double v : ObservationGravity(state, spec_.upright, chain_)) put(v, slot.scale);
        break;
      case Term::kVelocityCommand:
        for (float v : command_) put(v, slot.scale);
        break;
    }
  }
  return out;
}

std::vector<double> VelocityFamily::Target(const State &state, const std::optional<std::vector<float>> &depth, double) {
  const size_t n = spec_.joints.size();
  const auto frame_size = spec_.depth_frame_size();
  auto obs = Observation(state);
  Finite(obs, "observation");
  if (depth) {
    if (depth->size() != frame_size) throw std::invalid_argument("depth frame size mismatch");
    Finite(*depth, "depth");
    depth_ = *depth;
  } else {
    depth_.assign(frame_size, 0.f);
  }
  obs_ = std::move(obs);
  auto &input = model_->input(input_name_);
  std::copy(obs_.begin(), obs_.end(), input.begin());
  std::copy(depth_.begin(), depth_.end(), input.begin() + static_cast<std::ptrdiff_t>(obs_.size()));
  model_->run();
  ++calls_;
  const auto &raw = model_->output(output_name_);
  if (raw.size() != n) throw std::runtime_error("actions must match joint count");
  std::vector<float> action(n);
  for (size_t m = 0; m < n; ++m) action[m] = clip_policy_action(raw[m]);
  action_ = std::move(action);
  std::vector<double> target(n);
  for (size_t m = 0; m < n; ++m) {
    const size_t real = static_cast<size_t>(real_to_model_[m]);
    scaled_[real] = static_cast<float>(action_[m] * scale_model_[m]);
    target[real] = default_model_[m] + action_[m] * scale_model_[m];
  }
  // The clipped action, in model order, is the next tick's `actions` term.
  last_action_ = action_;
  return target;
}
}  // namespace

std::unique_ptr<Family> MakeVelocityFamily(std::shared_ptr<OnnxModel> model, const Spec &spec, const Options &options) {
  return std::make_unique<VelocityFamily>(std::move(model), spec, options);
}
}  // namespace holosoma::contextual::detail
