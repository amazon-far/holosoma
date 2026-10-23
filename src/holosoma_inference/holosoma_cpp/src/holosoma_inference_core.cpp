#include "holosoma_cpp/holosoma_inference_core.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "holosoma_cpp/urdf_kinematics.hpp"

namespace holosoma::live {

namespace {

using json = nlohmann::json;

// Tolerance in units of one control step, applied before flooring elapsed time
// into a motion timestep. See NextTimeStep for why it is needed.
constexpr double kTimeStepEpsilon = 1e-6;

bool IsFinite(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

bool IsFinite(const std::vector<float> &values) { return IsFinite(values.data(), values.size()); }

math::Quatf SafeQuat(const std::array<float, 4> &wxyz) {
  const auto sanitized = SanitizeQuatWxyz(wxyz.data());
  return math::quat_from_wxyz(sanitized.data()).normalized();
}

bool ParseTermOrdinal(const std::string &name, int &ordinal) {
  if (name.size() < 6 || name.compare(0, 3, "obs") != 0) return false;
  const std::size_t end = name.find('_', 3);
  if (end == std::string::npos || end == 3) return false;
  const std::string token = name.substr(3, end - 3);
  try {
    if (token.size() == 1 && ((token[0] >= 'A' && token[0] <= 'F') || (token[0] >= 'a' && token[0] <= 'f'))) {
      ordinal = std::stoi(token, nullptr, 16);
    } else {
      if (!std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); })) return false;
      ordinal = std::stoi(token);
    }
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

bool ResolveLookaheadOffsets(const std::string &func, std::vector<int> &offsets) {
  if (func.find("motion_command_lookahead_sonic_aligned") != std::string::npos) {
    offsets = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
  } else if (func.find("motion_command_lookahead_sonic") != std::string::npos) {
    offsets = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
  } else if (func.find("motion_command_lookahead_dense") != std::string::npos) {
    offsets = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  } else if (func.find("motion_command_lookahead_short") != std::string::npos) {
    offsets = {5, 10, 15, 20, 25};
  } else if (func.find("motion_command_lookahead_long") != std::string::npos) {
    offsets.clear();
    for (int i = 5; i <= 100; i += 5) offsets.push_back(i);
  } else {
    return false;
  }
  return true;
}

// Exporter function leaves define the flat actor_targets contract; numbered
// display names only define ordering. Width alone cannot identify these terms.
std::string FuncLeaf(const std::string &func) { return func.substr(func.find_last_of(":.") + 1); }

bool FlatFuture(const std::string &leaf, std::vector<int> &offsets, int &kind) {
  if (leaf == "motion_joint_pos_future_0p2s" || leaf == "motion_joint_vel_future_0p2s" ||
      leaf == "motion_future_ref_ori_b_0p2s") {
    offsets = {0, 2, 4, 6, 8};
  } else if (leaf == "motion_joint_pos_future_sonic_release" || leaf == "motion_joint_vel_future_sonic_release" ||
             leaf == "motion_future_ref_ori_b_sonic_aligned") {
    offsets = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
  } else {
    return false;
  }
  kind = leaf.find("joint_pos") != std::string::npos ? 0 : leaf.find("joint_vel") != std::string::npos ? 1 : 2;
  return true;
}

int FlatZeroWidth(const std::string &leaf) {
  static const std::map<std::string, int> widths{
      {"sonic_teleop_lower_body_future", 240},        {"sonic_teleop_tracker_positions", 9},
      {"sonic_teleop_tracker_quaternions_wxyz", 12},  {"sonic_teleop_anchor_rotation", 6},
      {"sonic_smpl_joints_local_future", 720},        {"sonic_smpl_root_rotation_future", 60},
      {"sonic_smpl_wrist_joint_positions_future", 60}};
  const auto it = widths.find(leaf);
  return it == widths.end() ? 0 : it->second;
}

template <std::size_t N>
bool JsonFloatArray(const std::unordered_map<std::string, std::string> &meta, const char *key,
                    std::array<float, N> &out, std::string &error) {
  try {
    const json values = json::parse(meta.at(key));
    if (!values.is_array() || values.size() != N) {
      error = std::string("metadata ") + key + " must contain " + std::to_string(N) + " numbers";
      return false;
    }
    for (std::size_t i = 0; i < N; ++i) out[i] = values.at(i).get<float>();
  } catch (const std::exception &e) {
    error = std::string("cannot parse metadata ") + key + ": " + e.what();
    return false;
  }
  if (!IsFinite(out.data(), out.size())) {
    error = std::string("metadata ") + key + " contains a non-finite value";
    return false;
  }
  return true;
}

}  // namespace

class WbtPolicy::FkBody {
 public:
  FkBody(const std::string &urdf, const std::string &body_name, const std::vector<std::string> &joint_names)
      : chain_(urdf, body_name, joint_names) {}

  bool Orientation(const std::array<float, 4> &root_quat_wxyz, const std::array<float, kActionDim> &dof_pos,
                   math::Quatf &out) const {
    const math::Quatf root = SafeQuat(root_quat_wxyz);
    try {
      const auto q = chain_.orientation({root.w(), root.x(), root.y(), root.z()}, {dof_pos.begin(), dof_pos.end()});
      out = math::Quatf(q[0], q[1], q[2], q[3]).normalized();
    } catch (const std::exception &) {
      return false;
    }
    return std::isfinite(out.w()) && std::isfinite(out.x()) && std::isfinite(out.y()) && std::isfinite(out.z());
  }

 private:
  holosoma::UrdfChain chain_;
};

const char *ToString(WbtHistoryInitialization mode) {
  return mode == WbtHistoryInitialization::kZeroPad ? "zero_pad" : "repeat_first_valid_obs";
}

bool ParseWbtHistoryInitialization(const std::string &text, WbtHistoryInitialization &mode) {
  if (text == "repeat_first_valid_obs") {
    mode = WbtHistoryInitialization::kRepeatFirstValidObs;
  } else if (text == "zero_pad") {
    mode = WbtHistoryInitialization::kZeroPad;
  } else {
    return false;
  }
  return true;
}

WbtPolicy::WbtPolicy() = default;

void WbtPolicy::SetHistoryInitialization(WbtHistoryInitialization mode) {
  history_initialization_ = mode;
  Reset();
}

WbtPolicy::~WbtPolicy() = default;

bool WbtPolicy::IsWbtModelFile(const std::string &model_path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path path(model_path);
  return fs::is_regular_file(path, ec) && path.extension() == ".onnx";
}

bool WbtPolicy::Initialize(const std::string &model_path, bool use_gpu, int intra_op_threads) {
  initialized_ = false;
  session_.reset();
  fk_body_.reset();
  anchor_body_name_ = "pelvis";
  motion_command_dim_ = 0;
  if (!IsWbtModelFile(model_path)) {
    std::cerr << "[WbtPolicy] single-file checkpoint does not exist or is not "
                 "an .onnx file: "
              << model_path << std::endl;
    return false;
  }

  try {
    auto session = std::make_shared<holosoma::OnnxModel>(model_path, use_gpu, intra_op_threads);
    std::cout << "[WbtPolicy] inference provider: "
              << (use_gpu ? "CUDAExecutionProvider (device 0)" : "CPUExecutionProvider") << std::endl;
    return Initialize(std::move(session));
  } catch (const std::exception &e) {
    std::cerr << "[WbtPolicy] failed to load " << model_path << ": " << e.what() << std::endl;
    return false;
  }
}

bool WbtPolicy::Initialize(std::shared_ptr<holosoma::OnnxModel> session) {
  initialized_ = false;
  session_ = std::move(session);
  fk_body_.reset();
  anchor_body_name_ = "pelvis";
  motion_command_dim_ = 0;
  if (!session_) return false;
  const auto &model_path = session_->path();
  model_path_ = model_path;

  std::string error;
  bool valid = false;
  try {
    if (!ParseMetadata(error)) {
      std::cerr << "[WbtPolicy] " << model_path << ": " << error << std::endl;
      return false;
    }
    valid = ValidateShapes(error);
  } catch (const std::exception &e) {
    std::cerr << "[WbtPolicy] " << model_path << ": unexpected exception while validating ONNX shapes: " << e.what()
              << std::endl;
    return false;
  }
  if (!valid) {
    std::cerr << "[WbtPolicy] " << model_path << ": " << error << std::endl;
    return false;
  }

  Reset();
  initialized_ = true;
  std::cout << "[WbtPolicy] Loaded holosoma WBT model " << model_path << " (obs=" << obs_input_.size()
            << ", anchor=" << anchor_body_name_ << ", lookahead=" << trained_lookahead_offsets_.size() << " slots / "
            << TrainedLookaheadFrames() << " trained frames, history=" << ToString(history_initialization_)
            << " [runtime; checkpoint "
            << (!declared_history_ ? std::string("declares none")
                : *declared_history_ == history_initialization_
                    ? std::string("agrees")
                    : std::string("declares ") + ToString(*declared_history_) + ", overridden")
            << "], action_clip=" << action_clip_ << " [" << action_clip_source_ << "])" << std::endl;
  return true;
}

bool WbtPolicy::ParseMetadata(std::string &error) {
  declared_history_.reset();
  action_clip_ = kWbtActionCeiling;
  action_clip_source_ = "ceiling; checkpoint declares none";
  bool training_action_clip_enabled = true;
  std::string stage = "reading the custom metadata map";
  try {
    std::unordered_map<std::string, std::string> metadata;
    try {
      metadata.insert(session_->metadata().begin(), session_->metadata().end());
    } catch (const std::exception &e) {
      error = std::string("cannot read ONNX metadata: ") + e.what();
      return false;
    }

    stage = "checking required metadata keys";
    const char *required[] = {"robot_urdf", "kp", "kd", "action_scale", "dof_names", "experiment_config"};
    for (const char *key : required) {
      if (metadata.find(key) == metadata.end()) {
        error = std::string("missing required ONNX metadata key \"") + key + "\"";
        return false;
      }
    }

    stage = "parsing dof_names";
    try {
      const json names = json::parse(metadata.at("dof_names"));
      if (!names.is_array() || names.size() != kActionDim) {
        error = "metadata dof_names must contain 29 unique named joints";
        return false;
      }
      joint_names_.clear();
      std::set<std::string> unique;
      for (int i = 0; i < kActionDim; ++i) {
        const auto name = names.at(i).get<std::string>();
        if (name.empty() || !unique.insert(name).second) {
          error = "metadata dof_names contains an empty or duplicate joint at index " + std::to_string(i);
          return false;
        }
        joint_names_.push_back(name);
      }
    } catch (const std::exception &e) {
      error = std::string("cannot parse metadata dof_names: ") + e.what();
      return false;
    }

    stage = "parsing gains and action scale";
    if (!JsonFloatArray(metadata, "kp", kp_, error) || !JsonFloatArray(metadata, "kd", kd_, error) ||
        !JsonFloatArray(metadata, "action_scale", action_scale_, error))
      return false;

    stage = "parsing experiment_config";
    json experiment;
    try {
      experiment = json::parse(metadata.at("experiment_config"));
    } catch (const std::exception &e) {
      error = std::string("cannot parse metadata experiment_config: ") + e.what();
      return false;
    }

    stage = "parsing wbt_history_initialization";
    const auto history = metadata.find("wbt_history_initialization");
    if (history != metadata.end()) {
      // Exports spell the two modes "repeat_first" and "zero"; the runtime mode still decides.
      if (history->second == "repeat_first") {
        declared_history_ = WbtHistoryInitialization::kRepeatFirstValidObs;
      } else if (history->second == "zero") {
        declared_history_ = WbtHistoryInitialization::kZeroPad;
      } else {
        error = "metadata wbt_history_initialization must be zero or repeat_first";
        return false;
      }
    }

    stage = "parsing experiment_config.robot.control.action_clip_value";
    const auto &robot = experiment.at("robot");
    if (robot.contains("control")) {
      const auto &control = robot.at("control");
      if (!control.is_object()) {
        error = "experiment_config.robot.control must be an object";
        return false;
      }
      if (control.contains("clip_actions")) {
        if (!control.at("clip_actions").is_boolean()) {
          error = "experiment_config.robot.control.clip_actions must be a boolean";
          return false;
        }
        training_action_clip_enabled = control.at("clip_actions").get<bool>();
        if (!training_action_clip_enabled) action_clip_source_ = "ceiling; checkpoint disables training clip";
      }
      if (control.contains("action_clip_value")) {
        const auto &value = control.at("action_clip_value");
        if (!value.is_number()) {
          error = "experiment_config.robot.control.action_clip_value must be a finite positive number";
          return false;
        }
        const float declared_clip = value.get<float>();
        if (!std::isfinite(declared_clip) || declared_clip <= 0.0f) {
          error = "experiment_config.robot.control.action_clip_value must be a finite positive number";
          return false;
        }
        std::ostringstream source;
        if (!training_action_clip_enabled) {
          source << "ceiling; checkpoint disables training clip";
        } else if (declared_clip <= kWbtActionCeiling) {
          action_clip_ = declared_clip;
          source << "checkpoint";
        } else {
          source << "ceiling; checkpoint declares " << declared_clip;
        }
        action_clip_source_ = source.str();
      }
    }

    stage = "parsing default_joint_angles";
    try {
      const json &defaults = experiment.at("robot").at("init_state").at("default_joint_angles");
      if (!defaults.is_object() || defaults.empty()) {
        error = "experiment_config default_joint_angles is missing or empty";
        return false;
      }
      for (int i = 0; i < kActionDim; ++i) {
        if (!defaults.contains(joint_names_[i])) {
          error = "experiment_config default_joint_angles is missing \"" + joint_names_[i] + "\"";
          return false;
        }
        default_dof_pos_[i] = defaults.at(joint_names_[i]).get<float>();
      }
    } catch (const std::exception &e) {
      error = std::string("cannot parse default_joint_angles: ") + e.what();
      return false;
    }
    if (!IsFinite(default_dof_pos_.data(), default_dof_pos_.size())) {
      error = "experiment_config default_joint_angles contains a non-finite value";
      return false;
    }

    stage = "parsing motion_config.body_name_ref";
    try {
      const json &motion_config =
          experiment.at("command").at("setup_terms").at("motion_command").at("params").at("motion_config");
      if (motion_config.contains("body_name_ref")) {
        const json &body = motion_config.at("body_name_ref");
        if (body.is_array() && !body.empty() && !body.at(0).is_null())
          anchor_body_name_ = body.at(0).get<std::string>();
        else if (body.is_string())
          anchor_body_name_ = body.get<std::string>();
      }
    } catch (const std::exception &e) {
      error = std::string("cannot parse motion_config.body_name_ref: ") + e.what();
      return false;
    }
    if (anchor_body_name_.empty()) anchor_body_name_ = "pelvis";

    stage = "initializing anchor FK";
    if (anchor_body_name_ != "pelvis") {
      try {
        json urdf_json = json::parse(metadata.at("robot_urdf"), nullptr, false);
        const std::string urdf = urdf_json.is_string() ? urdf_json.get<std::string>() : metadata.at("robot_urdf");
        fk_body_ = std::make_unique<FkBody>(urdf, anchor_body_name_, joint_names_);
      } catch (const std::exception &e) {
        error = std::string("cannot initialize URDF FK for anchor \"") + anchor_body_name_ + "\": " + e.what();
        return false;
      }
    }

    stage = "parsing observation groups";
    json groups;
    try {
      groups = experiment.at("observation").at("groups");
      observation_clip_ = experiment.at("observation").value("clip_observations", 100.0f);
    } catch (const std::exception &e) {
      error = std::string("cannot parse observation groups: ") + e.what();
      return false;
    }
    if (!groups.is_object() || groups.empty()) {
      error = "experiment_config has no observation groups";
      return false;
    }
    if (!std::isfinite(observation_clip_) || observation_clip_ <= 0.0f) {
      error = "observation clip must be a positive finite value";
      return false;
    }

    stage = "resolving actor input groups";
    std::vector<std::string> actor_groups;
    try {
      const json &input_dim = experiment.at("algo").at("config").at("module_dict").at("actor").at("input_dim");
      if (input_dim.is_array()) {
        for (const auto &name : input_dim) actor_groups.push_back(name.get<std::string>());
      }
    } catch (const std::exception &) {
      // Older exports omit the actor module config. Use the maintained
      // inference fallback: actor_obs first, then the remaining actor* groups
      // sorted.
    }
    if (actor_groups.empty()) {
      if (groups.contains("actor_obs")) actor_groups.push_back("actor_obs");
      std::vector<std::string> remaining;
      for (auto it = groups.begin(); it != groups.end(); ++it) {
        if (it.key() != "actor_obs" && it.key().compare(0, 5, "actor") == 0) remaining.push_back(it.key());
      }
      std::sort(remaining.begin(), remaining.end());
      actor_groups.insert(actor_groups.end(), remaining.begin(), remaining.end());
    }
    if (actor_groups.empty()) {
      error = "experiment_config names no actor observation group";
      return false;
    }

    stage = "resolving the motion command width";
    // Determine whether the base command contains position only or position and
    // velocity before sizing any lookahead term.
    for (const std::string &group_name : actor_groups) {
      if (!groups.contains(group_name)) {
        error = "actor input_dim names missing observation group \"" + group_name + "\"";
        return false;
      }
      const json &term_map = groups.at(group_name).value("terms", json::object());
      for (auto it = term_map.begin(); it != term_map.end(); ++it) {
        if (it.key().find("motion_command") != std::string::npos && it.key().find("lookahead") == std::string::npos) {
          const std::string func = it.value().value("func", "");
          const int dim = func.find("joint_pos_only") != std::string::npos ? kActionDim : 2 * kActionDim;
          if (motion_command_dim_ != 0 && motion_command_dim_ != dim) {
            error = "actor groups disagree on motion_command width";
            return false;
          }
          motion_command_dim_ = dim;
        }
      }
    }
    stage = "building the ordered observation layout";
    terms_.clear();
    trained_lookahead_offsets_.clear();
    std::set<int> ordinals;
    int expected_obs_dim = 0;
    std::set<int> flat_future_kinds;
    for (const std::string &group_name : actor_groups) {
      const json &group = groups.at(group_name);
      const int history_length = std::max(1, group.value("history_length", 1));
      const json &term_map = group.value("terms", json::object());
      if (!term_map.is_object() || term_map.empty()) {
        error = "actor observation group \"" + group_name + "\" has no terms";
        return false;
      }

      struct OrderedTerm {
        int ordinal;
        std::string name;
        json config;
      };
      std::vector<OrderedTerm> ordered;
      for (auto it = term_map.begin(); it != term_map.end(); ++it) {
        int ordinal = -1;
        if (!ParseTermOrdinal(it.key(), ordinal)) {
          error = "observation term \"" + it.key() + "\" lacks an obsN_ prefix";
          return false;
        }
        if (!ordinals.insert(ordinal).second) {
          error = "observation term ordinal " + std::to_string(ordinal) + " is duplicated";
          return false;
        }
        ordered.push_back({ordinal, it.key(), it.value()});
      }
      std::sort(ordered.begin(), ordered.end(),
                [](const OrderedTerm &a, const OrderedTerm &b) { return a.ordinal < b.ordinal; });

      for (const OrderedTerm &entry : ordered) {
        const std::string func = entry.config.value("func", "");
        Term term;
        term.name = entry.name;
        term.history_length = history_length;
        term.scale = entry.config.value("scale", 1.0f);
        if (!std::isfinite(term.scale)) {
          error = "observation term \"" + term.name + "\" has a non-finite scale";
          return false;
        }
        if (entry.config.contains("clip") && !entry.config.at("clip").is_null()) {
          error = "observation term \"" + term.name +
                  "\" uses per-term clipping, which the C++ WBT runtime does "
                  "not support";
          return false;
        }

        const std::string leaf = FuncLeaf(func);
        std::vector<int> flat_offsets;
        int flat_kind = -1;
        const int zero_width = FlatZeroWidth(leaf);
        if (FlatFuture(leaf, flat_offsets, flat_kind)) {
          if (group_name != "actor_targets" || group.value("history_length", 1) != 1 || motion_command_dim_ != 0 ||
              !flat_future_kinds.insert(flat_kind).second ||
              (!trained_lookahead_offsets_.empty() && trained_lookahead_offsets_ != flat_offsets)) {
            error =
                "flat actor_targets requires unique "
                "position/velocity/orientation terms with one shared "
                "horizon and history 1";
            return false;
          }
          trained_lookahead_offsets_ = flat_offsets;
          term.kind = flat_kind == 0   ? TermKind::kJointPosFuture
                      : flat_kind == 1 ? TermKind::kJointVelFuture
                                       : TermKind::kFutureRefOri;
          term.dim = static_cast<int>(flat_offsets.size()) * (flat_kind == 2 ? 6 : kActionDim);
        } else if (zero_width || leaf == "sonic_encoder_index") {
          if (group_name != "actor_targets" || group.value("history_length", 1) != 1) {
            error = "flat actor_targets selector/zero blocks require history 1";
            return false;
          }
          term.kind = zero_width ? TermKind::kZeroBlock : TermKind::kEncoderIndex;
          term.dim = zero_width ? zero_width : 3;
        } else if (term.name.find("motion_command_lookahead") != std::string::npos) {
          if (!trained_lookahead_offsets_.empty()) {
            error = "more than one motion_command_lookahead term is unsupported";
            return false;
          }
          if (!ResolveLookaheadOffsets(func, trained_lookahead_offsets_)) {
            error = "unsupported lookahead function \"" + func + "\"";
            return false;
          }
          term.kind = TermKind::kMotionCommandLookahead;
          term.dim = static_cast<int>(trained_lookahead_offsets_.size()) * motion_command_dim_;
        } else if (term.name.find("motion_command") != std::string::npos) {
          term.kind = TermKind::kMotionCommand;
          term.dim = motion_command_dim_;
        } else if (term.name.find("motion_ref_ori_b") != std::string::npos) {
          term.kind = TermKind::kMotionRefOri;
          term.dim = 6;
        } else if (term.name.find("projected_gravity") != std::string::npos) {
          term.kind = TermKind::kProjectedGravity;
          term.dim = 3;
        } else if (term.name.find("base_ang_vel") != std::string::npos) {
          term.kind = TermKind::kBaseAngVel;
          term.dim = 3;
        } else if (term.name.find("dof_pos") != std::string::npos) {
          term.kind = TermKind::kDofPos;
          term.dim = kActionDim;
        } else if (term.name.find("dof_vel") != std::string::npos) {
          term.kind = TermKind::kDofVel;
          term.dim = kActionDim;
        } else if (term.name.find("actions") != std::string::npos) {
          term.kind = TermKind::kActions;
          term.dim = kActionDim;
        } else {
          error = "unsupported live actor observation term \"" + term.name + "\"";
          return false;
        }

        expected_obs_dim += term.dim * term.history_length;
        terms_.push_back(std::move(term));
      }
    }
    if (motion_command_dim_ == 0 && flat_future_kinds.size() != 3) {
      error =
          "actor observation requires motion_command or complete flat "
          "position/velocity/orientation targets";
      return false;
    }
    stage = "allocating the observation input";
    obs_input_.assign(static_cast<std::size_t>(expected_obs_dim), 0.0f);
  } catch (const std::exception &e) {
    error = "unexpected exception while " + stage + ": " + e.what();
    return false;
  }
  return true;
}

bool WbtPolicy::ValidateShapes(std::string &error) {
  bool have_obs = false;
  bool have_time = false;
  bool have_actions = false;
  for (const auto &info : session_->inputs()) {
    const auto &name = info.name;
    const auto &shape = info.shape;
    if (name == "obs") {
      have_obs = true;
      const std::vector<int64_t> want{1, static_cast<int64_t>(obs_input_.size())};
      if (shape != want) {
        error = "ONNX obs shape does not match the metadata-derived width " + std::to_string(obs_input_.size());
        return false;
      }
    } else if (name == "time_step") {
      have_time = true;
      if (shape != std::vector<int64_t>({1, 1})) {
        error = "ONNX time_step input must have shape [1,1]";
        return false;
      }
    } else {
      error = "unsupported ONNX input \"" + name + "\"";
      return false;
    }
  }
  if (!have_obs || !have_time || session_->inputs().size() != 2) {
    error = "single-file WBT graph must have exactly obs and time_step inputs";
    return false;
  }

  for (const auto &info : session_->outputs()) {
    const auto &name = info.name;
    if (name != "actions") continue;
    have_actions = true;
    if (info.shape != std::vector<int64_t>({1, kActionDim})) {
      error = "ONNX actions output must be float [1,29]";
      return false;
    }
  }
  if (!have_actions) {
    error = "single-file WBT graph has no actions output";
    return false;
  }
  return true;
}

bool WbtPolicy::AnchorOrientation(const std::array<float, 4> &root_quat_wxyz,
                                  const std::array<float, kActionDim> &dof_pos, math::Quatf &orientation) const {
  if (fk_body_) return fk_body_->Orientation(root_quat_wxyz, dof_pos, orientation);
  orientation = SafeQuat(root_quat_wxyz);
  return true;
}

bool WbtPolicy::BuildTermValue(const Term &term, const RobotObservation &obs, const WbtReferenceSequence &reference,
                               const math::Quatf &robot_anchor, const math::Quatf &motion_anchor,
                               std::vector<float> &value) const {
  value.assign(static_cast<std::size_t>(term.dim), 0.0f);
  switch (term.kind) {
    case TermKind::kMotionCommand:
      std::copy(reference.current.dof_pos.begin(), reference.current.dof_pos.end(), value.begin());
      if (motion_command_dim_ == 2 * kActionDim) {
        std::copy(reference.current.dof_vel.begin(), reference.current.dof_vel.end(), value.begin() + kActionDim);
      }
      break;
    case TermKind::kMotionCommandLookahead: {
      if (static_cast<int>(reference.lookahead.size()) != LookaheadSlots()) return false;
      std::size_t offset = 0;
      for (const WbtReferenceFrame &frame : reference.lookahead) {
        std::copy(frame.dof_pos.begin(), frame.dof_pos.end(), value.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += kActionDim;
        if (motion_command_dim_ == 2 * kActionDim) {
          std::copy(frame.dof_vel.begin(), frame.dof_vel.end(), value.begin() + static_cast<std::ptrdiff_t>(offset));
          offset += kActionDim;
        }
      }
      break;
    }
    case TermKind::kMotionRefOri: {
      const math::Quatf relative = robot_anchor.inverse() * motion_anchor;
      QuatToRot6d(relative, value.data());
      break;
    }
    case TermKind::kJointPosFuture:
    case TermKind::kJointVelFuture:
    case TermKind::kFutureRefOri: {
      if (static_cast<int>(reference.lookahead.size()) != LookaheadSlots()) return false;
      std::size_t offset = 0;
      for (const auto &frame : reference.lookahead) {
        if (term.kind == TermKind::kFutureRefOri) {
          math::Quatf slot_raw;
          if (!AnchorOrientation(frame.root_quat_wxyz, frame.dof_pos, slot_raw)) return false;
          const math::Quatf slot = (motion_yaw_inv_ * slot_raw).normalized();
          const math::Quatf relative = robot_anchor.inverse() * slot;
          QuatToRot6d(relative, value.data() + offset);
          offset += 6;
        } else {
          const auto &source = term.kind == TermKind::kJointPosFuture ? frame.dof_pos : frame.dof_vel;
          std::copy(source.begin(), source.end(), value.begin() + offset);
          offset += kActionDim;
        }
      }
      break;
    }
    case TermKind::kZeroBlock:
      break;
    case TermKind::kEncoderIndex:
      value[0] = 1.0f;
      break;
    case TermKind::kProjectedGravity: {
      const Eigen::Vector3f gravity =
          obs.projected_gravity
              ? Eigen::Vector3f((*obs.projected_gravity)[0], (*obs.projected_gravity)[1], (*obs.projected_gravity)[2])
              : math::projected_gravity(SafeQuat(obs.root_quat_wxyz));
      value[0] = gravity.x();
      value[1] = gravity.y();
      value[2] = gravity.z();
      break;
    }
    case TermKind::kBaseAngVel:
      std::copy(obs.base_ang_vel.begin(), obs.base_ang_vel.end(), value.begin());
      break;
    case TermKind::kDofPos:
      for (int i = 0; i < kActionDim; ++i) value[static_cast<std::size_t>(i)] = obs.dof_pos_mj[i] - default_dof_pos_[i];
      break;
    case TermKind::kDofVel:
      std::copy(obs.dof_vel_mj.begin(), obs.dof_vel_mj.end(), value.begin());
      break;
    case TermKind::kActions:
      std::copy(last_action_.begin(), last_action_.end(), value.begin());
      break;
  }
  for (float &v : value) v *= term.scale;
  return IsFinite(value);
}

bool WbtPolicy::ComputeAction(const RobotObservation &obs, const WbtReferenceSequence &reference, float *action_out) {
  if (!initialized_ || action_out == nullptr) return false;
  if (!IsFinite(obs.dof_pos_mj.data(), obs.dof_pos_mj.size()) ||
      !IsFinite(obs.dof_vel_mj.data(), obs.dof_vel_mj.size()) ||
      !IsFinite(obs.base_ang_vel.data(), obs.base_ang_vel.size()) ||
      !IsFinite(reference.current.dof_pos.data(), reference.current.dof_pos.size()) ||
      !IsFinite(reference.current.dof_vel.data(), reference.current.dof_vel.size())) {
    std::cerr << "[WbtPolicy] non-finite robot/reference observation" << std::endl;
    return false;
  }
  for (const WbtReferenceFrame &frame : reference.lookahead) {
    if (!IsFinite(frame.dof_pos.data(), frame.dof_pos.size()) ||
        !IsFinite(frame.dof_vel.data(), frame.dof_vel.size())) {
      std::cerr << "[WbtPolicy] non-finite lookahead frame" << std::endl;
      return false;
    }
  }

  math::Quatf robot_raw;
  math::Quatf motion_raw;
  if (!AnchorOrientation(obs.root_quat_wxyz, obs.dof_pos_mj, robot_raw) ||
      !AnchorOrientation(reference.current.root_quat_wxyz, reference.current.dof_pos, motion_raw)) {
    std::cerr << "[WbtPolicy] anchor FK failed" << std::endl;
    return false;
  }
  if (!heading_initialized_) {
    robot_yaw_inv_ = HeadingQuat(robot_raw).inverse();
    motion_yaw_inv_ = HeadingQuat(motion_raw).inverse();
    heading_initialized_ = true;
  }
  const math::Quatf robot_anchor = (robot_yaw_inv_ * robot_raw).normalized();
  const math::Quatf motion_anchor = (motion_yaw_inv_ * motion_raw).normalized();

  std::vector<std::vector<float>> values(terms_.size());
  for (std::size_t i = 0; i < terms_.size(); ++i) {
    if (!BuildTermValue(terms_[i], obs, reference, robot_anchor, motion_anchor, values[i])) {
      std::cerr << "[WbtPolicy] failed to build finite observation term " << terms_[i].name << std::endl;
      return false;
    }
  }

  std::size_t output_offset = 0;
  for (std::size_t i = 0; i < terms_.size(); ++i) {
    Term &term = terms_[i];
    if (term.history.empty() && history_initialization_ == WbtHistoryInitialization::kRepeatFirstValidObs) {
      term.history.assign(static_cast<std::size_t>(term.history_length), values[i]);
    } else {
      term.history.push_back(values[i]);
    }
    while (term.history.size() > static_cast<std::size_t>(term.history_length)) term.history.pop_front();

    const int missing = term.history_length - static_cast<int>(term.history.size());
    for (int frame = 0; frame < missing; ++frame) {
      std::fill(obs_input_.begin() + static_cast<std::ptrdiff_t>(output_offset),
                obs_input_.begin() + static_cast<std::ptrdiff_t>(output_offset + term.dim), 0.0f);
      output_offset += static_cast<std::size_t>(term.dim);
    }
    for (const std::vector<float> &frame : term.history) {
      std::copy(frame.begin(), frame.end(), obs_input_.begin() + static_cast<std::ptrdiff_t>(output_offset));
      output_offset += frame.size();
    }
  }
  if (output_offset != obs_input_.size()) {
    std::cerr << "[WbtPolicy] observation builder wrote " << output_offset << " floats into a " << obs_input_.size()
              << "-float input" << std::endl;
    return false;
  }
  for (float &v : obs_input_) v = std::clamp(v, -observation_clip_, observation_clip_);

  float time_step = NextTimeStep();
  last_time_step_ = time_step;
  try {
    std::copy(obs_input_.begin(), obs_input_.end(), session_->input("obs").begin());
    session_->input("time_step")[0] = time_step;
    session_->run();
    const auto &raw = session_->output("actions");
    for (int i = 0; i < kActionDim; ++i) {
      if (!std::isfinite(raw[i])) {
        std::cerr << "[WbtPolicy] ONNX produced a non-finite action" << std::endl;
        return false;
      }
    }
    for (int i = 0; i < kActionDim; ++i) {
      last_raw_action_[i] = raw[i];
      last_action_[i] = std::clamp(raw[i], -action_clip_, action_clip_);
      action_out[i] = last_action_[i] * action_scale_[i];
    }
  } catch (const std::exception &e) {
    std::cerr << "[WbtPolicy] inference failed: " << e.what() << std::endl;
    return false;
  }
  ++tick_;
  return true;
}

void WbtPolicy::SetControlInterval(double interval_s) {
  if (!std::isfinite(interval_s) || interval_s <= 0.0)
    throw std::invalid_argument("WbtPolicy control interval must be finite and positive");
  control_interval_s_ = interval_s;
}

// Mirrors holosoma_inference TimestepUtil.get_timestep: the motion phase is
// elapsed wall time divided by the control interval, NOT a count of policy
// ticks. The two diverge exactly when a tick is late or dropped, and the
// divergence is permanent because a tick counter never catches up.
float WbtPolicy::NextTimeStep() {
  if (!elapsed_seconds_fn_) {
    // No clock wired (unit tests, and any caller that has not opted in): fall
    // back to the tick count so behaviour is unchanged rather than frozen.
    return static_cast<float>(tick_);
  }
  const double now = elapsed_seconds_fn_();
  if (!std::isfinite(now)) return last_time_step_;
  if (!engage_anchored_) {
    engage_elapsed_s_ = now;
    engage_anchored_ = true;
    return 0.0f;
  }
  const double elapsed = now - engage_elapsed_s_;
  if (elapsed <= 0.0) return 0.0f;
  // Nudge before flooring. `now` is steady_clock seconds since epoch, so it is
  // large (~1e5+ on a booted machine) and `now - anchor` carries double
  // subtraction error: an exact 0.02s advance can land at 0.019999999999996,
  // which floors to 0 instead of 1 and stalls the phase a step behind forever.
  // 1e-6 of a step is 20ns at 50Hz — far below any real timing effect, and
  // enough to absorb that error at realistic clock magnitudes.
  const double steps = std::floor(elapsed / control_interval_s_ + kTimeStepEpsilon);
  // TimestepUtil re-anchors instead of accepting a large jump on the first
  // step, for the same reason: a clock that jumped ahead before the first tick
  // would otherwise start the policy mid-motion.
  if (last_time_step_ == 0.0f && steps > 1.0) {
    engage_elapsed_s_ = now;
    return 0.0f;
  }
  return static_cast<float>(steps);
}

void WbtPolicy::ResetHeading() {
  heading_initialized_ = false;
  robot_yaw_inv_ = math::Quatf::Identity();
  motion_yaw_inv_ = math::Quatf::Identity();
  tick_ = 0;
  // Re-anchor the motion phase too: engage and unpause both mean "start this
  // motion from the top", which is what TimestepUtil.reset does on the python
  // path. Leaving the anchor stale would resume mid-motion.
  engage_anchored_ = false;
  engage_elapsed_s_ = 0.0;
  last_time_step_ = 0.0f;
}

void WbtPolicy::Reset() {
  ResetHeading();
  last_action_.fill(0.0f);
  last_raw_action_.fill(0.0f);
  for (Term &term : terms_) term.history.clear();
  std::fill(obs_input_.begin(), obs_input_.end(), 0.0f);
}

std::vector<int> WbtPolicy::LookaheadSampleOffsets(int horizon_frames) const {
  if (horizon_frames < 0) return {};
  std::vector<int> result;
  result.reserve(trained_lookahead_offsets_.size());
  const int trained_horizon = TrainedLookaheadFrames();
  for (int offset : trained_lookahead_offsets_) {
    const int sampled = trained_horizon == 0
                            ? 0
                            : static_cast<int>(std::lround(static_cast<double>(offset) * horizon_frames /
                                                           static_cast<double>(trained_horizon)));
    result.push_back(std::clamp(sampled, 0, horizon_frames));
  }
  return result;
}

}  // namespace holosoma::live
