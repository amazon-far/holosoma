#include "holosoma_cpp/cli.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>

#ifndef HOLOSOMA_CPP_DEFAULT_PRESET_DIR
#define HOLOSOMA_CPP_DEFAULT_PRESET_DIR ""
#endif

namespace holosoma {
namespace {

namespace fs = std::filesystem;

constexpr const char* kPresetPrefix = "inference:";

std::string to_snake(std::string s) {
  std::replace(s.begin(), s.end(), '-', '_');
  return s;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

// Only plain `true`/`false` count: quoted strings and YAML 1.1 words like `on`/`no` stay strings.
bool is_bool_scalar(const YAML::Node& node) {
  if (!node || !node.IsScalar() || node.Tag() == "!") {
    return false;
  }
  const std::string& s = node.Scalar();
  return s == "true" || s == "false" || s == "True" || s == "False";
}

bool is_null_token(const std::string& v) { return v == "None" || v == "none" || v == "null" || v == "~"; }

YAML::Node scalar_or_null(const std::string& value) {
  if (is_null_token(value)) {
    return YAML::Node(YAML::NodeType::Null);
  }
  return YAML::Node(value);
}

std::string resolve_cli_path(const std::string& path) {
  if (path.empty() || path.rfind("wandb://", 0) == 0 || path.rfind("https://", 0) == 0 ||
      path.rfind("http://", 0) == 0) {
    return path;
  }
  return fs::absolute(fs::path(path)).lexically_normal().string();
}

fs::path preset_file(const std::string& spec, const std::string& preset_dir) {
  if (spec.rfind(kPresetPrefix, 0) == 0) {
    const std::string name = spec.substr(std::string(kPresetPrefix).size());
    const fs::path path = fs::path(preset_dir) / (name + ".yaml");
    if (!fs::exists(path)) {
      std::ostringstream os;
      os << "Unknown preset '" << spec << "'. Available:";
      for (const auto& preset : list_presets(preset_dir)) {
        os << " " << kPresetPrefix << preset;
      }
      throw ConfigError(os.str());
    }
    return path;
  }
  if (!fs::exists(spec)) {
    throw ConfigError("Preset file not found: " + spec);
  }
  return fs::path(spec);
}

YAML::Node load_yaml(const fs::path& path) {
  try {
    return YAML::LoadFile(path.string());
  } catch (const YAML::Exception& e) {
    throw ConfigError("Failed to read " + path.string() + ": " + e.what());
  }
}

// One `--a.b.c v1 v2` occurrence.
struct Override {
  std::string flag;
  std::vector<std::string> path;
  std::vector<std::string> values;
  bool has_inline_value = false;
};

void apply_override(YAML::Node& root, const Override& o) {
  static const std::vector<std::string> kSections = {"robot", "observation", "task", "runtime", "secondary"};
  if (std::find(kSections.begin(), kSections.end(), o.path.front()) == kSections.end()) {
    throw ConfigError("Unknown option " + o.flag);
  }

  YAML::Node node = root;
  for (size_t i = 0; i + 1 < o.path.size(); ++i) {
    const std::string& key = o.path[i];
    YAML::Node next = node[key];
    if (!next || next.IsNull()) {
      if (key == "runtime") {
        node[key] = YAML::Node(YAML::NodeType::Map);
        next.reset(node[key]);
      } else if (key == "secondary") {
        throw ConfigError(o.flag + ": this preset has no secondary policy");
      } else {
        throw ConfigError("Unknown option " + o.flag);
      }
    }
    if (!next.IsMap()) {
      throw ConfigError("Unknown option " + o.flag);
    }
    node.reset(next);
  }

  std::string key = o.path.back();
  const bool in_runtime = std::find(o.path.begin(), o.path.end(), "runtime") != o.path.end();
  YAML::Node current = node[key];
  bool negate = false;
  if (!current && key.rfind("no_", 0) == 0 && is_bool_scalar(node[key.substr(3)])) {
    key = key.substr(3);
    current.reset(node[key]);
    negate = true;
  }
  if (!current && !in_runtime) {
    throw ConfigError("Unknown option " + o.flag);
  }

  const bool is_model_path = key == "model_path";
  if (current && is_bool_scalar(current)) {
    bool value = !negate;
    if (!o.values.empty()) {
      if (o.values.size() != 1 || negate || !YAML::convert<bool>::decode(YAML::Node(o.values[0]), value)) {
        throw ConfigError(o.flag + ": expected no value or a single true/false");
      }
    }
    node[key] = value;
    return;
  }
  if (negate) {
    throw ConfigError("Unknown option " + o.flag);
  }
  if (o.values.empty()) {
    throw ConfigError(o.flag + ": missing value");
  }
  if (is_model_path) {
    YAML::Node seq(YAML::NodeType::Sequence);
    for (const auto& v : o.values) {
      seq.push_back(resolve_cli_path(v));
    }
    node[key] = seq;
    return;
  }
  if ((current && current.IsSequence()) || o.values.size() > 1) {
    if (o.values.size() == 1 && is_null_token(o.values[0])) {
      node[key] = YAML::Node(YAML::NodeType::Null);
      return;
    }
    YAML::Node seq(YAML::NodeType::Sequence);
    seq.SetStyle(YAML::EmitterStyle::Flow);
    for (const auto& v : o.values) {
      seq.push_back(YAML::Node(v));
    }
    node[key] = seq;
    return;
  }
  if (current && current.IsMap()) {
    throw ConfigError(o.flag + ": is a group; set one of its fields instead");
  }
  node[key] = scalar_or_null(o.values[0]);
}

}  // namespace

std::string default_preset_dir() {
  if (const char* env = std::getenv("HOLOSOMA_CPP_PRESET_DIR")) {
    return env;
  }
  return HOLOSOMA_CPP_DEFAULT_PRESET_DIR;
}

std::vector<std::string> list_presets(const std::string& preset_dir) {
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(preset_dir, ec)) {
    // Skip hidden files (editor swap files, macOS "._" resource forks).
    if (entry.path().extension() == ".yaml" && entry.path().filename().string().front() != '.') {
      names.push_back(entry.path().stem().string());
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

CliOptions parse_cli(const std::vector<std::string>& args, const std::string& preset_dir) {
  CliOptions options;
  std::string preset_spec;
  std::string secondary_preset;
  bool disable_secondary = false;
  std::vector<Override> overrides;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      options.show_help = true;
      return options;
    }
    if (arg.rfind("--", 0) != 0) {
      if (!preset_spec.empty()) {
        throw ConfigError("Unexpected argument '" + arg + "' (preset already given as '" + preset_spec + "')");
      }
      preset_spec = arg;
      continue;
    }

    std::string name = arg.substr(2);
    Override o;
    o.flag = arg;
    const auto eq = name.find('=');
    if (eq != std::string::npos) {
      o.values.push_back(name.substr(eq + 1));
      o.has_inline_value = true;
      name = name.substr(0, eq);
      o.flag = "--" + name;
    } else {
      while (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
        o.values.push_back(args[++i]);
      }
    }

    if (name == "secondary") {
      if (o.values.size() != 1 || (o.values[0] != "none" && o.values[0] != "None")) {
        throw ConfigError("--secondary only accepts 'none' (to disable dual mode)");
      }
      disable_secondary = true;
      continue;
    }
    if (name == "secondary-preset") {
      if (o.values.size() != 1) {
        throw ConfigError("--secondary-preset expects one preset name");
      }
      secondary_preset = o.values[0];
      continue;
    }
    for (const auto& part : split(name, '.')) {
      if (part.empty()) {
        throw ConfigError("Malformed option " + arg);
      }
      o.path.push_back(to_snake(part));
    }
    if (o.path.size() < 2) {
      throw ConfigError("Unknown option " + arg);
    }
    overrides.push_back(o);
  }

  if (preset_spec.empty()) {
    throw ConfigError("Missing preset (e.g. inference:g1-29dof-wbt)");
  }
  const fs::path preset_path = preset_file(preset_spec, preset_dir);
  options.preset_path = fs::absolute(preset_path).lexically_normal().string();
  YAML::Node root = load_yaml(preset_path);
  if (!root.IsMap()) {
    throw ConfigError(preset_path.string() + ": expected a mapping at the top level");
  }

  if (disable_secondary && !secondary_preset.empty()) {
    throw ConfigError("--secondary none and --secondary-preset are mutually exclusive");
  }
  if (!secondary_preset.empty()) {
    const fs::path secondary_path = preset_file(kPresetPrefix + secondary_preset, preset_dir);
    YAML::Node secondary = load_yaml(secondary_path);
    if (fs::absolute(secondary_path.parent_path()) != fs::absolute(preset_path.parent_path())) {
      throw ConfigError("--secondary-preset must live next to the primary preset (relative model paths)");
    }
    secondary["secondary"] = YAML::Node(YAML::NodeType::Null);
    secondary.remove("runtime");
    root["secondary"] = secondary;
  }

  for (const auto& o : overrides) {
    apply_override(root, o);
  }
  if (disable_secondary) {
    root["secondary"] = YAML::Node(YAML::NodeType::Null);
  }

  options.config = parse_inference_config(root, fs::absolute(preset_path).parent_path().string());
  validate_inference_config(options.config);
  return options;
}

std::string cli_usage(const std::string& program, const std::string& preset_dir) {
  std::ostringstream os;
  os << "Usage: " << program << " inference:<preset>|<preset.yaml> [--<section>.<field> VALUE...] [options]\n\n"
     << "Runs a holosoma ONNX policy on a Unitree robot or the holosoma MuJoCo bridge (DDS).\n\n"
     << "Presets (" << preset_dir << "):\n";
  for (const auto& preset : list_presets(preset_dir)) {
    os << "  " << kPresetPrefix << preset << "\n";
  }
  os << "\nCommon overrides (same names as the Python run_policy.py):\n"
     << "  --task.model-path PATH [PATH...]   ONNX model(s); up to nine, switch with keys 1-9\n"
     << "  --task.interface NAME             Network interface (lo for sim2sim, eth0 on the robot, auto)\n"
     << "  --task.use-joystick               Use the Unitree wireless controller instead of the keyboard\n"
     << "  --task.use-sim-time               WBT: advance the motion clip with the simulator clock\n"
     << "  --task.rl-rate HZ                 Control rate\n"
     << "  --task.motion-start-timestep N    WBT clip start frame (and --task.motion-end-timestep N)\n"
     << "  --secondary none                  Disable the dual-mode safety policy\n"
     << "  --secondary.task.model-path PATH  Override the secondary policy model\n"
     << "  --runtime.state-timeout-s S       Damp if robot state is older than S seconds (0 disables)\n";
  return os.str();
}

}  // namespace holosoma
