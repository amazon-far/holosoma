// Command line handling that mirrors holosoma_inference/run_policy.py:
//
//   holosoma_run_policy inference:g1-29dof-wbt --task.model-path model.onnx --task.interface eth0
//
// The positional preset is either `inference:<name>` (looked up in the preset
// directory) or a path to a YAML file. Every `--<section>.<field>` flag
// overrides one preset field (kebab-case is accepted), `--secondary none`
// disables dual mode and `--secondary-preset <name>` swaps the secondary.
#pragma once

#include <string>
#include <vector>

#include "holosoma_cpp/config.hpp"

namespace holosoma {

struct CliOptions {
  InferenceConfig config;
  std::string preset_path;
  bool show_help = false;
};

// Directory holding <name>.yaml presets: $HOLOSOMA_CPP_PRESET_DIR, else the compiled-in default.
std::string default_preset_dir();

std::vector<std::string> list_presets(const std::string& preset_dir);

CliOptions parse_cli(const std::vector<std::string>& args, const std::string& preset_dir);

std::string cli_usage(const std::string& program, const std::string& preset_dir);

}  // namespace holosoma
