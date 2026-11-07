// Internal to the contextual front door: the pieces every command interface
// shares, and the two family constructors Policy chooses between.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "holosoma_cpp/contextual_kernel.hpp"

namespace holosoma::contextual::detail {
using Json = nlohmann::json;

template <class T>
void Finite(const T &values, const std::string &name) {
  if (!std::all_of(values.begin(), values.end(), [](auto x) { return std::isfinite(x); }))
    throw std::runtime_error(name + " contains non-finite values");
}

// ONNX metadata values are usually JSON; a value that does not parse is kept
// as its raw string.
Json Metadata(const OnnxModel &model);

// Nested JSON number lists flattened in row-major order; a scalar is one value.
std::vector<double> Numbers(const Json &value);

// Model joint names, as a JSON list or a comma-separated string.
std::vector<std::string> JointNames(const Json &value);

// Model-slot -> runtime index (real_to_model) and its inverse, from model joint
// names that must name every runtime joint exactly once.
void JointPermutation(const std::vector<std::string> &model_names, const std::vector<std::string> &runtime,
                      std::vector<int> &real_to_model, std::vector<int> &model_to_real);

using Chain = std::vector<std::pair<size_t, int>>;
// Resolve (joint name, axis) pairs to (runtime joint index, quaternion component).
Chain ResolveChain(const std::vector<std::pair<std::string, std::string>> &chain,
                   const std::vector<std::string> &joints);

// Validated state: joint widths, finite values, nonzero orientation.
void CheckState(const State &state, size_t joints);

// Projected gravity in the observation frame: forced upright, else the caller's
// IMU-frame gravity or the orientation's, then rotated through `chain`.
std::array<double, 3> ObservationGravity(const State &state, bool upright, const Chain &chain);

std::unique_ptr<Family> MakePrimitiveFamily(std::shared_ptr<OnnxModel> model, const Spec &spec, const Options &options);
std::unique_ptr<Family> MakeVelocityFamily(std::shared_ptr<OnnxModel> model, const Spec &spec, const Options &options);
}  // namespace holosoma::contextual::detail
