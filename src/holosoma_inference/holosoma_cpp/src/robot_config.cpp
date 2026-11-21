#include "holosoma_cpp/robot_config.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace holosoma {
namespace {

using json = nlohmann::json;

void ExpandEntry(const std::string &entry, const json &tags, const std::set<std::string> &names,
                 std::vector<std::string> &chain, std::set<std::string> &expanded) {
  if (!entry.empty() && entry.front() == '#') {
    if (std::find(chain.begin(), chain.end(), entry) != chain.end())
      throw std::invalid_argument("joint tag cycle at " + entry);
    if (!tags.contains(entry)) throw std::invalid_argument("Unknown joint tag " + entry);
    const auto &entries = tags.at(entry);
    if (!entries.is_array() || entries.empty())
      throw std::invalid_argument("joint tag " + entry + " must be a non-empty list");
    chain.push_back(entry);
    for (const auto &child : entries) ExpandEntry(child.get<std::string>(), tags, names, chain, expanded);
    chain.pop_back();
  } else {
    if (!names.count(entry)) throw std::invalid_argument("Unknown joint name " + entry);
    expanded.insert(entry);
  }
}

std::set<std::string> Expand(const std::string &entry, const json &tags, const std::set<std::string> &names) {
  std::vector<std::string> chain;
  std::set<std::string> expanded;
  ExpandEntry(entry, tags, names, chain, expanded);
  return expanded;
}

}  // namespace

std::vector<float> ResolveDefaultDofPositions(const json &robot, const std::vector<std::string> &metadata_dof_names) {
  const auto saved_names = robot.value("dof_names", std::vector<std::string>{});
  if (!saved_names.empty() && !metadata_dof_names.empty() && saved_names != metadata_dof_names)
    throw std::invalid_argument("ONNX dof_names and experiment_config.robot.dof_names disagree");
  const auto &order = metadata_dof_names.empty() ? saved_names : metadata_dof_names;
  const std::set<std::string> names(order.begin(), order.end());
  if (names.size() != order.size()) throw std::invalid_argument("dof_names has duplicate entries");

  const auto &init = robot.at("init_state");
  const bool current = init.contains("default_joint_positions");
  const bool legacy = init.contains("default_joint_angles");
  if (!current && !legacy) throw std::invalid_argument("robot.init_state is missing default_joint_positions");
  if (current && legacy && init.at("default_joint_positions") != init.at("default_joint_angles"))
    throw std::invalid_argument("conflicting default_joint_positions and default_joint_angles");
  const auto defaults =
      current ? init.at("default_joint_positions") : init.value("default_joint_angles", json::object());
  if (!defaults.is_object()) throw std::invalid_argument("default_joint_positions must be a mapping");
  if (!defaults.empty() && order.empty()) throw std::invalid_argument("Default joint positions require dof_names");

  json tags = robot.value("joint_tags", json::object());
  if (!tags.is_object()) throw std::invalid_argument("joint_tags must be a mapping");
  if (!order.empty() && !tags.contains("#all")) tags["#all"] = order;
  for (const auto &tag : tags.items()) {
    if (tag.key().size() < 2 || tag.key().front() != '#')
      throw std::invalid_argument("joint tag names require a leading '#'");
    Expand(tag.key(), tags, names);  // Validate unused tags too.
  }

  std::map<std::string, float> positions;
  std::map<std::string, std::string> key_for;
  for (const auto &assignment : defaults.items()) {
    const auto covered = Expand(assignment.key(), tags, names);
    if (!assignment.value().is_number()) throw std::invalid_argument("default joint positions must be numeric");
    const float value = assignment.value().get<float>();
    if (!std::isfinite(value)) throw std::invalid_argument("default joint positions must be finite");
    for (const auto &name : covered) {
      if (key_for.count(name))
        throw std::invalid_argument("keys " + key_for.at(name) + " and " + assignment.key() + " both cover DOF " +
                                    name);
      key_for[name] = assignment.key();
      positions[name] = value;
    }
  }
  std::vector<float> result;
  result.reserve(order.size());
  for (const auto &name : order) {
    if (!positions.count(name) && legacy) throw std::invalid_argument("No default position covers DOF " + name);
    result.push_back(positions.count(name) ? positions.at(name) : 0.0f);
  }
  return result;
}

}  // namespace holosoma
