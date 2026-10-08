#include "holosoma_cpp/commands.hpp"

#include <array>
#include <utility>

namespace holosoma {
namespace {

constexpr std::array<std::pair<StateCommand, const char*>, 25> kNames = {{
    {StateCommand::START, "START"},
    {StateCommand::STOP, "STOP"},
    {StateCommand::INIT, "INIT"},
    {StateCommand::NEXT_POLICY, "NEXT_POLICY"},
    {StateCommand::KILL, "KILL"},
    {StateCommand::KP_UP, "KP_UP"},
    {StateCommand::KP_DOWN, "KP_DOWN"},
    {StateCommand::KP_UP_FINE, "KP_UP_FINE"},
    {StateCommand::KP_DOWN_FINE, "KP_DOWN_FINE"},
    {StateCommand::KP_RESET, "KP_RESET"},
    {StateCommand::SWITCH_POLICY_1, "SWITCH_POLICY_1"},
    {StateCommand::SWITCH_POLICY_2, "SWITCH_POLICY_2"},
    {StateCommand::SWITCH_POLICY_3, "SWITCH_POLICY_3"},
    {StateCommand::SWITCH_POLICY_4, "SWITCH_POLICY_4"},
    {StateCommand::SWITCH_POLICY_5, "SWITCH_POLICY_5"},
    {StateCommand::SWITCH_POLICY_6, "SWITCH_POLICY_6"},
    {StateCommand::SWITCH_POLICY_7, "SWITCH_POLICY_7"},
    {StateCommand::SWITCH_POLICY_8, "SWITCH_POLICY_8"},
    {StateCommand::SWITCH_POLICY_9, "SWITCH_POLICY_9"},
    {StateCommand::STAND_TOGGLE, "STAND_TOGGLE"},
    {StateCommand::ZERO_VELOCITY, "ZERO_VELOCITY"},
    {StateCommand::WALK, "WALK"},
    {StateCommand::STAND, "STAND"},
    {StateCommand::START_MOTION_CLIP, "START_MOTION_CLIP"},
    {StateCommand::SWITCH_MODE, "SWITCH_MODE"},
}};

}  // namespace

std::optional<int> switch_policy_index(StateCommand command) {
  const int first = static_cast<int>(StateCommand::SWITCH_POLICY_1);
  const int value = static_cast<int>(command);
  if (value >= first && value <= static_cast<int>(StateCommand::SWITCH_POLICY_9)) {
    return value - first;
  }
  return std::nullopt;
}

const char* to_string(StateCommand command) {
  for (const auto& [cmd, name] : kNames) {
    if (cmd == command) return name;
  }
  return "UNKNOWN";
}

std::optional<StateCommand> state_command_from_string(const std::string& name) {
  for (const auto& [cmd, cmd_name] : kNames) {
    if (name == cmd_name) return cmd;
  }
  return std::nullopt;
}

}  // namespace holosoma
