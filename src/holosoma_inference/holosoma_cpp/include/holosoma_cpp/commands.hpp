// Discrete operator commands and velocity commands (holosoma_inference/inputs/api/commands.py).
#pragma once

#include <optional>
#include <string>

namespace holosoma {

enum class StateCommand {
  // Common
  START,
  STOP,
  INIT,
  NEXT_POLICY,
  KILL,
  KP_UP,
  KP_DOWN,
  KP_UP_FINE,
  KP_DOWN_FINE,
  KP_RESET,
  SWITCH_POLICY_1,
  SWITCH_POLICY_2,
  SWITCH_POLICY_3,
  SWITCH_POLICY_4,
  SWITCH_POLICY_5,
  SWITCH_POLICY_6,
  SWITCH_POLICY_7,
  SWITCH_POLICY_8,
  SWITCH_POLICY_9,
  // Locomotion
  STAND_TOGGLE,
  ZERO_VELOCITY,
  WALK,
  STAND,
  // Whole-body tracking
  START_MOTION_CLIP,
  // Dual mode
  SWITCH_MODE,
};

// 0-based policy index for SWITCH_POLICY_N, or nullopt for other commands.
std::optional<int> switch_policy_index(StateCommand command);

const char* to_string(StateCommand command);

// Parses the Python enum member name (e.g. "START_MOTION_CLIP").
std::optional<StateCommand> state_command_from_string(const std::string& name);

// Absolute velocity command (VelCmd).
struct VelCmd {
  double lin_vel_x = 0.0;
  double lin_vel_y = 0.0;
  double ang_vel = 0.0;
};

}  // namespace holosoma
