// Unitree SDK2 (DDS) backend for HG-message robots such as the G1.
//
// Matches the far-unitree-sdk `unitree_interface` binding used by the Python
// runtime: LowState on rt/lowstate (CRC checked), the wireless controller on
// rt/wirelesscontroller, and a 500 Hz writer thread that republishes the latest
// LowCmd (PR mode, mode_machine echoed from the robot, CRC filled in).
#pragma once

#include <memory>
#include <string>

#include "holosoma_cpp/config.hpp"
#include "holosoma_cpp/robot_interface.hpp"

namespace holosoma {

// Returns the first wired, operationally-up NIC (utils/network.py), or "lo".
std::string detect_robot_interface();

// Creates the SDK2 backend. Throws if the binary was built without Unitree SDK2.
std::unique_ptr<RobotInterface> make_unitree_interface(const RobotConfig& robot, int domain_id,
                                                       const std::string& network_interface);

bool unitree_sdk_available();

}  // namespace holosoma
