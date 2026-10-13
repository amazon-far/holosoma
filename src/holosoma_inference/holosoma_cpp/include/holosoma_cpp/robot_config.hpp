#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace holosoma {

// Resolve the saved robot defaults in metadata joint order without modifying
// the config. Accepts both default_joint_positions and legacy
// default_joint_angles; throws on invalid tags, overlapping coverage,
// conflicting field names or joint orders, and non-finite values. Unspecified
// current positions default to zero; legacy positions require full coverage.
std::vector<float> ResolveDefaultDofPositions(const nlohmann::json &robot,
                                              const std::vector<std::string> &metadata_dof_names);

}  // namespace holosoma
