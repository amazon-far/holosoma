#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace holosoma {
inline constexpr float kPolicyActionCeiling = 20.0f;

// Bound raw actions before scaling AND before previous-action feedback. A
// checkpoint may tighten the bound, but cannot relax the runtime ceiling.
inline float clip_policy_action(float raw, float limit = kPolicyActionCeiling) {
  if (!std::isfinite(raw) || !std::isfinite(limit) || limit <= 0)
    throw std::invalid_argument("Policy action and positive clip limit must be finite");
  const float bound = std::min(limit, kPolicyActionCeiling);
  return std::clamp(raw, -bound, bound);
}
}  // namespace holosoma
