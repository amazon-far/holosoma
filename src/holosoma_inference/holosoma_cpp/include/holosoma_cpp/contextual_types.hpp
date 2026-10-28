#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace holosoma {
// Presence of this section selects the depth-conditioned contextual policy.
// Robot poses/gains remain in RobotConfig; execution providers in RuntimeConfig.
struct ContextualConfig {
  int command_dim = 15;
  int depth_height = 58;
  int depth_width = 87;
  bool require_climb_head = true;
  bool force_manual_climbing = false;
  std::string climb_enter = "dz_or_pred";
  std::string climb_exit = "both_off";
  std::string discretizer = "simple";
  double yaw_request_gain = 2.0;
  double switch_margin = 0.0;
  // Measured (forward m/s, yaw rad/s) for each named gait, supplied by the caller.
  std::map<std::string, std::vector<double>> velocity_primitives;
  double forward_distance_scale = 0.5;
  double yaw_distance_scale = 1.2;
  std::optional<double> in_place_yaw_threshold;
  // The joint chain that carries the IMU frame to the observation frame, the
  // body frame the model was trained to read projected gravity in. Entries are
  // (joint name, rotation axis "x"/"y"/"z"), in kinematic order starting at the
  // IMU's link. Each tick the IMU-frame gravity is rotated through each joint's
  // current angle in turn. This is a chain rather than a fixed transform
  // because the rotation depends on those joint angles. Each joint is treated as
  // a pure rotation about its axis, with no fixed offset rotation between links
  // (true of the G1 waist joints, whose URDF origins all have rpy = 0).
  // The default is the G1 waist: pelvis IMU to torso-frame gravity. Empty means
  // the observation frame is the IMU frame. Every listed joint must be a
  // runtime joint.
  std::vector<std::pair<std::string, std::string>> imu_to_obs_joint_chain{
      {"waist_yaw_joint", "z"}, {"waist_roll_joint", "x"}, {"waist_pitch_joint", "y"}};
  // Transport-owned (the Python runner enforces it); carried for config parity.
  double max_depth_age_s = 0.25;
};

struct ContextualInput {
  // No depth inserts a zero frame. Freshness is the transport's decision: the
  // Python runner stops the policy once its newest frame is older than
  // ContextualConfig::max_depth_age_s.
  std::optional<std::vector<float>> depth;
  double monotonic_time = 0.0;
  // Optional fields are command events: absence preserves current state.
  std::optional<std::array<double, 3>> velocity;
  std::optional<int> command_index;
  std::optional<int> speed;
  std::optional<int> fixed_arm;
  std::optional<bool> dz;
  std::optional<bool> allow_climb_at_stand;
  std::optional<bool> allow_climb_continuation;
  std::optional<std::string> key;
};
}  // namespace holosoma
