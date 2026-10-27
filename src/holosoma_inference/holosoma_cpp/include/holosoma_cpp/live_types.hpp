#pragma once

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <optional>
#include <vector>

namespace holosoma::live {

inline constexpr int kActionDim = 29;
inline constexpr std::array<const char*, kActionDim> kG1BodyJointNames = {
    {"left_hip_pitch_joint",     "left_hip_roll_joint",     "left_hip_yaw_joint",         "left_knee_joint",
     "left_ankle_pitch_joint",   "left_ankle_roll_joint",   "right_hip_pitch_joint",      "right_hip_roll_joint",
     "right_hip_yaw_joint",      "right_knee_joint",        "right_ankle_pitch_joint",    "right_ankle_roll_joint",
     "waist_yaw_joint",          "waist_roll_joint",        "waist_pitch_joint",          "left_shoulder_pitch_joint",
     "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",           "left_wrist_roll_joint",
     "left_wrist_pitch_joint",   "left_wrist_yaw_joint",    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
     "right_shoulder_yaw_joint", "right_elbow_joint",       "right_wrist_roll_joint",     "right_wrist_pitch_joint",
     "right_wrist_yaw_joint"}};

// Canonical joint order; orientations are wxyz and angular velocity is body-local.
struct RobotObservation {
  std::array<float, 4> root_quat_wxyz{{1, 0, 0, 0}};
  std::array<float, kActionDim> dof_pos_mj{};
  std::array<float, kActionDim> dof_vel_mj{};
  std::array<float, 3> base_ang_vel{};
  // Gravity in the base frame when the host supplies a corrected one; otherwise
  // it is derived from root_quat_wxyz.
  std::optional<std::array<float, 3>> projected_gravity;
};
struct WbtReferenceFrame {
  std::array<float, kActionDim> dof_pos{};
  std::array<float, kActionDim> dof_vel{};
  std::array<float, 4> root_quat_wxyz{{1, 0, 0, 0}};
};
struct WbtReferenceSequence {
  WbtReferenceFrame current;
  std::vector<WbtReferenceFrame> lookahead;
};

namespace math {
using Quatf = Eigen::Quaternionf;
inline Quatf quat_from_wxyz(const float* q) { return {q[0], q[1], q[2], q[3]}; }
inline Eigen::Vector3f projected_gravity(const Quatf& q) { return q.conjugate() * Eigen::Vector3f(0, 0, -1); }
}  // namespace math

inline std::array<float, 4> SanitizeQuatWxyz(const float* q) {
  const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!std::isfinite(norm) || norm < 1e-6f) return {1, 0, 0, 0};
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}
inline math::Quatf HeadingQuat(const math::Quatf& q) {
  const auto forward = q * Eigen::Vector3f::UnitX();
  const float yaw = std::atan2(forward.y(), forward.x());
  return math::Quatf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
}
inline void QuatToRot6d(const math::Quatf& q, float* out) {
  const auto r = q.toRotationMatrix();
  for (int row = 0; row < 3; ++row) {
    out[2 * row] = r(row, 0);
    out[2 * row + 1] = r(row, 1);
  }
}
}  // namespace holosoma::live
