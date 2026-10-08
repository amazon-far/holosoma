// Quaternion and rotation helpers.
//
// Every function mirrors holosoma_inference/utils/math/quat.py, including the
// arithmetic order, so observations match the Python runtime to float rounding.
// Quaternions are stored as {w, x, y, z} unless the name says otherwise.
#pragma once

#include <array>
#include <cmath>

namespace holosoma::math {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;  // w, x, y, z
using Mat3 = std::array<double, 9>;  // row-major

inline constexpr double kPi = 3.14159265358979323846;

inline Quat identity_quat() { return {1.0, 0.0, 0.0, 0.0}; }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline Quat xyzw_to_wxyz(const std::array<double, 4>& xyzw) { return {xyzw[3], xyzw[0], xyzw[1], xyzw[2]}; }

inline std::array<double, 4> wxyz_to_xyzw(const Quat& q) { return {q[1], q[2], q[3], q[0]}; }

// Rotate v by the inverse of q (quat_rotate_inverse).
inline Vec3 quat_rotate_inverse(const Quat& q, const Vec3& v) {
  const double w = q[0];
  const Vec3 q_vec{q[1], q[2], q[3]};
  const double a_scale = 2.0 * w * w - 1.0;
  const Vec3 c_cross = cross(q_vec, v);
  const double dot = q_vec[0] * v[0] + q_vec[1] * v[1] + q_vec[2] * v[2];
  Vec3 out{};
  for (int i = 0; i < 3; ++i) {
    const double a = v[i] * a_scale;
    const double b = c_cross[i] * w * 2.0;
    const double c = q_vec[i] * dot * 2.0;
    out[i] = a - b + c;
  }
  return out;
}

// Roll/pitch/yaw (ZYX order) to quaternion (rpy_to_quat).
inline Quat rpy_to_quat(double roll, double pitch, double yaw) {
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  return {cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
          cr * cp * sy - sr * sp * cy};
}

// Quaternion to roll/pitch/yaw (quat_to_rpy).
inline Vec3 quat_to_rpy(const Quat& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const double sinp = 2 * (w * y - z * x);
  const double pitch = std::abs(sinp) >= 1 ? std::copysign(kPi / 2, sinp) : std::asin(sinp);
  const double yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return {roll, pitch, yaw};
}

inline Quat quat_inverse(const Quat& q) { return {q[0], -q[1], -q[2], -q[3]}; }

// Hamilton product a * b (quat_mul, same factorization as the Python code).
inline Quat quat_mul(const Quat& a, const Quat& b) {
  const double w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
  const double w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
  const double ww = (z1 + x1) * (x2 + y2);
  const double yy = (w1 - y1) * (w2 + z2);
  const double zz = (w1 + y1) * (w2 - z2);
  const double xx = ww + yy + zz;
  const double qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));
  return {qq - ww + (z1 - y1) * (y2 - z2), qq - xx + (x1 + w1) * (x2 + w2), qq - yy + (w1 - x1) * (y2 + z2),
          qq - zz + (z1 + y1) * (w2 - x2)};
}

// Orientation of frame 2 expressed in frame 1 (subtract_frame_transforms).
inline Quat subtract_frame_transforms(const Quat& q01, const Quat& q02) { return quat_mul(quat_inverse(q01), q02); }

// Rotation matrix of a (possibly unnormalized) quaternion (matrix_from_quat).
inline Mat3 matrix_from_quat(const Quat& q) {
  const double r = q[0], i = q[1], j = q[2], k = q[3];
  const double two_s = 2.0 / (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {1 - two_s * (j * j + k * k), two_s * (i * j - k * r),     two_s * (i * k + j * r),
          two_s * (i * j + k * r),     1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
          two_s * (i * k - j * r),     two_s * (j * k + i * r),     1 - two_s * (i * i + j * j)};
}

inline double quat_yaw(const Quat& q) { return quat_to_rpy(q)[2]; }

// Remove a yaw offset by left-multiplying with yaw(-offset) (WholeBodyTrackingPolicy._remove_yaw_offset).
inline Quat remove_yaw_offset(const Quat& q, double yaw_offset) {
  if (std::abs(yaw_offset) < 1e-6) {
    return q;
  }
  return quat_mul(rpy_to_quat(0.0, 0.0, -yaw_offset), q);
}

inline Mat3 mat_mul(const Mat3& a, const Mat3& b) {
  Mat3 out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
    }
  }
  return out;
}

inline Vec3 mat_vec(const Mat3& m, const Vec3& v) {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

inline Mat3 identity_mat() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

// URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll).
inline Mat3 rpy_to_matrix(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  return {cy * cp,
          cy * sp * sr - sy * cr,
          cy * sp * cr + sy * sr,
          sy * cp,
          sy * sp * sr + cy * cr,
          sy * sp * cr - cy * sr,
          -sp,
          cp * sr,
          cp * cr};
}

// Rotation of `angle` about a unit `axis` (Rodrigues).
inline Mat3 axis_angle_to_matrix(const Vec3& axis, double angle) {
  const double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
  const double x = axis[0], y = axis[1], z = axis[2];
  return {t * x * x + c,     t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c,
          t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c};
}

// Unit quaternion of a rotation matrix (Shepperd's method), w >= 0.
inline Quat matrix_to_quat(const Mat3& m) {
  const double trace = m[0] + m[4] + m[8];
  Quat q{};
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q = {0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s};
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
    q = {(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s};
  } else if (m[4] > m[8]) {
    const double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
    q = {(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s};
  } else {
    const double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
    q = {(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s};
  }
  if (q[0] < 0.0) {
    q = {-q[0], -q[1], -q[2], -q[3]};
  }
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

}  // namespace holosoma::math
