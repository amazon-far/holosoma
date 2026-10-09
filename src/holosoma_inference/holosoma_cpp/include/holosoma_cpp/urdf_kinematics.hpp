// Forward kinematics of one URDF link on a floating base.
//
// Replaces the Pinocchio model the Python WBT policy builds from the
// `robot_urdf` ONNX metadata: the root link gets the base orientation and every
// movable joint on the chain to the target link is driven by name from the
// robot's joint vector.
#pragma once

#include <string>
#include <vector>

#include "holosoma_cpp/math.hpp"

namespace holosoma {

class UrdfChain {
 public:
  // Parses `urdf_xml` and extracts the chain from the root link to `target_link`.
  // `dof_names` is the robot joint order used by compute(); every movable URDF joint
  // must appear in it and the counts must match (as the Python runtime asserts).
  UrdfChain(const std::string& urdf_xml, const std::string& target_link, const std::vector<std::string>& dof_names);

  // World orientation (w, x, y, z) of the target link.
  math::Quat orientation(const math::Quat& base_quat_wxyz, const std::vector<double>& dof_pos) const;

  // World position of the target link.
  math::Vec3 position(const math::Vec3& base_pos, const math::Quat& base_quat_wxyz,
                      const std::vector<double>& dof_pos) const;

  const std::string& root_link() const { return root_link_; }
  size_t chain_length() const { return chain_.size(); }

 private:
  enum class JointType { kFixed, kRevolute, kPrismatic };

  struct ChainJoint {
    std::string name;
    JointType type = JointType::kFixed;
    math::Mat3 origin_rot = math::identity_mat();
    math::Vec3 origin_xyz{0.0, 0.0, 0.0};
    math::Vec3 axis{1.0, 0.0, 0.0};
    int dof_index = -1;  // index into dof_pos, -1 for fixed joints
  };

  void pose(const math::Quat& base_quat_wxyz, const std::vector<double>& dof_pos, math::Mat3& rot,
            math::Vec3& pos) const;

  std::string root_link_;
  std::vector<ChainJoint> chain_;  // root -> target
};

}  // namespace holosoma
