#include "holosoma_cpp/urdf_kinematics.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace holosoma {
namespace {

// Element name without an XML namespace prefix.
const char* local_name(const tinyxml2::XMLElement* e) {
  const char* name = e->Name();
  const char* colon = std::strrchr(name, ':');
  return colon ? colon + 1 : name;
}

const tinyxml2::XMLElement* child(const tinyxml2::XMLElement* e, const char* name) {
  for (auto* c = e->FirstChildElement(); c != nullptr; c = c->NextSiblingElement()) {
    if (std::strcmp(local_name(c), name) == 0) return c;
  }
  return nullptr;
}

math::Vec3 parse_vec3(const char* text, const math::Vec3& fallback, const std::string& where) {
  if (text == nullptr) return fallback;
  std::istringstream is(text);
  math::Vec3 v{};
  if (!(is >> v[0] >> v[1] >> v[2])) {
    throw std::runtime_error("URDF: cannot parse three numbers from '" + std::string(text) + "' in " + where);
  }
  return v;
}

struct UrdfJoint {
  std::string name;
  std::string type;
  std::string parent;
  std::string child;
  math::Vec3 xyz{0.0, 0.0, 0.0};
  math::Vec3 rpy{0.0, 0.0, 0.0};
  math::Vec3 axis{1.0, 0.0, 0.0};
};

}  // namespace

UrdfChain::UrdfChain(const std::string& urdf_xml, const std::string& target_link,
                     const std::vector<std::string>& dof_names) {
  tinyxml2::XMLDocument doc;
  if (doc.Parse(urdf_xml.c_str(), urdf_xml.size()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(std::string("URDF: XML parse error: ") + doc.ErrorStr());
  }
  const tinyxml2::XMLElement* robot = doc.RootElement();
  if (robot == nullptr || std::strcmp(local_name(robot), "robot") != 0) {
    throw std::runtime_error("URDF: root element must be <robot>");
  }

  std::set<std::string> links;
  std::map<std::string, UrdfJoint> joint_by_child;
  std::vector<std::string> movable;
  for (auto* e = robot->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    const char* tag = local_name(e);
    if (std::strcmp(tag, "link") == 0) {
      const char* name = e->Attribute("name");
      if (name == nullptr) throw std::runtime_error("URDF: <link> without a name");
      links.insert(name);
    } else if (std::strcmp(tag, "joint") == 0) {
      UrdfJoint j;
      const char* name = e->Attribute("name");
      const char* type = e->Attribute("type");
      const auto* parent = child(e, "parent");
      const auto* child_el = child(e, "child");
      if (!name || !type || !parent || !child_el || !parent->Attribute("link") || !child_el->Attribute("link")) {
        throw std::runtime_error("URDF: malformed <joint> element");
      }
      j.name = name;
      j.type = type;
      j.parent = parent->Attribute("link");
      j.child = child_el->Attribute("link");
      if (const auto* origin = child(e, "origin")) {
        j.xyz = parse_vec3(origin->Attribute("xyz"), j.xyz, j.name + " origin");
        j.rpy = parse_vec3(origin->Attribute("rpy"), j.rpy, j.name + " origin");
      }
      if (const auto* axis = child(e, "axis")) {
        j.axis = parse_vec3(axis->Attribute("xyz"), j.axis, j.name + " axis");
      }
      if (j.type == "revolute" || j.type == "continuous" || j.type == "prismatic") {
        movable.push_back(j.name);
      } else if (j.type != "fixed") {
        throw std::runtime_error("URDF: joint '" + j.name + "' has unsupported type '" + j.type + "'");
      }
      if (!joint_by_child.emplace(j.child, j).second) {
        throw std::runtime_error("URDF: link '" + j.child + "' has more than one parent joint");
      }
    }
  }

  std::vector<std::string> roots;
  for (const auto& link : links) {
    if (joint_by_child.count(link) == 0) roots.push_back(link);
  }
  if (roots.size() != 1) {
    throw std::runtime_error("URDF: expected exactly one root link, found " + std::to_string(roots.size()));
  }
  root_link_ = roots.front();
  if (links.count(target_link) == 0) {
    throw std::runtime_error("URDF: reference body '" + target_link + "' is not a link of the robot");
  }

  // Same check as PinocchioRobot: the movable joints are exactly the robot's joints.
  if (movable.size() != dof_names.size()) {
    throw std::runtime_error("URDF: the model has " + std::to_string(movable.size()) +
                             " movable joints but the robot config lists " + std::to_string(dof_names.size()));
  }
  for (const auto& name : movable) {
    if (std::find(dof_names.begin(), dof_names.end(), name) == dof_names.end()) {
      throw std::runtime_error("URDF: joint '" + name + "' is not in robot.dof_names");
    }
  }

  std::vector<UrdfJoint> reversed;
  std::string link = target_link;
  while (link != root_link_) {
    const auto it = joint_by_child.find(link);
    if (it == joint_by_child.end() || reversed.size() > links.size()) {
      throw std::runtime_error("URDF: link '" + target_link + "' is not connected to the root");
    }
    reversed.push_back(it->second);
    link = it->second.parent;
  }
  for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
    ChainJoint j;
    j.name = it->name;
    j.origin_rot = math::rpy_to_matrix(it->rpy[0], it->rpy[1], it->rpy[2]);
    j.origin_xyz = it->xyz;
    if (it->type == "fixed") {
      j.type = JointType::kFixed;
    } else {
      j.type = it->type == "prismatic" ? JointType::kPrismatic : JointType::kRevolute;
      const double n = std::sqrt(it->axis[0] * it->axis[0] + it->axis[1] * it->axis[1] + it->axis[2] * it->axis[2]);
      if (!(n > 0.0)) throw std::runtime_error("URDF: joint '" + it->name + "' has a zero axis");
      j.axis = {it->axis[0] / n, it->axis[1] / n, it->axis[2] / n};
      j.dof_index = static_cast<int>(std::find(dof_names.begin(), dof_names.end(), it->name) - dof_names.begin());
    }
    chain_.push_back(j);
  }
}

void UrdfChain::pose(const math::Quat& base_quat_wxyz, const std::vector<double>& dof_pos, math::Mat3& rot,
                     math::Vec3& pos) const {
  rot = math::matrix_from_quat(base_quat_wxyz);
  for (const auto& j : chain_) {
    const math::Vec3 offset = math::mat_vec(rot, j.origin_xyz);
    for (int i = 0; i < 3; ++i) pos[i] += offset[i];
    rot = math::mat_mul(rot, j.origin_rot);
    if (j.type == JointType::kRevolute) {
      rot = math::mat_mul(rot, math::axis_angle_to_matrix(j.axis, dof_pos.at(static_cast<size_t>(j.dof_index))));
    } else if (j.type == JointType::kPrismatic) {
      const double d = dof_pos.at(static_cast<size_t>(j.dof_index));
      const math::Vec3 slide = math::mat_vec(rot, j.axis);
      for (int i = 0; i < 3; ++i) pos[i] += slide[i] * d;
    }
  }
}

math::Quat UrdfChain::orientation(const math::Quat& base_quat_wxyz, const std::vector<double>& dof_pos) const {
  math::Mat3 rot{};
  math::Vec3 pos{0.0, 0.0, 0.0};
  pose(base_quat_wxyz, dof_pos, rot, pos);
  return math::matrix_to_quat(rot);
}

math::Vec3 UrdfChain::position(const math::Vec3& base_pos, const math::Quat& base_quat_wxyz,
                               const std::vector<double>& dof_pos) const {
  math::Mat3 rot{};
  math::Vec3 pos = base_pos;
  pose(base_quat_wxyz, dof_pos, rot, pos);
  return pos;
}

}  // namespace holosoma
