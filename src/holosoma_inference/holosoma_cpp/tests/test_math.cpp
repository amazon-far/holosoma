// Quaternion helpers and URDF forward kinematics against holosoma_inference (numpy / Pinocchio).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "holosoma_cpp/config.hpp"
#include "holosoma_cpp/math.hpp"
#include "holosoma_cpp/onnx_model.hpp"
#include "holosoma_cpp/urdf_kinematics.hpp"

namespace holosoma {
namespace {

nlohmann::json load_golden(const std::string& name) {
  std::ifstream f(std::string(HOLOSOMA_TEST_GOLDEN_DIR) + "/" + name);
  if (!f) throw std::runtime_error("missing golden file " + name + "; run tests/gen_golden.py");
  return nlohmann::json::parse(f);
}

template <size_t N>
std::array<double, N> arr(const nlohmann::json& j) {
  std::array<double, N> a{};
  for (size_t i = 0; i < N; ++i) a[i] = j.at(i).get<double>();
  return a;
}

template <size_t N>
void expect_near(const std::array<double, N>& actual, const nlohmann::json& expected, double tol) {
  for (size_t i = 0; i < N; ++i) EXPECT_NEAR(actual[i], expected.at(i).get<double>(), tol) << "index " << i;
}

TEST(Math, MatchesPythonQuatHelpers) {
  const auto golden = load_golden("math.json");
  for (const auto& s : golden["samples"]) {
    const auto a = arr<4>(s["a"]);
    const auto b = arr<4>(s["b"]);
    const auto v = arr<3>(s["v"]);
    const auto rpy = arr<3>(s["rpy"]);
    expect_near(math::quat_mul(a, b), s["quat_mul"], 1e-12);
    expect_near(math::quat_rotate_inverse(a, v), s["quat_rotate_inverse"], 1e-12);
    expect_near(math::matrix_from_quat(b), s["matrix_from_quat"], 1e-12);
    expect_near(math::rpy_to_quat(rpy[0], rpy[1], rpy[2]), s["rpy_to_quat"], 1e-12);
    expect_near(math::quat_to_rpy(a), s["quat_to_rpy"], 1e-12);
    expect_near(math::subtract_frame_transforms(a, b), s["subtract_frame_transforms"], 1e-12);
  }
}

TEST(Kinematics, TorsoOrientationMatchesPinocchio) {
  const auto golden = load_golden("kinematics.json");
  OnnxModel model(std::string(HOLOSOMA_TEST_REPO_ROOT) + "/" + golden["model"].get<std::string>());
  const std::string urdf = nlohmann::json::parse(*model.metadata_value("robot_urdf")).get<std::string>();
  const std::vector<std::string> dof_names = nlohmann::json::parse(*model.metadata_value("dof_names"));
  const UrdfChain chain(urdf, golden["body"].get<std::string>(), dof_names);
  EXPECT_EQ(chain.root_link(), "pelvis");

  for (const auto& s : golden["samples"]) {
    const math::Quat base = arr<4>(s["base_quat_wxyz"]);
    const auto q = s["dof_pos"].get<std::vector<double>>();
    const math::Quat torso = chain.orientation(base, q);
    const math::Quat expected = math::xyzw_to_wxyz(arr<4>(s["torso_quat_xyzw"]));
    // q and -q are the same rotation.
    const double dot =
        torso[0] * expected[0] + torso[1] * expected[1] + torso[2] * expected[2] + torso[3] * expected[3];
    EXPECT_NEAR(std::abs(dot), 1.0, 1e-9);
  }
}

TEST(Kinematics, RejectsUrdfThatDoesNotMatchTheRobot) {
  const std::string urdf = R"(<robot name="r"><link name="base"/><link name="a"/>
    <joint name="j0" type="revolute"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/></joint></robot>)";
  EXPECT_NO_THROW(UrdfChain(urdf, "a", {"j0"}));
  EXPECT_THROW(UrdfChain(urdf, "a", {"other"}), std::runtime_error);
  EXPECT_THROW(UrdfChain(urdf, "a", {"j0", "j1"}), std::runtime_error);
  EXPECT_THROW(UrdfChain(urdf, "missing_link", {"j0"}), std::runtime_error);
}

}  // namespace
}  // namespace holosoma
