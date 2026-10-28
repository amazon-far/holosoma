#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

#include "holosoma_cpp/api.hpp"
#include "holosoma_cpp/contextual_policy.hpp"

namespace {
using namespace holosoma;
const std::filesystem::path fixtures(HOLOSOMA_CONTEXTUAL_FIXTURE_DIR);

InferenceConfig config(const char* model = "contextual.onnx") {
  InferenceConfig c;
  c.robot.num_joints = c.robot.num_motors = 3;
  c.robot.dof_names = {"hip", "knee", "ankle"};
  c.robot.default_dof_angles = c.robot.default_motor_angles = {.1, .2, .3};
  c.robot.joint2motor = c.robot.motor2joint = {0, 1, 2};
  c.observation.obs_dict = {{"actor_obs", {}}};
  c.task.model_path = {(fixtures / model).string()};
  c.task.use_phase = false;
  c.contextual.emplace();
  c.contextual->depth_height = 2;
  c.contextual->depth_width = 3;
  // The fixture robot has no waist; gravity-frame tests set their own chain.
  c.contextual->imu_to_obs_joint_chain = {};
  c.runtime.intra_op_threads = 2;
  return c;
}
RobotState state() {
  RobotState s;
  s.dof_pos = {.1, .2, .3};
  s.dof_vel = {0, 0, 0};
  return s;
}
const contextual::Policy& core(Policy& policy) { return dynamic_cast<const ContextualPolicy&>(policy.policy()).core(); }

TEST(ContextualApi, DepthFrameSizingDoesNotOverflowSignedInt) {
  contextual::Spec spec;
  EXPECT_EQ(spec.depth_frame_size(), 5046u);
  // Exercise the overflow boundary without allocating a multi-gigabyte ONNX tensor.
  spec.height = spec.width = 50000;
  ASSERT_GT(2500000000ull, std::numeric_limits<int>::max());
  EXPECT_EQ(spec.depth_frame_size(), 2500000000ull);
}

TEST(ContextualApi, DepthFramesAreSizedAndValidatedThroughPublicStep) {
  Policy policy(config());
  ContextualInput input;
  policy.step(state(), input);
  EXPECT_EQ(core(policy).depth(), std::vector<float>(12, 0.0f));
  input.depth = std::vector<float>(6, .25f);
  policy.step(state(), input);
  EXPECT_EQ(core(policy).depth(), (std::vector<float>{0, 0, 0, 0, 0, 0, .25f, .25f, .25f, .25f, .25f, .25f}));
  for (size_t size : {0u, 5u, 7u}) {
    input.depth = std::vector<float>(size, .5f);
    EXPECT_THROW(policy.step(state(), input), std::invalid_argument);
  }
}

TEST(ContextualApi, SharedFactoryClipsBeforeScalingAndFeedbackInNamedJointOrder) {
  Policy policy(config());
  ASSERT_EQ(policy.policy().kind(), "ContextualPolicy");
  ContextualInput input;
  input.monotonic_time = 1;
  const auto first = policy.step(state(), input);
  EXPECT_DOUBLE_EQ(first.q[0], 15.1);
  EXPECT_DOUBLE_EQ(first.q[1], -9.8);
  EXPECT_DOUBLE_EQ(first.q[2], 40.3);
  EXPECT_EQ(first.kp, (std::vector<double>{10, 20, 30}));
  EXPECT_EQ(first.kd, (std::vector<double>{1, 2, 3}));
  EXPECT_EQ(core(policy).action(), (std::vector<float>{20, -20, 5}));
  input.monotonic_time = 2;
  const auto second = policy.step(state(), input);
  EXPECT_FLOAT_EQ(core(policy).observation().at(12), 20);
  EXPECT_FLOAT_EQ(core(policy).observation().at(13), -20);
  EXPECT_FLOAT_EQ(core(policy).observation().at(14), 5);
  EXPECT_DOUBLE_EQ(second.q[0], 60.1);
  policy.reset();
  EXPECT_DOUBLE_EQ(policy.step(state(), input).q[0], first.q[0]);
}

TEST(ContextualApi, ClimbSecondPassDoesNotAdvanceDepthHistoryTwice) {
  Policy policy(config());
  ContextualInput input;
  input.monotonic_time = 1;
  input.command_index = 1;
  input.depth = std::vector<float>(6, .2f);
  policy.step(state(), input);
  EXPECT_EQ(core(policy).calls(), 2);
  EXPECT_EQ(core(policy).effective_command(), 14);
  input.command_index.reset();
  input.depth = std::vector<float>(6, .8f);
  input.monotonic_time = 2;
  policy.step(state(), input);
  ASSERT_EQ(core(policy).depth().size(), 12u);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(core(policy).depth()[i], .2f);
    EXPECT_FLOAT_EQ(core(policy).depth()[i + 6], .8f);
  }
  input.command_index = 0;
  input.allow_climb_at_stand = false;
  input.allow_climb_continuation = false;
  input.dz = true;
  policy.step(state(), input);
  EXPECT_EQ(core(policy).effective_command(), 0);
  EXPECT_FALSE(core(policy).climbing());
}

TEST(ContextualApi, RequiresFreshTypedInputAndKeepsInstancesIndependent) {
  Policy first(config()), second(config());
  ContextualInput input;
  input.monotonic_time = 1;
  first.step(state(), input);
  EXPECT_THROW(first.step(state()), std::invalid_argument);
  EXPECT_TRUE(core(second).observation().empty());
  EXPECT_DOUBLE_EQ(second.step(state(), input).q[0], 15.1);
  input.monotonic_time = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(first.step(state(), input), std::invalid_argument);
}

TEST(ContextualApi, CommonVelocityApiReachesContextualCommandMapping) {
  Policy policy(config());
  policy.velocity({.5, 0, 0});
  policy.step(state(), ContextualInput{});
  EXPECT_EQ(core(policy).active_command(), 1);
}

TEST(ContextualApi, CommonZeroActionDebugOptionPreservesRawFeedback) {
  auto c = config();
  c.task.debug.force_zero_action = true;
  Policy policy(c);
  EXPECT_EQ(policy.step(state(), ContextualInput{}).q, c.robot.default_dof_angles);
  EXPECT_EQ(core(policy).action(), (std::vector<float>{20, -20, 5}));
}

TEST(ContextualApi, InactiveTicksCannotSupplyAnOldDepthFrameToLaterEngagement) {
  Policy policy(config(), false);
  ContextualInput input;
  input.depth = std::vector<float>(6, .8f);
  policy.step(state(), input);
  EXPECT_TRUE(core(policy).observation().empty());
  policy.command(StateCommand::START);
  EXPECT_THROW(policy.step(state()), std::invalid_argument);
  input.depth = std::vector<float>(6, .2f);
  EXPECT_NO_THROW(policy.step(state(), input));
}

TEST(ContextualApi, ValidatesGraphAndOptionalClimbHeadThroughPublicFactory) {
  EXPECT_THROW(Policy(config("wrong-width.onnx")), std::runtime_error);
  EXPECT_THROW(Policy(config("wrong-actions.onnx")), std::runtime_error);
  EXPECT_THROW(Policy(config("no-head.onnx")), std::runtime_error);
  auto c = config("no-head.onnx");
  c.contextual->require_climb_head = false;
  Policy policy(c);
  ContextualInput input;
  EXPECT_NO_THROW(policy.step(state(), input));
  input.dz = true;
  EXPECT_THROW(policy.step(state(), input), std::invalid_argument);
  Policy invalid(config("nonfinite.onnx"));
  EXPECT_THROW(invalid.step(state(), ContextualInput{}), std::invalid_argument);
}

TEST(ContextualApi, ConfiguredGainsAndSharedModelProfilingArePreserved) {
  auto c = config();
  c.robot.motor_kp = std::vector<double>{7, 8, 9};
  c.robot.motor_kd = std::vector<double>{.7, .8, .9};
  c.runtime.profile_prefix = (fixtures / "contextual-profile").string();
  Policy policy(c);
  EXPECT_EQ(policy.step(state(), ContextualInput{}).kp, *c.robot.motor_kp);
  const auto profile = dynamic_cast<ContextualPolicy&>(policy.policy()).core().FinishProfile();
  const auto events = nlohmann::json::parse(std::ifstream(profile));
  size_t matrix_calls = 0;
  for (const auto& event : events) {
    if (!event.contains("args")) continue;
    const auto& args = event["args"];
    const auto op = args.value("op_name", "");
    if ((op == "MatMul" || op == "Gemm") && args.value("provider", "") == "CPUExecutionProvider") ++matrix_calls;
  }
  EXPECT_GT(matrix_calls, 0u);
}

TEST(ContextualApi, UnusableOrientationFaultsInsteadOfReadingUpright) {
  Policy policy(config());
  auto s = state();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& quat : std::vector<std::vector<double>>{{0, 0, 0, 0}, {nan, 0, 0, 0}, {1, 0, nan, 0}}) {
    s.base_quat_wxyz = quat;
    EXPECT_THROW(policy.step(s, ContextualInput{}), std::invalid_argument);
  }
  // The kernel itself refuses too, for callers that drive it without the public state check.
  contextual::State raw;
  raw.q = {.1, .2, .3};
  raw.dq = {0, 0, 0};
  raw.quat = {0, 0, 0, 0};
  EXPECT_THROW(core(policy).Observation(raw), std::invalid_argument);
  raw.quat = {2, 0, 0, 0};  // Unnormalised but usable.
  EXPECT_NO_THROW(core(policy).Observation(raw));
  // A valid orientation still steps, and other fields keep their checks.
  s.base_quat_wxyz = {1, 0, 0, 0};
  EXPECT_DOUBLE_EQ(policy.step(s, ContextualInput{}).q[0], 15.1);
  s.dof_vel[0] = nan;
  EXPECT_THROW(policy.step(s, ContextualInput{}), std::invalid_argument);
}

// Independent reference: gravity (0, 0, -1) expressed in the frame of a unit quaternion,
// through the rotation matrix rather than the kernel's quaternion sandwich.
std::array<double, 3> gravity_in(const std::array<double, 4>& q) {
  const auto [w, x, y, z] = q;
  // Third row of R(q), negated: R^T (0, 0, -1).
  return {-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y))};
}
std::array<double, 4> mul(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

TEST(ContextualApi, OrientationAndCorrectedGravityReachTheModelInOneFrame) {
  auto c = config();
  c.contextual->imu_to_obs_joint_chain = {{"hip", "z"}, {"knee", "x"}, {"ankle", "y"}};
  Policy policy(c);
  auto s = state();
  s.dof_pos = {.4, -.3, .25};
  const double n = std::sqrt(.9 * .9 + .2 * .2 + .3 * .3 + .1 * .1);
  const std::array<double, 4> base{.9 / n, .2 / n, -.3 / n, .1 / n};
  s.base_quat_wxyz.assign(base.begin(), base.end());
  // Expected: the IMU orientation composed with the chain, in configured order.
  auto torso = base;
  for (const auto& [angle, axis] : std::vector<std::pair<double, int>>{{.4, 3}, {-.3, 1}, {.25, 2}}) {
    std::array<double, 4> r{std::cos(angle / 2), 0, 0, 0};
    r[axis] = std::sin(angle / 2);
    torso = mul(torso, r);
  }
  const auto expected = gravity_in(torso);
  const auto imu = gravity_in(base);
  // Anti-vacuity: the chain must actually move gravity at these angles.
  ASSERT_GT(std::abs(expected[0] - imu[0]) + std::abs(expected[1] - imu[1]) + std::abs(expected[2] - imu[2]), .1);

  policy.step(s, ContextualInput{});
  const auto from_quat = core(policy).observation();
  // The interface's corrected gravity is IMU-frame; it must take the same rotation.
  s.projected_gravity = std::vector<double>(imu.begin(), imu.end());
  s.base_quat_wxyz = {1, 0, 0, 0};  // Proves the supplied vector, not the quaternion, is used.
  policy.reset();
  policy.step(s, ContextualInput{});
  const auto from_supplied = core(policy).observation();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(from_quat[i], expected[i], 1e-6) << i;
    EXPECT_NEAR(from_supplied[i], expected[i], 1e-6) << i;
  }

  // An empty chain keeps IMU-frame gravity for both sources.
  c.contextual->imu_to_obs_joint_chain = {};
  Policy imu_frame(c);
  imu_frame.step(s, ContextualInput{});
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(core(imu_frame).observation()[i], imu[i], 1e-6) << i;
}

TEST(ContextualApi, ImuToObsJointChainMustNameRuntimeJoints) {
  auto c = config();
  c.contextual = ContextualConfig{};  // Default G1 waist chain on a robot without a waist.
  c.contextual->depth_height = 2;
  c.contextual->depth_width = 3;
  EXPECT_THROW(Policy{c}, std::invalid_argument);
  c.contextual->imu_to_obs_joint_chain = {{"hip", "w"}};
  EXPECT_THROW(Policy{c}, std::invalid_argument);
  c.contextual->imu_to_obs_joint_chain = {{"hip", "z"}};
  EXPECT_NO_THROW(Policy{c});
}

}  // namespace
