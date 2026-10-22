#include <gtest/gtest.h>

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

TEST(ContextualApi, PreservesContextualIdentityFallbackWithoutWeakeningOtherStateChecks) {
  Policy policy(config());
  auto s = state();
  s.base_quat_wxyz = {0, 0, 0, 0};
  EXPECT_DOUBLE_EQ(policy.step(s, ContextualInput{}).q[0], 15.1);
  s.dof_vel[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(policy.step(s, ContextualInput{}), std::invalid_argument);
}
}  // namespace
