#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "holosoma_cpp/api.hpp"
#include "holosoma_cpp/holosoma_inference_core.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace holosoma {
namespace {
std::string model(const char* name) { return std::string(HOLOSOMA_LIVE_FIXTURE_DIR) + "/" + name; }

TEST(PolicyApi, MetadataDrivenTrackingPreservesObservationAndActionParity) {
  const auto path = model("sonic_flat_0p2s.onnx");
  Policy policy(std::string(HOLOSOMA_TEST_PRESET_DIR) + "/g1-29dof-general-wbt.yaml", path);
  live::WbtPolicy baseline;
  ASSERT_TRUE(baseline.Initialize(path));
  RobotState state;
  state.dof_pos.assign(29, 0.0);
  state.dof_vel.assign(29, 0.0);
  ReferenceWindow window;
  window.q.assign(5 * 29, 0.0);
  window.dq.assign(5 * 29, 0.0);
  window.quat.assign(5, {1, 0, 0, 0});
  policy.set_reference(&window);
  for (int tick = 0; tick < 12; ++tick) {
    live::RobotObservation obs;
    live::WbtReferenceSequence reference;
    reference.lookahead.resize(5);
    for (int j = 0; j < 29; ++j) {
      state.dof_pos[j] = obs.dof_pos_mj[j] = 0.01f * (tick + j);
      for (int slot = 0; slot < 5; ++slot) {
        window.q[slot * 29 + j] = reference.lookahead[slot].dof_pos[j] = 0.02f * (tick + j + slot);
      }
    }
    reference.current = reference.lookahead.front();
    std::array<float, 29> offsets{};
    ASSERT_TRUE(baseline.ComputeAction(obs, reference, offsets.data()));
    const auto& command = policy.step(state);
    ASSERT_EQ(command.q.size(), 29u);
    ASSERT_FALSE(baseline.LastObservation().empty());
    EXPECT_EQ(policy.policy().last_observation(), baseline.LastObservation());
    for (int j = 0; j < 29; ++j) {
      EXPECT_NEAR(command.q[j], offsets[j] + baseline.DefaultDofPos()[j], 1e-6);
      EXPECT_NEAR(command.kp[j], baseline.Kp()[j], 1e-6);
      EXPECT_NEAR(command.kd[j], baseline.Kd()[j], 1e-6);
    }
  }
}

RobotState robot_state() {
  RobotState state;
  state.dof_pos.assign(29, 0.0);
  state.dof_vel.assign(29, 0.0);
  return state;
}

ReferenceWindow reference_window(int slots) {
  ReferenceWindow window;
  window.q.assign(static_cast<size_t>(std::max(1, slots) * 29), 0.0);
  window.dq = window.q;
  window.quat.assign(static_cast<size_t>(std::max(1, slots)), {1, 0, 0, 0});
  return window;
}

TEST(PolicyApi, RejectsInvalidWindowsAndCanRecoverWithoutReloading) {
  Policy policy(live_policy_config(model("sonic_flat_0p2s.onnx")));
  auto state = robot_state();
  EXPECT_THROW(policy.step(state), std::runtime_error);
  auto window = reference_window(5);
  policy.set_reference(&window);
  window.quat.clear();
  EXPECT_THROW(policy.step(state), std::invalid_argument);
  window = reference_window(5);
  window.q.pop_back();
  EXPECT_THROW(policy.step(state), std::invalid_argument);
  window = reference_window(5);
  window.dq[2] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(policy.step(state), std::invalid_argument);
  window = reference_window(5);
  window.quat[0] = {0, 0, 0, 0};
  EXPECT_THROW(policy.step(state), std::invalid_argument);
  window = reference_window(5);
  state.dof_pos.clear();
  EXPECT_THROW(policy.step(state), std::invalid_argument);
  state = robot_state();
  EXPECT_EQ(policy.step(state).q.size(), 29u);
  policy.set_reference(nullptr);
  EXPECT_THROW(policy.step(state), std::runtime_error);
}

TEST(PolicyApi, ExportedHistoryAndClipReachJointCommands) {
  Policy clipped(live_policy_config(model("action_50_clip_20.onnx")));
  auto window = reference_window(1);
  clipped.set_reference(&window);
  const auto& command = clipped.step(robot_state());
  for (int joint = 0; joint < 29; ++joint) EXPECT_NEAR(command.q[joint], 0.01 * joint + (joint % 2 ? -5.0 : 5.0), 1e-6);

  Policy repeated(live_policy_config(model("history_repeat_first.onnx")));
  window = reference_window(10);
  window.q.assign(290, 1.0);
  repeated.set_reference(&window);
  EXPECT_NEAR(repeated.step(robot_state()).q[0], 1.0, 1e-6);
}

TEST(PolicyApi, HistoryInitializationReachesTheSharedCore) {
  for (const auto& [setting, mode, padded] :
       {std::make_tuple("repeat_first_valid_obs", live::WbtHistoryInitialization::kRepeatFirstValidObs, 1.0f),
        std::make_tuple("zero_pad", live::WbtHistoryInitialization::kZeroPad, 0.0f)}) {
    SCOPED_TRACE(setting);
    auto config = live_policy_config(model("history_absent.onnx"));
    config.task.wbt_history_initialization = setting;
    Policy policy(config);
    auto& core = dynamic_cast<GeneralWBTPolicy&>(policy.policy()).core();
    EXPECT_EQ(core.HistoryInitialization(), mode);
    auto window = reference_window(core.LookaheadSlots());
    window.q.assign(window.q.size(), 0.5);
    policy.set_reference(&window);
    policy.step(robot_state());
    ASSERT_FLOAT_EQ(core.LastObservation()[2 * 58], 0.5f);  // the newest frame carries the marker
    EXPECT_FLOAT_EQ(core.LastObservation()[0], padded * 0.5f);
  }
  auto config = live_policy_config(model("history_absent.onnx"));
  config.task.wbt_history_initialization = "zero";
  EXPECT_THROW(Policy{config}, ConfigError);
}

TEST(PolicyApi, AllLiveLayoutsPreserveTimingResetAndCurrentReference) {
  for (const auto* name :
       {"pelvis.onnx", "torso.onnx", "split_history.onnx", "legacy_history.onnx", "sonic_flat.onnx"}) {
    SCOPED_TRACE(name);
    Policy policy(live_policy_config(model(name)));
    auto& core = dynamic_cast<GeneralWBTPolicy&>(policy.policy()).core();
    live::WbtPolicy baseline;
    ASSERT_TRUE(baseline.Initialize(model(name)));
    double now = 10.0;
    core.SetElapsedSecondsFn([&] { return now; });
    baseline.SetElapsedSecondsFn([&] { return now; });
    auto state = robot_state();
    auto window = reference_window(core.LookaheadSlots());
    ReferenceFrame current;
    current.q.assign(29, 0.25);
    current.dq.assign(29, 0.0);
    window.current = current;
    policy.set_reference(&window);
    for (int tick = 0; tick < 6; ++tick) {
      if (tick == 3) {
        policy.reset();
        baseline.Reset();
      }
      now += 0.06;  // host missed two control intervals
      live::RobotObservation obs;
      live::WbtReferenceSequence reference;
      reference.current.dof_pos.fill(0.25f);
      reference.lookahead.resize(static_cast<size_t>(core.LookaheadSlots()));
      std::array<float, 29> offsets{};
      ASSERT_TRUE(baseline.ComputeAction(obs, reference, offsets.data()));
      const auto& command = policy.step(state);
      EXPECT_EQ(core.LastTimeStep(), baseline.LastTimeStep());
      EXPECT_EQ(policy.policy().last_observation(), baseline.LastObservation());
      for (size_t j = 0; j < 29; ++j)
        EXPECT_NEAR(command.q[j], static_cast<double>(offsets[j]) + baseline.DefaultDofPos()[j], 1e-6);
    }
    auto replacement = window;
    replacement.current->q.assign(29, 0.5);
    policy.set_reference(&replacement);
    EXPECT_NO_THROW(policy.step(state));
  }
}

TEST(PolicyApi, RelativePresetModelsAndResearchMotionWorkThroughFrontDoor) {
  for (const auto* name : {"g1-29dof-loco", "g1-29dof-wbt"}) {
    SCOPED_TRACE(name);
    const std::string checkpoint = std::string(name) == "g1-29dof-wbt"
                                       ? std::string(HOLOSOMA_TEST_PRESET_DIR) +
                                             "/../../../holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx"
                                       : "";
    Policy policy(std::string(HOLOSOMA_TEST_PRESET_DIR) + "/" + name + ".yaml", checkpoint);
    auto state = robot_state();
    state.dof_pos = policy.config().robot.default_dof_angles;
    EXPECT_EQ(policy.num_dofs(), 29);
    EXPECT_EQ(policy.dof_names().size(), 29u);
    EXPECT_THROW(policy.set_reference(nullptr), std::runtime_error);
    EXPECT_EQ(policy.step(state).q.size(), 29u);
    EXPECT_FALSE(policy.policy().last_observation().empty());
    policy.step(state);
    if (auto* wbt = dynamic_cast<WholeBodyTrackingPolicy*>(&policy.policy())) {
      EXPECT_EQ(wbt->motion_timestep(), 2);
      policy.reset();
      policy.step(state);
      EXPECT_EQ(wbt->motion_timestep(), 1);
    }
  }
}

TEST(PolicyApi, NewClipRequiresResetToMatchItsFreshHeadingAnchor) {
  const auto config = live_policy_config(model("sonic_flat_0p2s.onnx"));
  Policy streaming(config), fresh_rotated(config);
  auto state = robot_state();
  auto clip_a = reference_window(5);
  auto clip_b = reference_window(5);
  const double half_yaw = std::acos(-1.0) / 4.0;
  clip_b.quat.assign(5, {std::cos(half_yaw), 0, 0, std::sin(half_yaw)});
  // This fixture's real metadata declares 9 * 93 proprio values, then five
  // frames each of q and dq, followed by the obs07 relative-orientation block.
  constexpr size_t orientation = 9 * 93 + 2 * 5 * 29;
  auto rotation = [&](const Policy& policy) {
    const auto& obs = policy.policy().last_observation();
    if (obs.size() != 2267u) throw std::runtime_error("The reference-swap fixture must have 2267 observations");
    return std::vector<float>(obs.begin() + orientation, obs.begin() + orientation + 30);
  };
  auto expect_rotation = [](const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5) << i;
  };

  streaming.set_reference(&clip_a);
  streaming.step(state);
  const auto a = rotation(streaming);
  fresh_rotated.set_reference(&clip_b);
  fresh_rotated.step(state);
  const auto b = rotation(fresh_rotated);
  expect_rotation(a, b);  // fresh yaw-0 and yaw-90 clips both start aligned
  EXPECT_NEAR(b[0], 1, 1e-5);
  EXPECT_NEAR(b[3], 1, 1e-5);

  streaming.set_reference(&clip_b);
  streaming.step(state);
  const auto without_reset = rotation(streaming);
  EXPECT_NEAR(without_reset[0], 0, 1e-5);
  EXPECT_NEAR(without_reset[1], -1, 1e-5);
  EXPECT_NEAR(without_reset[2], 1, 1e-5);

  // The documented clip-transition sequence must match a fresh policy.
  streaming.reset();
  streaming.step(state);
  expect_rotation(rotation(streaming), b);
}

TEST(PolicyApi, OfflineInterfaceNeverPublishesAndHoldsOnlySuppliedState) {
  OfflineInterface robot;
  RobotState state;
  EXPECT_FALSE(robot.read_state(state));
  robot.set_state(robot_state());
  EXPECT_TRUE(robot.read_state(state));
  EXPECT_EQ(state.dof_pos.size(), 29u);
  EXPECT_FALSE(robot.read_wireless_controller());
  JointCommand command;
  command.q = {1, 2};
  robot.send_command(command);
  EXPECT_EQ(robot.command().q, command.q);
}
}  // namespace
}  // namespace holosoma
