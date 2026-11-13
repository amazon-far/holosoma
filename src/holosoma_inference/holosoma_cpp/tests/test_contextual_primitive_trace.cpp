// Primitive-command contextual policy, replayed against a trace recorded by the
// shared pipeline before it took a second command interface. Every tick's
// observation, depth stack, action, joint command and climb/command state must
// be bit-identical, so any change the generalisation made to the existing
// family shows up here by step and field.
//
// The trace in golden/contextual_primitive_trace.json was recorded by building
// this file against the pre-generalisation pipeline and running it with
// HOLOSOMA_RECORD_CONTEXTUAL_TRACE=<path>.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "holosoma_cpp/api.hpp"
#include "holosoma_cpp/contextual_policy.hpp"

namespace {
using namespace holosoma;
using Json = nlohmann::json;
using Chain = std::vector<std::pair<std::string, std::string>>;
const std::filesystem::path fixtures(HOLOSOMA_CONTEXTUAL_FIXTURE_DIR);
const std::filesystem::path golden(HOLOSOMA_CONTEXTUAL_GOLDEN_DIR);

InferenceConfig config(const std::string& discretizer) {
  InferenceConfig c;
  c.robot.num_joints = c.robot.num_motors = 3;
  c.robot.dof_names = {"hip", "knee", "ankle"};
  c.robot.default_dof_angles = c.robot.default_motor_angles = {.1, .2, .3};
  c.robot.joint2motor = c.robot.motor2joint = {0, 1, 2};
  c.observation.obs_dict = {{"actor_obs", {}}};
  c.task.model_path = {(fixtures / "contextual.onnx").string()};
  c.task.use_phase = false;
  c.contextual.emplace();
  c.contextual->depth_height = 2;
  c.contextual->depth_width = 3;
  c.contextual->imu_to_obs_joint_chain = Chain{{"hip", "z"}, {"knee", "x"}};
  c.contextual->discretizer = discretizer;
  if (discretizer == "nearest_primitive") {
    c.contextual->velocity_primitives = {{"stand", {0, 0}},       {"forward", {.6, 0}},   {"left_45", {.4, .5}},
                                         {"right_45", {.4, -.5}}, {"turn_left", {0, .8}}, {"turn_right", {0, -.8}}};
    c.contextual->switch_margin = .05;
    c.contextual->in_place_yaw_threshold = .1;
  }
  c.runtime.intra_op_threads = 1;
  return c;
}

const contextual::Policy& core(Policy& policy) { return dynamic_cast<const ContextualPolicy&>(policy.policy()).core(); }

// The scripted operator and robot for tick i: every command event the
// primitive family takes, a tilting base, bent chain joints, and both gravity
// sources.
Json Trace(const std::string& discretizer) {
  Policy policy(config(discretizer));
  const std::vector<std::array<double, 3>> velocities{{.5, 0, 0}, {0, 0, .6},   {.3, 0, -.4}, {-.4, 0, 0},
                                                      {0, .5, 0}, {0, -.5, .1}, {.15, 0, .5}, {0, 0, 0}};
  const std::vector<std::string> keys{"w", "a", "d", "s", "j", "l", "z"};
  Json steps = Json::array();
  for (int i = 0; i < 40; ++i) {
    if (i == 33) policy.reset();
    ContextualInput input;
    input.monotonic_time = .25 * i;
    if (i % 5 != 2) {
      std::vector<float> frame(6);
      for (size_t k = 0; k < frame.size(); ++k) frame[k] = static_cast<float>(.1 + .09 * std::sin(.7 * i + 2 * k));
      input.depth = frame;
    }
    if (i % 3 == 1) input.velocity = velocities[static_cast<size_t>(i / 3) % velocities.size()];
    if (i % 9 == 4) input.key = keys[static_cast<size_t>(i / 9) % keys.size()];
    if (i == 20) input.speed = 1;
    if (i == 30) input.speed = 0;
    if (i >= 10 && i < 16) input.dz = true;
    if (i == 16) input.dz = false;
    if (i % 8 == 0) {
      input.allow_climb_at_stand = (i / 8) % 2 == 1;
      input.allow_climb_continuation = (i / 8) % 3 == 1;
    }
    if (i % 7 == 3) input.fixed_arm = (i / 7) % 3;
    if (i == 25) input.command_index = 14;
    if (i == 27) input.command_index = 0;
    RobotState s;
    s.dof_pos = s.dof_vel = std::vector<double>(3);
    for (size_t j = 0; j < 3; ++j) {
      s.dof_pos[j] = .1 * (j + 1) + .3 * std::sin(.4 * i + j);
      s.dof_vel[j] = std::cos(.3 * i + 2 * j);
    }
    std::array<double, 4> q{1, .2 * std::sin(.2 * i), .15 * std::cos(.3 * i), .1 * std::sin(.5 * i)};
    const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    s.base_quat_wxyz.clear();
    for (double v : q) s.base_quat_wxyz.push_back(v / norm);
    s.base_ang_vel = {.1 * i, -.05 * i, .02};
    if (i % 4 == 3) s.projected_gravity = std::vector<double>{.1, -.05, -.99};
    const auto& command = policy.step(s, input);
    const auto& p = core(policy);
    steps.push_back({
        {"q", command.q},
        {"kp", command.kp},
        {"kd", command.kd},
        {"observation", p.observation()},
        {"depth", p.depth()},
        {"action", p.action()},
        {"scaled", p.scaled()},
        {"active", p.active_command()},
        {"effective", p.effective_command()},
        {"climbing", p.climbing()},
        {"climb_pred", p.climb_pred()},
        {"pending", p.pending() ? *p.pending() : -1},
        {"speed", p.speed()},
        {"calls", p.calls()},
    });
  }
  return steps;
}

TEST(ContextualPrimitiveTrace, MatchesThePreGeneralisationPipelineBitForBit) {
  Json actual = Json::object();
  for (const char* discretizer : {"simple", "nearest_primitive"}) actual[discretizer] = Trace(discretizer);
  if (const char* record = std::getenv("HOLOSOMA_RECORD_CONTEXTUAL_TRACE")) {
    std::ofstream(record) << actual.dump(1) << "\n";
    GTEST_SKIP() << "recorded " << record;
  }
  const auto expected = Json::parse(std::ifstream(golden / "contextual_primitive_trace.json"));
  size_t compared = 0;
  for (const char* discretizer : {"simple", "nearest_primitive"}) {
    const auto& want = expected.at(discretizer);
    const auto& got = actual.at(discretizer);
    ASSERT_EQ(got.size(), want.size()) << discretizer;
    for (size_t i = 0; i < want.size(); ++i)
      for (const auto& [field, value] : want[i].items()) {
        EXPECT_EQ(got[i].at(field), value) << discretizer << " step " << i << " field " << field;
        ++compared;
      }
  }
  // Anti-vacuity: both runs, every step, every field.
  EXPECT_EQ(compared, 2u * 40u * 14u);
}
}  // namespace
