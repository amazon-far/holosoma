// The continuous-velocity command interface of the shared contextual pipeline.
//
// The reference traces come from an independent numpy statement of the
// deployment contract (golden/velocity_reference.json); the export stubs carry
// the verbatim metadata of a real continuous-velocity export
// (golden/velocity_export_metadata.json), unchanged or with one key broken.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "holosoma_cpp/api.hpp"
#include "holosoma_cpp/contextual_policy.hpp"

namespace {
using namespace holosoma;
using Json = nlohmann::json;
using Chain = std::vector<std::pair<std::string, std::string>>;
const std::filesystem::path fixtures(HOLOSOMA_VELOCITY_FIXTURE_DIR);
const std::filesystem::path golden(HOLOSOMA_CONTEXTUAL_GOLDEN_DIR);

Json Load(const std::filesystem::path& path) { return Json::parse(std::ifstream(path)); }
const Json& Export() {
  static const Json value = Load(golden / "velocity_export_metadata.json");
  return value;
}
std::vector<std::string> G1Joints() {
  return Json::parse(Export().at("metadata_props").at("dof_names").get<std::string>()).get<std::vector<std::string>>();
}
std::vector<double> G1Defaults() {
  return Json::parse(Export().at("metadata_props").at("default_dof_angles").get<std::string>())
      .get<std::vector<double>>();
}

contextual::Spec SmallDepthSpec() {
  contextual::Spec spec;
  spec.joints = G1Joints();
  spec.defaults = G1Defaults();
  spec.height = 2;
  spec.width = 3;
  return spec;
}
contextual::Options Continuous() {
  contextual::Options options;
  options.discretizer.reset();
  options.require_climb_head = false;
  return options;
}
std::shared_ptr<OnnxModel> Model(const std::string& name) {
  return std::make_shared<OnnxModel>((fixtures / name).string());
}
std::unique_ptr<contextual::Policy> Make(const std::string& model, contextual::Spec spec = SmallDepthSpec(),
                                         contextual::Options options = Continuous()) {
  return std::make_unique<contextual::Policy>(Model(model), std::move(spec), std::move(options));
}

template <class T>
void Near(const T& actual, const Json& expected, double tol, const char* what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i].template get<double>(), tol) << what << " element " << i;
}

contextual::State StepState(const Json& step) {
  contextual::State state;
  state.q = step.at("q").get<std::vector<double>>();
  state.dq = step.at("dq").get<std::vector<double>>();
  state.quat = step.at("quat").get<std::array<double, 4>>();
  state.omega = step.at("omega").get<std::array<double, 3>>();
  if (!step.at("gravity").is_null()) state.gravity = step.at("gravity").get<std::array<double, 3>>();
  return state;
}

// Replays one reference case. The reference reports 0/1 standing/moving as its
// command index; this interface reports no primitive at all, so the index
// field is checked as kNoPrimitive rather than copied.
size_t Replay(contextual::Policy& policy, const Json& steps) {
  size_t ticks = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& step = steps[i];
    if (step.at("reset")) policy.Reset();
    if (step.contains("fixed")) policy.SetFixedArm(step.at("fixed"));
    const auto& v = step.at("velocity");
    EXPECT_EQ(policy.SetVelocity(v[0], v[1], v[2]), contextual::kNoPrimitive);
    std::optional<std::vector<float>> depth;
    if (!step.at("depth").is_null()) depth = step.at("depth").get<std::vector<float>>();
    const auto target = policy.Target(StepState(step), depth, .02 * i);
    Near(policy.observation(), step.at("obs"), 1e-6, "observation");
    Near(policy.action(), step.at("action"), 2e-5, "action");
    Near(target, step.at("target"), 2e-5, "target");
    EXPECT_EQ(policy.effective_command(), contextual::kNoPrimitive);
    ++ticks;
  }
  return ticks;
}

const Json& Reference() {
  static const Json value = Load(golden / "velocity_reference.json");
  return value;
}

TEST(ContextualVelocity, ReplaysTheIndependentReferenceContract) {
  const auto& ref = Reference();
  ASSERT_EQ(ref.at("schema"), 1);
  ASSERT_EQ(ref.at("raw_action_clip"), 20);
  const auto& linear = ref.at("cases")[0];
  ASSERT_EQ(linear.at("model"), "hybrid_linear.onnx");
  auto policy = Make("velocity_linear.onnx");
  // The trace bends every joint, the waist included, around the stance and
  // tilts the base; a third of its ticks supply gravity directly.
  EXPECT_EQ(Replay(*policy, linear.at("steps")), 30u);
  EXPECT_EQ(policy->calls(), 30);
}

TEST(ContextualVelocity, ClipsAtTheSharedCeilingAndFeedsTheClippedActionBack) {
  const auto& feedback = Reference().at("cases")[1];
  ASSERT_EQ(feedback.at("model"), "hybrid_feedback.onnx");
  auto policy = Make("velocity_feedback.onnx");
  // action = previous action + 15: 15, then 30 -> 20, then 35 -> 20.
  EXPECT_EQ(Replay(*policy, feedback.at("steps")), 4u);
  for (float value : policy->action()) EXPECT_FLOAT_EQ(value, 20.f);
  for (size_t m = 0; m < 29; ++m) EXPECT_FLOAT_EQ(policy->observation()[m], 20.f) << m;
}

TEST(ContextualVelocity, ScaledActionsAreInRuntimeJointOrder) {
  // The reference graph's joint order is a permutation of the runtime order,
  // so a model-order scaled() cannot pass by coincidence.
  auto policy = Make("velocity_linear.onnx");
  const auto& order = policy->real_to_model();
  size_t moved = 0;
  for (size_t m = 0; m < order.size(); ++m) moved += order[m] != static_cast<int>(m);
  ASSERT_GT(moved, 20u);
  const auto& step = Reference().at("cases")[0].at("steps")[0];
  std::optional<std::vector<float>> depth;
  if (!step.at("depth").is_null()) depth = step.at("depth").get<std::vector<float>>();
  const auto target = policy->Target(StepState(step), depth, 0);
  const auto& stance = policy->default_dof_pos();
  ASSERT_EQ(policy->scaled().size(), target.size());
  for (size_t real = 0; real < target.size(); ++real)
    EXPECT_NEAR(policy->scaled()[real], target[real] - stance[real], 1e-5) << real;
}

TEST(ContextualVelocity, ExposesTheModelsStanceAndGains) {
  const auto& linear = Reference().at("cases")[0];
  auto policy = Make("velocity_linear.onnx");
  EXPECT_TRUE(policy->continuous());
  EXPECT_TRUE(policy->gains_from_model());
  EXPECT_FALSE(policy->has_climb_head());
  EXPECT_FALSE(policy->climbing());
  Near(policy->kp(), linear.at("kp"), 1e-9, "kp");
  Near(policy->kd(), linear.at("kd"), 1e-9, "kd");
  Near(policy->default_dof_pos(), linear.at("defaults"), 1e-9, "stance");
  EXPECT_EQ(policy->depth_shape(), (std::vector<int64_t>{2, 3}));
}

TEST(ContextualVelocity, GravityStaysInTheImuFrameUnlessAChainIsConfigured) {
  const auto& step = Reference().at("cases")[0].at("steps")[1];  // quaternion-derived gravity
  ASSERT_TRUE(step.at("gravity").is_null());
  auto state = StepState(step);
  // Bend the waist well away from the stance so an inherited chain must move gravity.
  const auto joints = G1Joints();
  for (const char* waist : {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"})
    state.q[std::find(joints.begin(), joints.end(), waist) - joints.begin()] = .5;
  constexpr size_t kGravity = 29 + 3 + 29 + 29 + 2;  // after actions, ang vel, pos, vel, arm flag
  auto gravity = [&](contextual::Options options) {
    auto policy = Make("velocity_linear.onnx", SmallDepthSpec(), options);
    const auto obs = policy->Observation(state);
    return std::array<float, 3>{obs[kGravity], obs[kGravity + 1], obs[kGravity + 2]};
  };
  // Independent IMU-frame reference: R(q)^T (0, 0, -1).
  const auto [w, x, y, z] = state.quat;
  const std::array<double, 3> imu{-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y))};
  const auto unset = gravity(Continuous());
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(unset[i], imu[i], 1e-6) << i;
  auto waist = Continuous();
  waist.imu_to_obs_joint_chain = Chain{{"waist_yaw_joint", "z"}, {"waist_roll_joint", "x"}, {"waist_pitch_joint", "y"}};
  const auto chained = gravity(waist);
  EXPECT_GT(std::abs(chained[0] - unset[0]) + std::abs(chained[1] - unset[1]) + std::abs(chained[2] - unset[2]), .1);
  // Supplied gravity is used as given when no chain is configured.
  state.gravity = std::array<double, 3>{.1, -.2, -.97};
  const auto supplied = gravity(Continuous());
  EXPECT_FLOAT_EQ(supplied[0], .1f);
  EXPECT_FLOAT_EQ(supplied[1], -.2f);
  EXPECT_FLOAT_EQ(supplied[2], -.97f);
}

TEST(ContextualVelocity, RefusesCallsThatOnlyAPrimitiveCommandHas) {
  auto policy = Make("velocity_linear.onnx");
  EXPECT_THROW(policy->SetCommand(1), std::invalid_argument);
  EXPECT_THROW(policy->RequestKey("w", 0), std::invalid_argument);
  EXPECT_THROW(policy->SetSpeed(1), std::invalid_argument);
  EXPECT_THROW(policy->SetDz(true), std::invalid_argument);
  EXPECT_NO_THROW(policy->SetDz(false));  // a controller's reset path
  EXPECT_FALSE(policy->dz());
  EXPECT_NO_THROW(policy->SetClimbAtStandAllowed(true, true));
  EXPECT_FALSE(policy->UpdateTransition(100));
  EXPECT_EQ(policy->active_command(), contextual::kNoPrimitive);
  EXPECT_THROW(policy->SetFixedArm(3), std::invalid_argument);
  const double nan = std::nan("");
  EXPECT_THROW(policy->SetVelocity(nan, 0, 0), std::runtime_error);
}

TEST(ContextualVelocity, ClampsTheCommandToTheTrainedRangesWithoutADeadzone) {
  auto policy = Make("velocity_linear.onnx");
  const auto state = StepState(Reference().at("cases")[0].at("steps")[0]);
  constexpr size_t kCommand = 29 + 3 + 29 + 29 + 2 + 3;
  auto command = [&](double vx, double vy, double wz) {
    policy->SetVelocity(vx, vy, wz);
    const auto obs = policy->Observation(state);
    return std::array<float, 3>{obs[kCommand], obs[kCommand + 1], obs[kCommand + 2]};
  };
  // The reference ranges: x [-1, 1.5], y [-0.5, 0.5], yaw [-1.5, 1.5].
  EXPECT_EQ(command(9, -9, 9), (std::array<float, 3>{1.5f, -.5f, 1.5f}));
  EXPECT_EQ(command(-9, 9, -9), (std::array<float, 3>{-1.f, .5f, -1.5f}));
  EXPECT_EQ(command(.01, -.02, .03), (std::array<float, 3>{.01f, -.02f, .03f}));
}

TEST(ContextualVelocity, RefusesOptionsThatOnlyAPrimitiveCommandHas) {
  auto expect_refused = [](contextual::Options options, contextual::Spec spec, const char* text) {
    try {
      Make("velocity_linear.onnx", std::move(spec), std::move(options));
      ADD_FAILURE() << "accepted: " << text;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
    }
  };
  auto o = Continuous();
  o.require_climb_head = true;
  expect_refused(o, SmallDepthSpec(), "require_climb_head");
  o = Continuous();
  o.force_manual_climbing = true;
  expect_refused(o, SmallDepthSpec(), "climb options");
  o = Continuous();
  o.climb_enter = "dz_only";
  expect_refused(o, SmallDepthSpec(), "climb options");
  o = Continuous();
  o.velocity_primitives = {{"forward", {.5, 0}}};
  expect_refused(o, SmallDepthSpec(), "velocity_primitives");
  auto spec = SmallDepthSpec();
  spec.kp = std::vector<double>(29, 1.0);
  expect_refused(Continuous(), spec, "gains its model was trained with");
  // Control: the same options and spec load once nothing primitive-only is set.
  EXPECT_NO_THROW(Make("velocity_linear.onnx"));
}

TEST(ContextualVelocity, APrimitiveCommandRefusesAContinuousExport) {
  contextual::Options primitive;
  primitive.require_climb_head = false;
  primitive.imu_to_obs_joint_chain = Chain{};
  try {
    Make("velocity_linear.onnx", SmallDepthSpec(), primitive);
    ADD_FAILURE() << "a primitive command accepted a continuous-velocity export";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("discretizer: null"), std::string::npos) << e.what();
  }
}

contextual::Spec ExportSpec() {
  contextual::Spec spec;
  spec.joints = G1Joints();
  spec.defaults = G1Defaults();
  spec.height = 58;
  spec.width = 87;
  return spec;
}

TEST(ContextualVelocity, AcceptsTheRealExportContractVerbatim) {
  // The runtime order is a permutation of the export's, so the mapping is exercised.
  auto spec = ExportSpec();
  std::reverse(spec.joints.begin(), spec.joints.end());
  std::reverse(spec.defaults.begin(), spec.defaults.end());
  auto policy = Make("velocity_export.onnx", spec);
  const auto& meta = Export().at("metadata_props");
  const auto kp = Json::parse(meta.at("kp").get<std::string>()).get<std::vector<double>>();
  const auto stance = Json::parse(meta.at("default_dof_angles").get<std::string>()).get<std::vector<double>>();
  for (size_t m = 0; m < 29; ++m) {
    EXPECT_DOUBLE_EQ(policy->kp()[28 - m], kp[m]) << m;
    EXPECT_DOUBLE_EQ(policy->default_dof_pos()[28 - m], stance[m]) << m;
  }
  contextual::State state;
  state.q = spec.defaults;
  state.dq.assign(29, 0);
  policy->SetVelocity(.7, -.2, .4);
  policy->Target(state, std::vector<float>(58 * 87, .25f), 0);
  const auto first = policy->action();
  ASSERT_EQ(policy->observation().size(), 98u);
  // velocity_command_projection.observation_start/end in the export: [95, 98).
  EXPECT_FLOAT_EQ(policy->observation()[95], .7f);
  EXPECT_FLOAT_EQ(policy->observation()[96], -.2f);
  EXPECT_FLOAT_EQ(policy->observation()[97], .4f);
  // Stub graph: action = previous action + bias, so the second tick reads the first.
  policy->Target(state, std::nullopt, .02);
  for (size_t m = 0; m < 29; ++m) {
    EXPECT_FLOAT_EQ(policy->observation()[m], first[m]) << m;
    EXPECT_FLOAT_EQ(policy->action()[m], 2 * first[m]) << m;
  }
  EXPECT_NE(first.front(), first.back());
}

TEST(ContextualVelocity, ExposesTheDepthContractTheRealExportDeclares) {
  auto policy = Make("velocity_export.onnx", ExportSpec());
  const auto contract = policy->depth_contract();
  ASSERT_TRUE(contract.has_value());
  // Read back from the checked-in export metadata, not restated.
  const auto& meta = Export().at("metadata_props");
  const auto clip = Json::parse(meta.at("depth_clip_m").get<std::string>()).get<std::vector<double>>();
  const auto params = Json::parse(meta.at("depth_observation_params").get<std::string>());
  EXPECT_DOUBLE_EQ(contract->near_m, clip[0]);
  EXPECT_DOUBLE_EQ(contract->far_m, clip[1]);
  EXPECT_EQ(contract->invalid_fill, params.at("invalid_fill").get<std::string>());
  // And the values themselves, so a fixture edit cannot move both sides at once.
  EXPECT_DOUBLE_EQ(contract->near_m, 0.3);
  EXPECT_DOUBLE_EQ(contract->far_m, 2.0);
  EXPECT_EQ(contract->invalid_fill, "near");
}

TEST(ContextualVelocity, RefusesEachBrokenExportNamingWhatIsWrong) {
  const auto expected = Load(fixtures / "velocity_broken.json");
  // Anti-vacuity: the manifest must hold every case the generator writes.
  ASSERT_GE(expected.size(), 40u);
  for (const auto& [model, text] : expected.items()) {
    SCOPED_TRACE(model);
    try {
      Make(model, ExportSpec());
      ADD_FAILURE() << "accepted";
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      EXPECT_NE(what.find(text.get<std::string>()), std::string::npos) << what;
      EXPECT_NE(what.find(model), std::string::npos) << "the refusal names the file: " << what;
    }
  }
}

InferenceConfig ExportConfig() {
  InferenceConfig c;
  c.robot.num_joints = c.robot.num_motors = 29;
  c.robot.dof_names = G1Joints();
  // A robot stance unlike the model's, so the test can tell which one is used.
  c.robot.default_dof_angles = c.robot.default_motor_angles = std::vector<double>(29, 0.0);
  for (int i = 0; i < 29; ++i) c.robot.joint2motor.push_back(i), c.robot.motor2joint.push_back(i);
  c.observation.obs_dict = {{"actor_obs", {}}};
  c.task.model_path = {(fixtures / "velocity_export.onnx").string()};
  c.task.use_phase = false;
  c.contextual.emplace();
  c.contextual->discretizer.reset();
  c.contextual->require_climb_head = false;
  c.runtime.intra_op_threads = 1;
  return c;
}

TEST(ContextualVelocity, ThePublicFactoryRunsAContinuousExportOnItsOwnStanceAndGains) {
  Policy policy(ExportConfig());
  ASSERT_EQ(policy.policy().kind(), "ContextualPolicy");
  RobotState s;
  s.dof_pos = G1Defaults();
  s.dof_vel.assign(29, 0);
  ContextualInput input;
  input.velocity = std::array<double, 3>{.4, 0, 0};
  input.depth = std::vector<float>(58 * 87, 0);
  const auto& command = policy.step(s, input);
  const auto& core = dynamic_cast<const ContextualPolicy&>(policy.policy()).core();
  const auto& meta = Export().at("metadata_props");
  EXPECT_EQ(command.kp, Json::parse(meta.at("kp").get<std::string>()).get<std::vector<double>>());
  EXPECT_EQ(command.kd, Json::parse(meta.at("kd").get<std::string>()).get<std::vector<double>>());
  const auto stance = G1Defaults();
  for (size_t i = 0; i < 29; ++i)
    EXPECT_NEAR(command.q[i], stance[i] + core.scaled()[i], 1e-6) << i;  // scaled() is float32
  EXPECT_FLOAT_EQ(core.observation()[95], .4f);
  // Get-ready ramps to the model's stance, not to the robot's (all zeros here).
  Policy ready(ExportConfig(), false);
  ready.command(StateCommand::INIT);
  RobotState zero;
  zero.dof_pos.assign(29, 0);
  zero.dof_vel.assign(29, 0);
  const JointCommand* ramp = nullptr;
  for (int i = 0; i <= 500; ++i) ramp = &ready.step(zero, ContextualInput{});
  for (size_t i = 0; i < 29; ++i) EXPECT_NEAR(ramp->q[i], stance[i], 1e-12) << i;
  // Configured gains are refused rather than silently replacing the trained ones.
  auto gains = ExportConfig();
  gains.robot.motor_kp = gains.robot.motor_kd = std::vector<double>(29, 1.0);
  EXPECT_THROW(Policy{gains}, std::invalid_argument);
}
}  // namespace
