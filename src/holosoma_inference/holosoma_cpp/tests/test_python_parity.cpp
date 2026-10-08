// Replays the Python reference runs (tests/gen_golden.py) through the C++ controller:
// same CLI arguments, same robot states, same operator input, tick by tick.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "holosoma_cpp/cli.hpp"
#include "holosoma_cpp/controller.hpp"
#include "holosoma_cpp/input.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace holosoma {
namespace {

class ScriptedRobot final : public RobotInterface {
 public:
  bool read_state(RobotState& state) override {
    state = state_;
    state.stamp = SteadyClock::now();
    return true;
  }
  void send_command(const JointCommand& command) override {
    last_command_ = command;
    ++commands_;
  }
  std::optional<WirelessController> read_wireless_controller() override { return wireless_; }
  std::string description() const override { return "scripted robot"; }

  void set_state(const nlohmann::json& s) {
    state_.dof_pos = s["q"].get<std::vector<double>>();
    state_.dof_vel = s["dq"].get<std::vector<double>>();
    state_.base_quat_wxyz = s["quat"].get<std::vector<double>>();
    state_.base_ang_vel = s["omega"].get<std::vector<double>>();
    state_.tick_ms = s["tick"].get<uint32_t>();
  }
  void set_wireless(const nlohmann::json& w) {
    wireless_.keys = w["keys"].get<uint16_t>();
    wireless_.lx = w["lx"].get<float>();
    wireless_.ly = w["ly"].get<float>();
    wireless_.rx = w["rx"].get<float>();
  }

  RobotState state_;
  WirelessController wireless_;
  JointCommand last_command_;
  int commands_ = 0;
};

void expect_close(const std::vector<double>& actual, const std::vector<double>& expected, double atol,
                  const std::string& what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double tol = atol + 1e-5 * std::abs(expected[i]);
    ASSERT_NEAR(actual[i], expected[i], tol) << what << " index " << i;
  }
}

void run_scenario(const std::string& name) {
  std::ifstream f(std::string(HOLOSOMA_TEST_GOLDEN_DIR) + "/" + name + ".json");
  ASSERT_TRUE(f) << "missing golden " << name << "; run tests/gen_golden.py";
  const auto golden = nlohmann::json::parse(f);

  // Model paths in the golden files are relative to the repository root.
  std::filesystem::current_path(HOLOSOMA_TEST_REPO_ROOT);
  std::vector<std::string> args = golden["args"].get<std::vector<std::string>>();
  const InferenceConfig config = parse_cli(args, HOLOSOMA_TEST_PRESET_DIR).config;

  ScriptedRobot robot;
  robot.set_state(golden["ticks"][0]["state"]);
  auto queue = std::make_shared<KeyQueue>();
  KeyboardInput keyboard(queue, true);
  InterfaceInput joystick(robot);
  const bool use_joystick = golden["input"] == "interface";
  PolicyContext context;
  context.robot = &robot;
  context.velocity_input = use_joystick ? static_cast<VelocityProvider*>(&joystick) : &keyboard;
  context.command_provider = use_joystick ? static_cast<CommandProvider*>(&joystick) : &keyboard;
  PolicyController controller(config, context);
  Runner runner(controller, context, config);

  std::vector<double> kp, kd;
  int compared_obs = 0;
  for (size_t t = 0; t < golden["ticks"].size(); ++t) {
    SCOPED_TRACE(name + " tick " + std::to_string(t));
    const auto& tick = golden["ticks"][t];
    robot.set_state(tick["state"]);
    if (tick.contains("wireless")) robot.set_wireless(tick["wireless"]);
    for (char key : tick["keys"].get<std::string>()) queue->push(std::string(1, key));

    const StepResult result = runner.step();
    if (tick.contains("killed")) {
      EXPECT_EQ(result, StepResult::kKilled);
      return;
    }
    ASSERT_EQ(result, StepResult::kContinue) << runner.fault_reason();

    const auto& cmd = tick["command"];
    if (cmd.contains("kp")) kp = cmd["kp"].get<std::vector<double>>();
    if (cmd.contains("kd")) kd = cmd["kd"].get<std::vector<double>>();
    const JointCommand& sent = robot.last_command_;
    expect_close(sent.q, cmd["q"].get<std::vector<double>>(), 2e-5, "q");
    expect_close(sent.kp, kp, 1e-4, "kp");
    expect_close(sent.kd, kd, 1e-4, "kd");

    const BasePolicy& active = controller.active();
    if (tick.contains("obs")) {
      ASSERT_TRUE(active.inference_ran());
      const auto& obs = active.last_observation();
      expect_close(std::vector<double>(obs.begin(), obs.end()), tick["obs"].get<std::vector<double>>(), 2e-5, "obs");
      ++compared_obs;
    }
    if (tick.contains("motion_timestep")) {
      const auto* wbt = dynamic_cast<const WholeBodyTrackingPolicy*>(&active);
      ASSERT_NE(wbt, nullptr);
      EXPECT_EQ(wbt->motion_timestep(), tick["motion_timestep"].get<int>());
    }
  }
  EXPECT_GT(compared_obs, 0);
}

TEST(PythonParity, LocomotionKeyboardMultiModel) { run_scenario("loco_keyboard_multi_model"); }
TEST(PythonParity, WholeBodyTrackingDualMode) { run_scenario("wbt_dual_mode_keyboard"); }
TEST(PythonParity, WholeBodyTrackingSimTimeAndOptions) { run_scenario("wbt_sim_time_options"); }
TEST(PythonParity, LocomotionJoystickKill) { run_scenario("loco_joystick_kill"); }

}  // namespace
}  // namespace holosoma
