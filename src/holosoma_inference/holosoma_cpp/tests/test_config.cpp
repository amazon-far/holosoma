// Preset loading and the run_policy.py-style command line.

#include <gtest/gtest.h>

#include <filesystem>

#include "holosoma_cpp/cli.hpp"
#include "holosoma_cpp/config.hpp"

namespace holosoma {
namespace {

const std::string kPresetDir = HOLOSOMA_TEST_PRESET_DIR;
const std::string kWbtModel = std::string(HOLOSOMA_TEST_REPO_ROOT) +
                              "/src/holosoma_inference/holosoma_inference/models/wbt/ppo_g1_29dof_dancing.onnx";

CliOptions parse(std::vector<std::string> args) { return parse_cli(args, kPresetDir); }

TEST(Config, PresetsLoad) {
  const auto presets = list_presets(kPresetDir);
  ASSERT_EQ(presets, (std::vector<std::string>{"g1-29dof-loco", "g1-29dof-wbt"}));

  const auto loco = parse({"inference:g1-29dof-loco"}).config;
  EXPECT_FALSE(loco.is_wbt());
  EXPECT_EQ(loco.robot.dof_names.size(), 29u);
  EXPECT_DOUBLE_EQ(loco.task.policy_action_scale, 0.25);
  ASSERT_TRUE(loco.secondary);
  EXPECT_TRUE(std::filesystem::exists(loco.task.model_path.at(0)));  // the bundled FastSAC safety policy

  const auto wbt = parse({"inference:g1-29dof-wbt", "--task.model-path", kWbtModel}).config;
  EXPECT_TRUE(wbt.is_wbt());
  EXPECT_TRUE(wbt.task.action_scales_by_effort_limit_over_p_gain);
  ASSERT_TRUE(wbt.robot.stiff_startup_kp);
  EXPECT_DOUBLE_EQ(wbt.robot.stiff_startup_kp->at(0), 350.0);
  ASSERT_TRUE(wbt.secondary);
  EXPECT_FALSE(wbt.secondary->is_wbt());
}

TEST(Config, OverridesFollowTyroSpelling) {
  const auto o =
      parse({"inference:g1-29dof-wbt", "--task.model-path", kWbtModel, "--task.use-sim-time",
             "--task.motion-start-timestep", "5", "--task.motion-end-timestep=40", "--task.interface", "eth0",
             "--robot.interp-gain-scale", "1.5", "--runtime.state-timeout-s", "0", "--secondary", "none"})
          .config;
  EXPECT_TRUE(o.task.use_sim_time);
  EXPECT_EQ(o.task.motion_start_timestep, 5);
  ASSERT_TRUE(o.task.motion_end_timestep);
  EXPECT_EQ(*o.task.motion_end_timestep, 40);
  EXPECT_EQ(o.task.interface, "eth0");
  EXPECT_DOUBLE_EQ(o.robot.interp_gain_scale, 1.5);
  EXPECT_DOUBLE_EQ(o.runtime.state_timeout_s, 0.0);
  EXPECT_FALSE(o.secondary);

  const auto negated = parse({"inference:g1-29dof-loco", "--task.no-use-phase"}).config;
  EXPECT_FALSE(negated.task.use_phase);
}

TEST(Config, ModelPathsAreResolvedAndListed) {
  const auto o = parse({"inference:g1-29dof-wbt", "--task.model-path", kWbtModel, kWbtModel}).config;
  ASSERT_EQ(o.task.model_path.size(), 2u);
  EXPECT_TRUE(std::filesystem::path(o.task.model_path[0]).is_absolute());
}

TEST(Config, JoystickShortcutSelectsTheWirelessController) {
  const auto o = parse({"inference:g1-29dof-loco", "--task.use-joystick"}).config;
  EXPECT_EQ(o.task.velocity_input, "interface");
  EXPECT_EQ(o.task.state_input, "interface");
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--task.use-joystick", "--task.velocity-input", "keyboard",
                      "--task.state-input", "interface"}),
               ConfigError);
}

TEST(Config, RejectsMistakes) {
  EXPECT_THROW(parse({}), ConfigError);
  EXPECT_THROW(parse({"inference:does-not-exist"}), ConfigError);
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--task.no-such-field", "1"}), ConfigError);
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--task.rl-rate", "fast"}), ConfigError);
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--task.velocity-input", "ros2"}), ConfigError);
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--task.model-path", "wandb://entity/project/run/model.onnx"}),
               ConfigError);
  EXPECT_THROW(parse({"inference:g1-29dof-wbt"}), ConfigError);  // the WBT preset ships without a model
  EXPECT_THROW(parse({"inference:g1-29dof-loco", "--robot.motor-kp", "1", "2"}), ConfigError);
}

}  // namespace
}  // namespace holosoma
