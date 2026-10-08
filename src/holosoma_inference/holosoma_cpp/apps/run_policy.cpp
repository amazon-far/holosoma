// holosoma_run_policy: C++ counterpart of holosoma_inference/run_policy.py, e.g.
//
//   holosoma_run_policy inference:g1-29dof-wbt --task.model-path <model.onnx> --task.interface lo

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "holosoma_cpp/cli.hpp"
#include "holosoma_cpp/controller.hpp"
#include "holosoma_cpp/input.hpp"
#include "holosoma_cpp/logging.hpp"
#include "holosoma_cpp/terminal.hpp"
#include "holosoma_cpp/unitree_interface.hpp"

namespace {

using namespace holosoma;

void on_signal(int) { Runner::request_stop(); }

void print_control_guide(bool wbt, bool joystick, bool dual_mode) {
  const auto line = [](const std::string& s) { log::info(s); };
  line(std::string(80, '='));
  line("POLICY CONTROLS");
  line(std::string(80, '='));
  if (joystick) {
    line("Using JOYSTICK control mode");
    line("  A button       - Start the policy");
    line("  B button       - Stop the policy");
    line("  Y button       - Set robot to default pose");
    line("  L1+R1 (LB+RB)  - Kill controller program (robot is left damped)");
    if (wbt) {
      line("  Select+A       - Start motion clip");
    } else {
      line("  Start button   - Switch walking/standing mode");
      line("  Left stick     - Adjust linear velocity");
      line("  Right stick    - Adjust angular velocity");
    }
    line("  Up/Down, Left/Right, F1 - kp level +-0.1, +-0.01, reset");
  } else {
    line("Using KEYBOARD control mode (keys go to THIS terminal)");
    line("  ]  - Start the policy");
    line("  o  - Stop the policy");
    line("  i  - Set robot to default pose");
    if (wbt) {
      line("  m  - Start motion clip");
    } else {
      line("  =          - Switch walking/standing mode");
      line("  w/s        - Increase/decrease forward velocity");
      line("  a/d        - Increase/decrease lateral velocity");
      line("  q/e        - Increase/decrease angular velocity");
      line("  z          - Set all velocities to zero");
    }
    line("  g/f, b/v, r - kp level +-0.1, +-0.01, reset");
    line("  1-9        - Switch between loaded models");
    line("  Ctrl-C     - Exit (robot is left damped)");
  }
  line("MuJoCo simulator keys (in the MuJoCo window): 7/8 band length, 9 toggle band, BACKSPACE reset");
  if (dual_mode) {
    line(joystick ? "  X button       - Switch between primary and secondary policy"
                  : "  x              - Switch between primary and secondary policy");
  }
  line(std::string(80, '='));
}

int run(const CliOptions& options) {
  const InferenceConfig& config = options.config;
  log::info("Starting policy (C++ runtime)");
  log::info("Robot: " + config.robot.robot_type);
  log::info(log::cat("RL Rate: ", config.task.rl_rate, " Hz"));
  log::info("Preset: " + options.preset_path);
  for (const auto& path : config.task.model_path) log::info("Model path: " + path);

  std::unique_ptr<RobotInterface> robot =
      make_unitree_interface(config.robot, config.task.domain_id, config.task.interface);

  // Input providers. When both channels use the same source a single provider serves both.
  const std::string& vel_source = config.task.velocity_input;
  const std::string& cmd_source = config.task.state_input;
  const bool uses_keyboard = vel_source == "keyboard" || cmd_source == "keyboard";
  std::unique_ptr<KeyboardInput> keyboard_vel, keyboard_cmd;
  std::unique_ptr<InterfaceInput> joystick;
  auto make_keyboard = [](bool velocity_keys) {
    return std::make_unique<KeyboardInput>(KeyboardListener::instance().subscribe(), velocity_keys);
  };
  VelocityProvider* velocity_input = nullptr;
  CommandProvider* command_provider = nullptr;
  if (vel_source == "interface" || cmd_source == "interface") joystick = std::make_unique<InterfaceInput>(*robot);
  if (vel_source == "keyboard") {
    keyboard_vel = make_keyboard(true);
    velocity_input = keyboard_vel.get();
  } else {
    velocity_input = joystick.get();
  }
  if (cmd_source == vel_source) {
    command_provider = vel_source == "keyboard" ? static_cast<CommandProvider*>(keyboard_vel.get()) : joystick.get();
  } else if (cmd_source == "keyboard") {
    keyboard_cmd = make_keyboard(false);
    command_provider = keyboard_cmd.get();
  } else {
    command_provider = joystick.get();
  }

  PolicyContext context;
  context.robot = robot.get();
  context.velocity_input = velocity_input;
  context.command_provider = command_provider;
  PolicyController controller(config, context);
  log::info("Policy initialized successfully!", log::Color::kGreen);
  print_control_guide(config.is_wbt(), vel_source == "interface" || cmd_source == "interface", controller.dual_mode());

  if (config.is_wbt()) {
    if (config.task.skip_stiff_prompt) {
      log::info("Entering stiff hold mode (skip_stiff_prompt)", log::Color::kGreen);
    } else if (stdin_is_tty()) {
      log::info("Ready to enter stiff hold mode. Press Enter to continue...", log::Color::kYellow);
      if (!wait_for_enter()) log::warning("Non-interactive mode detected - cannot prompt for stiff mode confirmation!");
      log::info("Entering stiff hold mode", log::Color::kGreen);
    } else {
      log::warning("Non-interactive mode detected - cannot prompt for stiff mode confirmation!");
    }
  }

  if (uses_keyboard && !KeyboardListener::instance().start()) {
    log::warning("No TTY - keyboard input disabled; starting the policy immediately");
    controller.primary().force_policy_active();
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  Runner runner(controller, context, config);
  const int code = runner.run();
  KeyboardListener::instance().stop();
  if (runner.faulted()) {
    log::error("Stopped after a fault: " + runner.fault_reason());
  } else {
    log::info("Policy execution completed!", log::Color::kGreen);
  }
  return code;
}

}  // namespace

int main(int argc, char** argv) {
  holosoma::log::init_from_env();
  const std::string program = argc > 0 ? argv[0] : "holosoma_run_policy";
  const std::string preset_dir = holosoma::default_preset_dir();
  holosoma::CliOptions options;
  try {
    options = holosoma::parse_cli(std::vector<std::string>(argv + 1, argv + argc), preset_dir);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n\n" << holosoma::cli_usage(program, preset_dir);
    return 2;
  }
  if (options.show_help) {
    std::cout << holosoma::cli_usage(program, preset_dir);
    return 0;
  }
  try {
    return run(options);
  } catch (const std::exception& e) {
    holosoma::TerminalMode::instance().restore();
    holosoma::log::error(std::string("Error running policy: ") + e.what());
    return 1;
  }
}
