#include "holosoma_cpp/controller.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <thread>

#include "holosoma_cpp/locomotion_policy.hpp"
#include "holosoma_cpp/logging.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace holosoma {
namespace {

std::atomic<bool> g_stop_requested{false};

}  // namespace

std::unique_ptr<BasePolicy> make_policy(const InferenceConfig& config, PolicyContext& context) {
  InferenceConfig own = config;
  own.secondary.reset();
  std::unique_ptr<BasePolicy> policy;
  if (own.is_wbt()) {
    policy = std::make_unique<WholeBodyTrackingPolicy>(std::move(own), context);
  } else {
    policy = std::make_unique<LocomotionPolicy>(std::move(own), context);
  }
  policy->initialize();
  return policy;
}

PolicyController::PolicyController(const InferenceConfig& config, PolicyContext& context) {
  primary_ = make_policy(config, context);
  active_ = primary_.get();
  if (config.secondary) {
    if (config.secondary->task.rl_rate != config.task.rl_rate) {
      throw ConfigError("Dual mode needs the same task.rl_rate for both policies (the loop runs at one rate)");
    }
    secondary_ = make_policy(*config.secondary, context);
    // Both default key maps already bind X (joystick) and x (keyboard) to SWITCH_MODE.
    context.command_provider->bind("X", StateCommand::SWITCH_MODE);
    context.command_provider->bind("x", StateCommand::SWITCH_MODE);
    log::info(log::cat("Dual-mode: primary=", primary_->kind(), ", secondary=", secondary_->kind(),
                       ". Press X (joystick) or x (keyboard) to switch policies."),
              log::Color::kMagenta);
  }
}

void PolicyController::dispatch(StateCommand command) {
  if (command == StateCommand::SWITCH_MODE && dual_mode()) {
    switch_mode();
    return;
  }
  active_->dispatch_command(command);
}

void PolicyController::switch_mode() {
  active_->handle_stop_policy();
  BasePolicy* target = active_ == primary_.get() ? secondary_.get() : primary_.get();
  active_ = target;
  // Gains are per model slot, and both policies share one input provider, so
  // nothing else carries over: re-arm the gait phase and start the target.
  active_->init_phase_components();
  active_->handle_start_policy();
  log::info(log::cat("Switched to ", active_is_primary() ? "primary" : "secondary", " policy (", active_->kind(), ")"),
            log::Color::kMagenta);
}

Runner::Runner(PolicyController& controller, PolicyContext& context, const InferenceConfig& config)
    : controller_(controller), context_(context), config_(config) {}

void Runner::request_stop() { g_stop_requested = true; }
bool Runner::stop_requested() { return g_stop_requested; }

void Runner::fault(const std::string& reason) {
  if (!faulted_) {
    faulted_ = true;
    fault_reason_ = reason;
    log::error("FAULT: " + reason + " -> damping");
  }
  send_damping();
}

void Runner::send_damping() {
  const int n = config_.robot.num_joints;
  JointCommand cmd;
  RobotState state;
  const bool have_state = context_.robot->read_state(state);
  cmd.q.assign(static_cast<size_t>(n), 0.0);
  if (have_state && state.dof_pos.size() == static_cast<size_t>(n)) {
    for (int i = 0; i < n; ++i) {
      if (std::isfinite(state.dof_pos[static_cast<size_t>(i)])) cmd.q[static_cast<size_t>(i)] = state.dof_pos[i];
    }
  }
  cmd.dq.assign(static_cast<size_t>(n), 0.0);
  cmd.tau.assign(static_cast<size_t>(n), 0.0);
  cmd.kp.assign(static_cast<size_t>(n), 0.0);
  cmd.kd.assign(static_cast<size_t>(n), config_.runtime.damping_kd);
  context_.robot->send_command(cmd);
}

StepResult Runner::step() {
  if (faulted_) {
    send_damping();
    return StepResult::kFault;
  }
  ++iteration_;
  BasePolicy* policy = &controller_.active();
  policy->latency().start_cycle();
  try {
    if (const auto vc = context_.velocity_input->poll_velocity()) {
      policy->apply_velocity(*vc);
    }
    const auto commands = context_.command_provider->poll_commands();
    for (StateCommand command : commands) {
      controller_.dispatch(command);
      if (context_.kill_requested) return StepResult::kKilled;
    }
    policy = &controller_.active();
    if (!commands.empty()) policy->print_control_status();
    if (policy->use_phase()) policy->update_phase_time();

    {
      LatencyTracker::Scope scope(policy->latency(), "read_state");
      if (!context_.robot->read_state(state_)) {
        fault("robot state unavailable");
        return StepResult::kFault;
      }
    }
    const double timeout = config_.runtime.state_timeout_s;
    if (timeout > 0.0) {
      const double age = std::chrono::duration<double>(SteadyClock::now() - state_.stamp).count();
      if (age > timeout) {
        fault(log::cat("robot state is stale (", age, " s old, timeout ", timeout, " s)"));
        return StepResult::kFault;
      }
    }
    policy->policy_action(state_);
  } catch (const std::exception& e) {
    fault(e.what());
    return StepResult::kFault;
  }
  policy->latency().end_cycle();
  if (iteration_ % 50 == 1 && policy->use_policy_action()) {
    char fps[32];
    std::snprintf(fps, sizeof(fps), "%.2f", policy->latency().fps());
    log::info(log::cat("RL FPS: ", fps, " | ", policy->latency().stats_string()));
  }
  return StepResult::kContinue;
}

bool Runner::wait_for_first_state() {
  RobotState state;
  const auto start = SteadyClock::now();
  auto last_log = start;
  while (!stop_requested()) {
    if (context_.robot->read_state(state)) {
      log::info("Robot state received from " + context_.robot->description(), log::Color::kGreen);
      return true;
    }
    const auto now = SteadyClock::now();
    if (now - last_log > std::chrono::seconds(2)) {
      log::warning("Waiting for robot state from " + context_.robot->description() + " ...");
      last_log = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

int Runner::run() {
  if (!wait_for_first_state()) {
    return 0;
  }
  RateLimiter rate(config_.task.rl_rate);
  StepResult result = StepResult::kContinue;
  while (!stop_requested()) {
    result = step();
    if (result != StepResult::kContinue) break;
    rate.sleep();
  }

  // Leave the robot damped rather than holding the last policy target.
  constexpr double kShutdownDampingS = 0.5;
  const int ticks = std::max(1, static_cast<int>(kShutdownDampingS * config_.task.rl_rate));
  log::info(log::cat("Holding damping for ", kShutdownDampingS, " s before exit"));
  RateLimiter damping_rate(config_.task.rl_rate);
  for (int i = 0; i < ticks; ++i) {
    send_damping();
    damping_rate.sleep();
  }
  if (rate.overruns() > 0) {
    log::warning(log::cat("Control loop missed its deadline ", rate.overruns(), " times"));
  }
  return result == StepResult::kFault ? 1 : 0;
}

}  // namespace holosoma
