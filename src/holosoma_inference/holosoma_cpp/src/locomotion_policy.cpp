#include "holosoma_cpp/locomotion_policy.hpp"

#include <cmath>
#include <cstdio>

#include "holosoma_cpp/logging.hpp"
#include "holosoma_cpp/math.hpp"

namespace holosoma {

LocomotionPolicy::LocomotionPolicy(InferenceConfig config, PolicyContext& context)
    : BasePolicy(std::move(config), context) {
  kind_ = "LocomotionPolicy";
}

std::set<std::string> LocomotionPolicy::available_terms() const {
  auto terms = BasePolicy::available_terms();
  terms.insert({"actions", "command_lin_vel", "command_ang_vel", "command_stand", "sin_phase", "cos_phase"});
  return terms;
}

void LocomotionPolicy::compute_terms(const RobotState& state, TermValues& terms) {
  BasePolicy::compute_terms(state, terms);
  terms["actions"] = last_policy_action_;
  terms["command_lin_vel"] = {lin_vel_command_[0], lin_vel_command_[1]};
  terms["command_ang_vel"] = {ang_vel_command_};
  terms["command_stand"] = {stand_command_};
  terms["sin_phase"] = {std::sin(phase_[0]), std::sin(phase_[1])};
  terms["cos_phase"] = {std::cos(phase_[0]), std::cos(phase_[1])};
}

void LocomotionPolicy::apply_velocity(const VelCmd& vc) {
  maybe_switch_to_walk_mode(vc);
  const double s = stand_command_;
  lin_vel_command_[0] = vc.lin_vel_x * s;
  lin_vel_command_[1] = vc.lin_vel_y * s;
  ang_vel_command_ = vc.ang_vel * s;
}

void LocomotionPolicy::maybe_switch_to_walk_mode(const VelCmd& vc) {
  if (!config_.task.auto_walk_on_vel_cmd || stand_command_ == 1.0) return;
  if (std::abs(vc.lin_vel_x) < 1e-3 && std::abs(vc.lin_vel_y) < 1e-3 && std::abs(vc.ang_vel) < 1e-3) return;
  stand_command_ = 1.0;
  base_height_command_ = desired_base_height_;
  log::info("Auto-walk: non-zero velocity received", log::Color::kBlue);
}

void LocomotionPolicy::update_phase_time() {
  BasePolicy::update_phase_time();
  const double lin_norm =
      std::sqrt(lin_vel_command_[0] * lin_vel_command_[0] + lin_vel_command_[1] * lin_vel_command_[1]);
  if (lin_norm < 0.01 && std::abs(ang_vel_command_) < 0.01) {
    // Standing still: both feet share the same phase.
    phase_[0] = phase_[1] = math::kPi;
    is_standing_ = true;
  } else if (is_standing_) {
    // Starting to move again: restart the gait cycle.
    phase_[0] = 0.0;
    phase_[1] = math::kPi;
    is_standing_ = false;
  }
}

void LocomotionPolicy::dispatch_command(StateCommand command) {
  switch (command) {
    case StateCommand::STAND_TOGGLE:
      handle_stand_command();
      return;
    case StateCommand::ZERO_VELOCITY:
      handle_zero_velocity();
      return;
    case StateCommand::WALK:
      stand_command_ = 1.0;
      base_height_command_ = desired_base_height_;
      log::info("ROS2 command: walk");
      return;
    case StateCommand::STAND:
      stand_command_ = 0.0;
      log::info("ROS2 command: stand");
      return;
    default:
      BasePolicy::dispatch_command(command);
  }
}

void LocomotionPolicy::handle_stand_command() {
  stand_command_ = 1.0 - stand_command_;
  if (stand_command_ == 0.0) {
    context_.velocity_input->zero();
    ang_vel_command_ = 0.0;
    lin_vel_command_[0] = 0.0;
    lin_vel_command_[1] = 0.0;
    log::info("Stance command", log::Color::kBlue);
  } else {
    base_height_command_ = desired_base_height_;
    log::info("Walk command", log::Color::kBlue);
  }
}

void LocomotionPolicy::handle_zero_velocity() {
  context_.velocity_input->zero();
  ang_vel_command_ = 0.0;
  lin_vel_command_[0] = 0.0;
  lin_vel_command_[1] = 0.0;
  log::info("Velocities set to zero", log::Color::kBlue);
}

void LocomotionPolicy::print_control_status() const {
  BasePolicy::print_control_status();
  char buf[128];
  std::snprintf(buf, sizeof(buf), "Linear velocity: x=%+.2f m/s, y=%+.2f m/s", lin_vel_command_[0],
                lin_vel_command_[1]);
  log::info(buf);
  std::snprintf(buf, sizeof(buf), "Angular velocity: %+.2f rad/s", ang_vel_command_);
  log::info(buf);
  log::info(walking() ? "Mode: Walking (applied)" : "Mode: Standing (not applied)");
  log::info("Terminal keys: W/A/S/D (lin) | Q/E (ang) | = (toggle mode)");
}

}  // namespace holosoma
