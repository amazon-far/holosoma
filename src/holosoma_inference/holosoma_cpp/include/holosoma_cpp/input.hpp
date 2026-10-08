// Operator input providers (holosoma_inference/inputs).
//
// KeyboardInput reads the controlling terminal, InterfaceInput reads the
// Unitree wireless controller through the RobotInterface. Key maps and edge
// detection follow the Python providers exactly.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "holosoma_cpp/commands.hpp"
#include "holosoma_cpp/robot_interface.hpp"

namespace holosoma {

class VelocityProvider {
 public:
  virtual ~VelocityProvider() = default;
  virtual std::optional<VelCmd> poll_velocity() = 0;
  virtual void zero() = 0;
};

class CommandProvider {
 public:
  virtual ~CommandProvider() = default;
  virtual std::vector<StateCommand> poll_commands() = 0;
  // Adds (or replaces) a key binding; used by dual mode to bind its switch key.
  virtual void bind(const std::string& key, StateCommand command) = 0;
};

// keyboard.py KEYBOARD_COMMANDS / KEYBOARD_VELOCITY_LOCOMOTION.
const std::map<std::string, StateCommand>& keyboard_commands();
// key -> (array index: 0 = lin_vel, 1 = ang_vel, column, delta)
const std::map<std::string, std::tuple<int, int, double>>& keyboard_velocity_locomotion();

// joystick.py JOYSTICK_COMMANDS and BaseInterface._default_wc_key_map.
const std::map<std::string, StateCommand>& joystick_commands();
const std::map<uint16_t, std::string>& wireless_key_names();

// Thread-safe queue of key names; the terminal listener (or a test) pushes into it.
class KeyQueue {
 public:
  void push(const std::string& key);
  std::optional<std::string> pop();

 private:
  std::mutex mutex_;
  std::deque<std::string> keys_;
};

// Turns terminal input into key presses the way sshkeyboard (the Python runtime's
// listener) does: a held key's auto-repeat counts as one press, and the key is
// released once kDelaySecondChar has passed since the press and the input has
// been quiet for kDelayOtherChars. Pressing another key releases the held one.
class KeyRepeatFilter {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds kDelaySecondChar{750};
  static constexpr std::chrono::milliseconds kDelayOtherChars{50};

  // Returns true when `key` is a new press.
  bool on_key(const std::string& key, Clock::time_point now);
  // Call when no input arrived; may release the held key.
  void on_idle(Clock::time_point now);

 private:
  std::string held_;
  Clock::time_point initial_press_{};
  Clock::time_point last_input_{};
};

// Broadcasts key presses from stdin to subscribed queues. Letters are lower-cased
// (as sshkeyboard does). Escape sequences (arrows, function keys) take part in the
// repeat logic but are never published, so their bytes cannot alias a command key.
class KeyboardListener {
 public:
  static KeyboardListener& instance();

  // Starts the reader thread. Returns false when stdin is not a TTY.
  bool start();
  void stop();
  bool active() const { return active_; }
  std::shared_ptr<KeyQueue> subscribe();

 private:
  KeyboardListener() = default;
  ~KeyboardListener();
  void run();

  std::mutex mutex_;
  std::vector<std::shared_ptr<KeyQueue>> subscribers_;
  std::thread thread_;
  std::atomic<bool> active_{false};
  std::atomic<bool> stop_{false};
};

// KeyboardInput: velocity increments + discrete commands from one key queue.
class KeyboardInput : public VelocityProvider, public CommandProvider {
 public:
  KeyboardInput(std::shared_ptr<KeyQueue> queue, bool velocity_keys);

  std::optional<VelCmd> poll_velocity() override;
  void zero() override;
  std::vector<StateCommand> poll_commands() override;
  void bind(const std::string& key, StateCommand command) override { mapping_[key] = command; }

 private:
  void drain();

  std::shared_ptr<KeyQueue> queue_;
  bool velocity_keys_;
  std::map<std::string, StateCommand> mapping_;
  double lin_vel_[2] = {0.0, 0.0};
  double ang_vel_ = 0.0;
  std::vector<StateCommand> pending_;
};

// InterfaceInput: sticks and buttons of the robot's wireless controller.
class InterfaceInput : public VelocityProvider, public CommandProvider {
 public:
  static constexpr double kStickDeadzone = 0.1;

  explicit InterfaceInput(RobotInterface& robot);

  std::optional<VelCmd> poll_velocity() override;
  void zero() override {}
  std::vector<StateCommand> poll_commands() override;
  void bind(const std::string& key, StateCommand command) override { mapping_[key] = command; }

 private:
  RobotInterface& robot_;
  std::map<std::string, StateCommand> mapping_;
  std::map<std::string, bool> key_states_;
  std::map<std::string, bool> last_key_states_;
};

}  // namespace holosoma
