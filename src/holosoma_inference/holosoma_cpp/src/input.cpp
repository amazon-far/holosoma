#include "holosoma_cpp/input.hpp"

#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cctype>
#include <cmath>

#include "holosoma_cpp/terminal.hpp"

namespace holosoma {

const std::map<std::string, StateCommand>& keyboard_commands() {
  static const std::map<std::string, StateCommand> kMap = {
      {"]", StateCommand::START},
      {"o", StateCommand::STOP},
      {"i", StateCommand::INIT},
      {"v", StateCommand::KP_DOWN_FINE},
      {"b", StateCommand::KP_UP_FINE},
      {"f", StateCommand::KP_DOWN},
      {"g", StateCommand::KP_UP},
      {"r", StateCommand::KP_RESET},
      {"=", StateCommand::STAND_TOGGLE},
      {"z", StateCommand::ZERO_VELOCITY},
      {"m", StateCommand::START_MOTION_CLIP},
      {"x", StateCommand::SWITCH_MODE},
      {"1", StateCommand::SWITCH_POLICY_1},
      {"2", StateCommand::SWITCH_POLICY_2},
      {"3", StateCommand::SWITCH_POLICY_3},
      {"4", StateCommand::SWITCH_POLICY_4},
      {"5", StateCommand::SWITCH_POLICY_5},
      {"6", StateCommand::SWITCH_POLICY_6},
      {"7", StateCommand::SWITCH_POLICY_7},
      {"8", StateCommand::SWITCH_POLICY_8},
      {"9", StateCommand::SWITCH_POLICY_9},
  };
  return kMap;
}

const std::map<std::string, std::tuple<int, int, double>>& keyboard_velocity_locomotion() {
  static const std::map<std::string, std::tuple<int, int, double>> kMap = {
      {"w", {0, 0, +0.1}}, {"s", {0, 0, -0.1}}, {"a", {0, 1, +0.1}},
      {"d", {0, 1, -0.1}}, {"q", {1, 0, -0.1}}, {"e", {1, 0, +0.1}},
  };
  return kMap;
}

const std::map<std::string, StateCommand>& joystick_commands() {
  static const std::map<std::string, StateCommand> kMap = {
      {"A", StateCommand::START},
      {"B", StateCommand::STOP},
      {"Y", StateCommand::INIT},
      {"up", StateCommand::KP_UP},
      {"down", StateCommand::KP_DOWN},
      {"left", StateCommand::KP_DOWN_FINE},
      {"right", StateCommand::KP_UP_FINE},
      {"F1", StateCommand::KP_RESET},
      {"select", StateCommand::NEXT_POLICY},
      {"L1+R1", StateCommand::KILL},
      {"start", StateCommand::STAND_TOGGLE},
      {"L2", StateCommand::ZERO_VELOCITY},
      {"select+A", StateCommand::START_MOTION_CLIP},
      {"X", StateCommand::SWITCH_MODE},
      {"x", StateCommand::SWITCH_MODE},
  };
  return kMap;
}

const std::map<uint16_t, std::string>& wireless_key_names() {
  static const std::map<uint16_t, std::string> kMap = {
      {1, "R1"},
      {2, "L1"},
      {3, "L1+R1"},
      {4, "start"},
      {8, "select"},
      {10, "L1+select"},
      {16, "R2"},
      {32, "L2"},
      {64, "F1"},
      {128, "F2"},
      {256, "A"},
      {264, "select+A"},
      {512, "B"},
      {520, "select+B"},
      {768, "A+B"},
      {1024, "X"},
      {1032, "select+X"},
      {1280, "A+X"},
      {1536, "B+X"},
      {2048, "Y"},
      {2304, "A+Y"},
      {2560, "B+Y"},
      {2056, "select+Y"},
      {3072, "X+Y"},
      {4096, "up"},
      {4097, "R1+up"},
      {4352, "A+up"},
      {4608, "B+up"},
      {4104, "select+up"},
      {5120, "X+up"},
      {6144, "Y+up"},
      {8192, "right"},
      {8193, "R1+right"},
      {8448, "A+right"},
      {9216, "X+right"},
      {10240, "Y+right"},
      {8200, "select+right"},
      {16384, "down"},
      {16392, "select+down"},
      {16385, "R1+down"},
      {16640, "A+down"},
      {16896, "B+down"},
      {17408, "X+down"},
      {18432, "Y+down"},
      {32768, "left"},
      {32769, "R1+left"},
      {32776, "select+left"},
      {33024, "A+left"},
      {33792, "X+left"},
      {34816, "Y+left"},
  };
  return kMap;
}

// ---------------------------------------------------------------------------
// KeyQueue / KeyboardListener
// ---------------------------------------------------------------------------

void KeyQueue::push(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  keys_.push_back(key);
}

std::optional<std::string> KeyQueue::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (keys_.empty()) return std::nullopt;
  std::string key = std::move(keys_.front());
  keys_.pop_front();
  return key;
}

KeyboardListener& KeyboardListener::instance() {
  static KeyboardListener listener;
  return listener;
}

KeyboardListener::~KeyboardListener() { stop(); }

bool KeyboardListener::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_) return true;
  if (!stdin_is_tty()) return false;
  TerminalMode::instance().enter_raw();
  stop_ = false;
  active_ = true;
  thread_ = std::thread(&KeyboardListener::run, this);
  return true;
}

void KeyboardListener::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  active_ = false;
  TerminalMode::instance().restore();
}

std::shared_ptr<KeyQueue> KeyboardListener::subscribe() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto queue = std::make_shared<KeyQueue>();
  subscribers_.push_back(queue);
  return queue;
}

bool KeyRepeatFilter::on_key(const std::string& key, Clock::time_point now) {
  const bool pressed = key != held_;
  if (pressed) {
    held_ = key;
    initial_press_ = now;
  }
  last_input_ = now;
  return pressed;
}

void KeyRepeatFilter::on_idle(Clock::time_point now) {
  if (!held_.empty() && now - initial_press_ > kDelaySecondChar && now - last_input_ > kDelayOtherChars) {
    held_.clear();
  }
}

void KeyboardListener::run() {
  enum class State { kNormal, kEscape, kSequence };
  State state = State::kNormal;
  std::string sequence;
  KeyRepeatFilter filter;
  // Keys that only take part in the repeat logic carry a leading '\x1b' and are never published.
  auto handle = [&](const std::string& key) {
    if (!filter.on_key(key, KeyRepeatFilter::Clock::now()) || key[0] == '\x1b') return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& queue : subscribers_) queue->push(key);
  };
  while (!stop_) {
    pollfd fd{STDIN_FILENO, POLLIN, 0};
    const int ready = ::poll(&fd, 1, 10);
    if (ready < 0) continue;
    if (ready == 0) {
      if (state != State::kNormal) handle("\x1b" + sequence);  // a lone ESC or a cut-off sequence
      state = State::kNormal;
      filter.on_idle(KeyRepeatFilter::Clock::now());
      continue;
    }
    unsigned char c = 0;
    if (::read(STDIN_FILENO, &c, 1) != 1) {
      if (!stop_) ::usleep(100000);  // stdin closed; keep the thread alive but idle
      continue;
    }
    if (state == State::kEscape) {
      sequence.push_back(static_cast<char>(c));
      state = (c == '[' || c == 'O') ? State::kSequence : State::kNormal;
      if (state == State::kNormal) handle("\x1b" + sequence);
      continue;
    }
    if (state == State::kSequence) {
      sequence.push_back(static_cast<char>(c));
      if (c >= 0x40 && c <= 0x7E) {  // final byte of a CSI/SS3 sequence
        state = State::kNormal;
        handle("\x1b" + sequence);
      }
      continue;
    }
    if (c == 0x1B) {
      state = State::kEscape;
      sequence.clear();
      continue;
    }
    if (c < 0x20 || c > 0x7E) {
      handle(std::string("\x1b") + static_cast<char>(c));  // enter, tab, backspace, ... are unmapped
      continue;
    }
    handle(std::string(1, static_cast<char>(std::tolower(c))));
  }
}

// ---------------------------------------------------------------------------
// KeyboardInput
// ---------------------------------------------------------------------------

KeyboardInput::KeyboardInput(std::shared_ptr<KeyQueue> queue, bool velocity_keys)
    : queue_(std::move(queue)), velocity_keys_(velocity_keys), mapping_(keyboard_commands()) {}

void KeyboardInput::drain() {
  while (auto key = queue_->pop()) {
    if (velocity_keys_) {
      const auto& vel = keyboard_velocity_locomotion();
      const auto it = vel.find(*key);
      if (it != vel.end()) {
        const auto& [array_idx, col, delta] = it->second;
        if (array_idx == 0) {
          lin_vel_[col] += delta;
        } else {
          ang_vel_ += delta;
        }
        continue;
      }
    }
    const auto cmd = mapping_.find(*key);
    if (cmd != mapping_.end()) pending_.push_back(cmd->second);
  }
}

std::optional<VelCmd> KeyboardInput::poll_velocity() {
  drain();
  if (!velocity_keys_) return std::nullopt;
  return VelCmd{lin_vel_[0], lin_vel_[1], ang_vel_};
}

void KeyboardInput::zero() {
  lin_vel_[0] = lin_vel_[1] = 0.0;
  ang_vel_ = 0.0;
}

std::vector<StateCommand> KeyboardInput::poll_commands() {
  drain();
  std::vector<StateCommand> out;
  out.swap(pending_);
  return out;
}

// ---------------------------------------------------------------------------
// InterfaceInput
// ---------------------------------------------------------------------------

InterfaceInput::InterfaceInput(RobotInterface& robot) : robot_(robot), mapping_(joystick_commands()) {}

std::optional<VelCmd> InterfaceInput::poll_velocity() {
  const auto msg = robot_.read_wireless_controller();
  if (!msg) return std::nullopt;
  if (msg->keys != 0) return std::nullopt;  // sticks are ignored while a button is held
  const double lx = msg->lx, ly = msg->ly, rx = msg->rx;
  VelCmd v;
  v.lin_vel_x = std::abs(ly) > kStickDeadzone ? ly : 0.0;
  v.lin_vel_y = std::abs(lx) > kStickDeadzone ? -lx : 0.0;
  v.ang_vel = std::abs(rx) > kStickDeadzone ? -rx : 0.0;
  return v;
}

std::vector<StateCommand> InterfaceInput::poll_commands() {
  const auto msg = robot_.read_wireless_controller();
  last_key_states_ = key_states_;
  if (msg) {
    const auto& names = wireless_key_names();
    const auto it = names.find(msg->keys);
    if (it != names.end()) {
      key_states_[it->second] = true;
    } else {
      for (auto& item : key_states_) item.second = false;
    }
  }
  std::vector<StateCommand> commands;
  for (const auto& [key, pressed] : key_states_) {
    const auto last = last_key_states_.find(key);
    const bool was_pressed = last != last_key_states_.end() && last->second;
    if (pressed && !was_pressed) {
      const auto cmd = mapping_.find(key);
      if (cmd != mapping_.end()) commands.push_back(cmd->second);
    }
  }
  return commands;
}

}  // namespace holosoma
