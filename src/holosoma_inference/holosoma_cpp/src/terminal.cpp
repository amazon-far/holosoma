#include "holosoma_cpp/terminal.hpp"

#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

namespace holosoma {
namespace {

termios g_saved{};
std::atomic<bool> g_raw{false};

// Async-signal-safe and idempotent; registered with atexit().
void restore_saved_terminal() {
  if (g_raw.exchange(false)) {
    ::tcsetattr(STDIN_FILENO, TCSANOW, &g_saved);
  }
}

}  // namespace

bool stdin_is_tty() { return ::isatty(STDIN_FILENO) == 1; }
bool stdout_is_tty() { return ::isatty(STDOUT_FILENO) == 1; }

TerminalMode& TerminalMode::instance() {
  static TerminalMode mode;
  return mode;
}

void TerminalMode::enter_raw() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (g_raw || !stdin_is_tty()) return;
  if (::tcgetattr(STDIN_FILENO, &g_saved) != 0) return;
  termios raw = g_saved;
  raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) {
    g_raw = true;
    static bool registered = false;
    if (!registered) {
      std::atexit(restore_saved_terminal);
      registered = true;
    }
  }
}

void TerminalMode::restore() {
  std::lock_guard<std::mutex> lock(mutex_);
  restore_saved_terminal();
}

bool wait_for_enter() {
  if (!stdin_is_tty()) return false;
  std::string line;
  return static_cast<bool>(std::getline(std::cin, line));
}

}  // namespace holosoma
