// Controlling-terminal helpers: single-key input mode and guaranteed restore.
#pragma once

#include <mutex>

namespace holosoma {

bool stdin_is_tty();
bool stdout_is_tty();

// Puts the terminal into non-canonical, no-echo mode (Ctrl-C still raises SIGINT)
// and restores the original settings on restore(), at exit, or on a fatal signal.
class TerminalMode {
 public:
  static TerminalMode& instance();
  void enter_raw();
  void restore();

 private:
  TerminalMode() = default;
  std::mutex mutex_;
};

// Blocks until the operator presses Enter. Returns false on EOF / non-TTY.
bool wait_for_enter();

}  // namespace holosoma
