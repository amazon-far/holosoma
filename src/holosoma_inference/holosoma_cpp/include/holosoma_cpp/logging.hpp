// Minimal thread-safe logger (timestamp, level, optional color on a TTY).
#pragma once

#include <sstream>
#include <string>

namespace holosoma::log {

enum class Level { kDebug = 0, kInfo = 1, kWarning = 2, kError = 3 };
enum class Color { kNone, kRed, kGreen, kYellow, kBlue, kMagenta };

void set_level(Level level);
Level level();
// Reads HOLOSOMA_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR).
void init_from_env();

void write(Level level, const std::string& message, Color color = Color::kNone);

inline void debug(const std::string& m) { write(Level::kDebug, m); }
inline void info(const std::string& m, Color c = Color::kNone) { write(Level::kInfo, m, c); }
inline void warning(const std::string& m) { write(Level::kWarning, m, Color::kYellow); }
inline void error(const std::string& m) { write(Level::kError, m, Color::kRed); }

template <typename... Args>
std::string cat(const Args&... args) {
  std::ostringstream os;
  (os << ... << args);
  return os.str();
}

}  // namespace holosoma::log
