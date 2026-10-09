#include "holosoma_cpp/logging.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace holosoma::log {
namespace {

std::atomic<int> g_level{static_cast<int>(Level::kInfo)};
std::mutex g_mutex;

const char* level_name(Level level) {
  switch (level) {
    case Level::kDebug:
      return "DEBUG";
    case Level::kInfo:
      return "INFO";
    case Level::kWarning:
      return "WARNING";
    case Level::kError:
      return "ERROR";
  }
  return "INFO";
}

const char* color_code(Color color) {
  switch (color) {
    case Color::kRed:
      return "\033[31m";
    case Color::kGreen:
      return "\033[32m";
    case Color::kYellow:
      return "\033[33m";
    case Color::kBlue:
      return "\033[34m";
    case Color::kMagenta:
      return "\033[35m";
    case Color::kNone:
      break;
  }
  return "";
}

}  // namespace

void set_level(Level level) { g_level = static_cast<int>(level); }

Level level() { return static_cast<Level>(g_level.load()); }

void init_from_env() {
  const char* env = std::getenv("HOLOSOMA_LOG_LEVEL");
  if (env == nullptr) return;
  const std::string v(env);
  if (v == "DEBUG") set_level(Level::kDebug);
  if (v == "INFO") set_level(Level::kInfo);
  if (v == "WARNING") set_level(Level::kWarning);
  if (v == "ERROR") set_level(Level::kError);
}

void write(Level lvl, const std::string& message, Color color) {
  if (static_cast<int>(lvl) < g_level.load()) return;
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm tm{};
  localtime_r(&t, &tm);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &tm);

  std::lock_guard<std::mutex> lock(g_mutex);
  FILE* out = lvl >= Level::kWarning ? stderr : stdout;
  const bool tty = ::isatty(fileno(out)) == 1;
  const bool colored = tty && color != Color::kNone;
  // A raw-mode terminal does not translate "\n" into "\r\n".
  std::fprintf(out, "%s.%03d | %-7s | %s%s%s\r\n", stamp, static_cast<int>(ms), level_name(lvl),
               colored ? color_code(color) : "", message.c_str(), colored ? "\033[0m" : "");
  std::fflush(out);
}

}  // namespace holosoma::log
