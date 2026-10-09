// Loop timing, latency statistics and the WBT motion clock.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>

namespace holosoma {

// Fixed-rate scheduler with absolute deadlines (utils/rate.py PreciseRateLimiter):
// an overrun restarts the schedule instead of bursting to catch up.
class RateLimiter {
 public:
  explicit RateLimiter(double frequency_hz);
  void sleep();
  uint64_t overruns() const { return overruns_; }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::duration period_;
  std::optional<Clock::time_point> next_;
  uint64_t overruns_ = 0;
};

// Per-stage latency statistics over a sliding window (utils/latency.py).
class LatencyTracker {
 public:
  explicit LatencyTracker(int window_size);
  void start_cycle();
  void end_cycle();  // records the "total" stage (cycle compute time, excluding the sleep)
  void record(const std::string& stage, double milliseconds);
  double fps() const;
  std::string stats_string() const;

  class Scope {
   public:
    Scope(LatencyTracker& tracker, const char* stage);
    ~Scope();

   private:
    LatencyTracker& tracker_;
    const char* stage_;
    std::chrono::steady_clock::time_point start_;
  };

 private:
  size_t window_;
  std::map<std::string, std::deque<double>> measurements_;
  std::deque<double> fps_;
  std::optional<std::chrono::steady_clock::time_point> last_cycle_start_;
};

// Millisecond clock with a resettable origin (utils/clock.py ClockSub semantics).
// `source` returns the latest raw clock value; the simulator bridge publishes sim time in LowState.tick.
class ClockSource {
 public:
  explicit ClockSource(std::function<int64_t()> source) : source_(std::move(source)) {}
  int64_t get_clock();
  void reset_origin();

 private:
  std::function<int64_t()> source_;
  int64_t offset_ = 0;
};

// Elapsed milliseconds since reset, robust to backward clock jumps (wbt_utils.MotionClockUtil).
class MotionClock {
 public:
  explicit MotionClock(ClockSource& clock) : clock_(clock) {}
  void reset();
  int64_t elapsed_ms(bool* jumped_back = nullptr);

 private:
  ClockSource& clock_;
  std::optional<int64_t> anchor_;
  std::optional<int64_t> last_;
  int64_t elapsed_at_anchor_ = 0;
};

// Converts elapsed time into motion frames (wbt_utils.TimestepUtil).
class TimestepUtil {
 public:
  TimestepUtil(MotionClock& clock, double interval_ms, int start_timestep);
  void reset(std::optional<int> start_timestep = std::nullopt);
  int get_timestep(bool* jumped = nullptr);
  int timestep() const { return timestep_; }

 private:
  MotionClock& clock_;
  double interval_ms_;
  int start_timestep_;
  int timestep_;
};

}  // namespace holosoma
