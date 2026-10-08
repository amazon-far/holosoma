#include "holosoma_cpp/timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace holosoma {

RateLimiter::RateLimiter(double frequency_hz)
    : period_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / frequency_hz))) {}

void RateLimiter::sleep() {
  const auto now = Clock::now();
  if (!next_) next_ = now + period_;
  if (*next_ <= now) {
    ++overruns_;
    next_ = now + period_;
    return;
  }
  std::this_thread::sleep_until(*next_);
  *next_ += period_;
}

LatencyTracker::LatencyTracker(int window_size) : window_(static_cast<size_t>(std::max(1, window_size))) {}

void LatencyTracker::start_cycle() {
  const auto now = std::chrono::steady_clock::now();
  if (last_cycle_start_) {
    const double dt = std::chrono::duration<double>(now - *last_cycle_start_).count();
    fps_.push_back(dt > 0 ? 1.0 / dt : 0.0);
    if (fps_.size() > window_) fps_.pop_front();
  }
  last_cycle_start_ = now;
}

void LatencyTracker::end_cycle() {
  if (last_cycle_start_) {
    record("total",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *last_cycle_start_).count());
  }
}

void LatencyTracker::record(const std::string& stage, double milliseconds) {
  auto& values = measurements_[stage];
  values.push_back(milliseconds);
  if (values.size() > window_) values.pop_front();
}

double LatencyTracker::fps() const {
  if (fps_.empty()) return 0.0;
  return std::accumulate(fps_.begin(), fps_.end(), 0.0) / static_cast<double>(fps_.size());
}

// Same stages, order and format as LatencyTracker.get_stats_str in utils/latency.py.
std::string LatencyTracker::stats_string() const {
  std::ostringstream os;
  bool first = true;
  for (const char* stage : {"read_state", "preprocessing", "inference", "postprocessing", "action_pub", "total"}) {
    const auto it = measurements_.find(stage);
    if (it == measurements_.end() || it->second.empty()) continue;
    const auto& values = it->second;
    const double n = static_cast<double>(values.size());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
    double var = 0.0;
    for (double v : values) var += (v - mean) * (v - mean);
    const double stdev = values.size() > 1 ? std::sqrt(var / (n - 1.0)) : 0.0;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s: %.3f±%.3fms", stage, mean, stdev);
    os << (first ? "" : " | ") << buf;
    first = false;
  }
  return os.str();
}

LatencyTracker::Scope::Scope(LatencyTracker& tracker, const char* stage)
    : tracker_(tracker), stage_(stage), start_(std::chrono::steady_clock::now()) {}

LatencyTracker::Scope::~Scope() {
  tracker_.record(stage_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count());
}

int64_t ClockSource::get_clock() {
  const int64_t last = source_();
  offset_ = std::min(offset_, last);
  return std::max<int64_t>(last - offset_, 0);
}

void ClockSource::reset_origin() { offset_ = source_(); }

void MotionClock::reset() {
  anchor_.reset();
  last_.reset();
  elapsed_at_anchor_ = 0;
  clock_.reset_origin();
}

int64_t MotionClock::elapsed_ms(bool* jumped_back) {
  const int64_t now = clock_.get_clock();
  if (!anchor_) anchor_ = now;
  if (last_ && now < *last_) {
    // Clock jumped backwards (e.g. simulator reset): re-anchor and keep the progress.
    elapsed_at_anchor_ += *last_ - *anchor_;
    anchor_ = now;
    if (jumped_back) *jumped_back = true;
  }
  last_ = now;
  return elapsed_at_anchor_ + (now - *anchor_);
}

TimestepUtil::TimestepUtil(MotionClock& clock, double interval_ms, int start_timestep)
    : clock_(clock), interval_ms_(interval_ms), start_timestep_(start_timestep), timestep_(start_timestep) {}

void TimestepUtil::reset(std::optional<int> start_timestep) {
  if (start_timestep) start_timestep_ = *start_timestep;
  timestep_ = start_timestep_;
  clock_.reset();
}

int TimestepUtil::get_timestep(bool* jumped) {
  const int64_t elapsed = clock_.elapsed_ms(jumped);
  // int(elapsed // interval_ms): floor division of an int by a float.
  const int elapsed_steps = static_cast<int>(std::floor(static_cast<double>(elapsed) / interval_ms_));
  if (timestep_ == start_timestep_ && elapsed_steps > 1) {
    // The clock jumped ahead before the first frame: start over from now.
    if (jumped) *jumped = true;
    clock_.reset();
    return timestep_;
  }
  timestep_ = elapsed_steps + start_timestep_;
  return timestep_;
}

}  // namespace holosoma
