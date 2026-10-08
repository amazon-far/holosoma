// Keyboard repeat handling (sshkeyboard semantics) and the key-to-command maps.

#include <gtest/gtest.h>

#include "holosoma_cpp/input.hpp"

namespace holosoma {
namespace {

using Clock = KeyRepeatFilter::Clock;
using std::chrono::milliseconds;

TEST(KeyRepeatFilter, HeldKeyAutoRepeatCountsOnce) {
  KeyRepeatFilter filter;
  const auto t0 = Clock::now();
  EXPECT_TRUE(filter.on_key("w", t0));
  // Terminal auto-repeat: the first repeat after ~0.5 s, then every ~33 ms, for 2 s.
  for (int ms = 500; ms <= 2000; ms += 33) {
    filter.on_idle(t0 + milliseconds(ms - 10));
    EXPECT_FALSE(filter.on_key("w", t0 + milliseconds(ms))) << ms;
  }
  filter.on_idle(t0 + milliseconds(2100));  // released after 50 ms of silence
  EXPECT_TRUE(filter.on_key("w", t0 + milliseconds(2150)));
}

TEST(KeyRepeatFilter, QuickRepeatsWithinTheFirstDelayAreOnePress) {
  KeyRepeatFilter filter;
  const auto t0 = Clock::now();
  EXPECT_TRUE(filter.on_key("w", t0));
  filter.on_idle(t0 + milliseconds(300));
  EXPECT_FALSE(filter.on_key("w", t0 + milliseconds(400)));  // < 0.75 s since the press
  filter.on_idle(t0 + milliseconds(800));                    // > 0.75 s and quiet -> released
  EXPECT_TRUE(filter.on_key("w", t0 + milliseconds(900)));
}

TEST(KeyRepeatFilter, AnotherKeyReleasesTheHeldOne) {
  KeyRepeatFilter filter;
  const auto t0 = Clock::now();
  EXPECT_TRUE(filter.on_key("w", t0));
  EXPECT_TRUE(filter.on_key("a", t0 + milliseconds(50)));
  EXPECT_TRUE(filter.on_key("w", t0 + milliseconds(100)));
}

TEST(KeyboardInput, AppliesVelocityKeysAndCommands) {
  auto queue = std::make_shared<KeyQueue>();
  KeyboardInput keyboard(queue, true);
  for (const char* key : {"w", "w", "a", "q", "]", "m", "x", "3"}) queue->push(key);
  const auto vel = keyboard.poll_velocity();
  ASSERT_TRUE(vel);
  EXPECT_NEAR(vel->lin_vel_x, 0.2, 1e-12);
  EXPECT_NEAR(vel->lin_vel_y, 0.1, 1e-12);
  EXPECT_NEAR(vel->ang_vel, -0.1, 1e-12);
  EXPECT_EQ(keyboard.poll_commands(),
            (std::vector<StateCommand>{StateCommand::START, StateCommand::START_MOTION_CLIP, StateCommand::SWITCH_MODE,
                                       StateCommand::SWITCH_POLICY_3}));
  keyboard.zero();
  EXPECT_NEAR(keyboard.poll_velocity()->lin_vel_x, 0.0, 1e-12);
}

}  // namespace
}  // namespace holosoma
