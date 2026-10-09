#include "holosoma_cpp/unitree_interface.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "holosoma_cpp/logging.hpp"

#if HOLOSOMA_WITH_UNITREE_SDK
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#endif

namespace holosoma {

std::string detect_robot_interface() {
  namespace fs = std::filesystem;
  static const char* kSkip[] = {"lo", "wl", "docker", "br-", "veth", "virbr", "vnet", "tun", "tap"};
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/sys/class/net", ec)) {
    names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  for (const auto& name : names) {
    if (std::any_of(std::begin(kSkip), std::end(kSkip), [&](const char* p) { return name.rfind(p, 0) == 0; })) {
      continue;
    }
    std::ifstream f("/sys/class/net/" + name + "/operstate");
    std::string state;
    if (f >> state && (state == "up" || state == "UP")) {
      log::info("[network] auto-detected interface: " + name);
      return name;
    }
  }
  log::info("[network] no wired NIC found, falling back to loopback (lo)");
  return "lo";
}

#if HOLOSOMA_WITH_UNITREE_SDK

namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using WirelessControllerMsg = unitree_go::msg::dds_::WirelessController_;

constexpr const char* kLowCmdTopic = "rt/lowcmd";
constexpr const char* kLowStateTopic = "rt/lowstate";
constexpr const char* kWirelessTopic = "rt/wirelesscontroller";
constexpr size_t kHgMotorSlots = 35;
constexpr auto kWriterPeriod = std::chrono::microseconds(2000);  // 500 Hz, as the Python binding

// Unitree's message checksum (CRC-32, polynomial 0x04C11DB7, over 32-bit words).
uint32_t crc32_core(const uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  const uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1u << 31;
    const uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (crc & 0x80000000) {
        crc = (crc << 1) ^ polynomial;
      } else {
        crc <<= 1;
      }
      if (data & xbit) crc ^= polynomial;
      xbit >>= 1;
    }
  }
  return crc;
}

template <typename Msg>
uint32_t message_crc(const Msg& msg) {
  return crc32_core(reinterpret_cast<const uint32_t*>(&msg), (sizeof(Msg) >> 2) - 1);
}

class UnitreeInterface final : public RobotInterface {
 public:
  UnitreeInterface(const RobotConfig& robot, int domain_id, const std::string& nic)
      : nic_(nic), joint2motor_(robot.joint2motor), num_joints_(robot.num_joints), num_motors_(robot.num_motors) {
    if (robot.sdk_type != "unitree") {
      throw std::runtime_error("The C++ runtime only supports sdk_type 'unitree' (got '" + robot.sdk_type + "')");
    }
    if (robot.message_type != "HG") {
      throw std::runtime_error("The C++ runtime only supports HG messages (G1/H1-2), got '" + robot.message_type + "'");
    }
    if (static_cast<size_t>(num_motors_) > kHgMotorSlots) {
      throw std::runtime_error("HG LowCmd carries at most 35 motors");
    }

    unitree::robot::ChannelFactory::Instance()->Init(domain_id, nic_);
    low_cmd_pub_ = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(kLowCmdTopic);
    low_cmd_pub_->InitChannel();
    low_state_sub_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(kLowStateTopic);
    low_state_sub_->InitChannel([this](const void* msg) { on_low_state(*static_cast<const LowState*>(msg)); }, 1);
    wireless_sub_ = std::make_shared<unitree::robot::ChannelSubscriber<WirelessControllerMsg>>(kWirelessTopic);
    wireless_sub_->InitChannel(
        [this](const void* msg) { on_wireless(*static_cast<const WirelessControllerMsg*>(msg)); }, 1);
    writer_ = std::thread(&UnitreeInterface::writer_loop, this);
    log::info(
        log::cat("UnitreeInterface initialized: ", num_motors_, " motors (HG) on ", nic_, ", DDS domain ", domain_id));
  }

  ~UnitreeInterface() override {
    stop_ = true;
    if (writer_.joinable()) writer_.join();
    wireless_sub_.reset();
    low_state_sub_.reset();
    low_cmd_pub_.reset();
  }

  bool read_state(RobotState& state) override {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!have_state_) return false;
    state = state_;
    return true;
  }

  void send_command(const JointCommand& command) override {
    LowCmd msg;
    msg.mode_pr() = 0;  // PR (pitch/roll) ankle mode
    for (size_t m = 0; m < kHgMotorSlots; ++m) {
      auto& motor = msg.motor_cmd()[m];
      motor.mode() = static_cast<size_t>(m) < static_cast<size_t>(num_motors_) ? 1 : 0;
      motor.q() = 0.0f;
      motor.dq() = 0.0f;
      motor.tau() = 0.0f;
      motor.kp() = 0.0f;
      motor.kd() = 0.0f;
    }
    for (int j = 0; j < num_joints_; ++j) {
      auto& motor = msg.motor_cmd()[static_cast<size_t>(joint2motor_[static_cast<size_t>(j)])];
      const auto i = static_cast<size_t>(j);
      motor.q() = static_cast<float>(command.q[i]);
      motor.dq() = static_cast<float>(command.dq[i]);
      motor.tau() = static_cast<float>(command.tau[i]);
      motor.kp() = static_cast<float>(command.kp[i]);
      motor.kd() = static_cast<float>(command.kd[i]);
    }
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    pending_cmd_ = msg;
    have_cmd_ = true;
  }

  std::optional<WirelessController> read_wireless_controller() override {
    // Like the Python binding: a zeroed sample until the first message arrives.
    std::lock_guard<std::mutex> lock(wireless_mutex_);
    return wireless_;
  }

  std::string description() const override { return "Unitree SDK2 (DDS) on " + nic_; }

 private:
  void on_low_state(const LowState& msg) {
    if (msg.crc() != message_crc(msg)) {
      if (++crc_errors_ % 100 == 1) log::error("low_state CRC error (HG)");
      return;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.dof_pos.resize(static_cast<size_t>(num_joints_));
    state_.dof_vel.resize(static_cast<size_t>(num_joints_));
    for (int j = 0; j < num_joints_; ++j) {
      const auto& motor = msg.motor_state()[static_cast<size_t>(joint2motor_[static_cast<size_t>(j)])];
      state_.dof_pos[static_cast<size_t>(j)] = motor.q();
      state_.dof_vel[static_cast<size_t>(j)] = motor.dq();
    }
    const auto& imu = msg.imu_state();
    state_.base_quat_wxyz = {imu.quaternion()[0], imu.quaternion()[1], imu.quaternion()[2], imu.quaternion()[3]};
    state_.base_ang_vel = {imu.gyroscope()[0], imu.gyroscope()[1], imu.gyroscope()[2]};
    state_.tick_ms = msg.tick();
    state_.sequence += 1;
    state_.stamp = SteadyClock::now();
    have_state_ = true;
    mode_machine_ = msg.mode_machine();
  }

  void on_wireless(const WirelessControllerMsg& msg) {
    std::lock_guard<std::mutex> lock(wireless_mutex_);
    wireless_.lx = msg.lx();
    wireless_.ly = msg.ly();
    wireless_.rx = msg.rx();
    wireless_.ry = msg.ry();
    wireless_.keys = msg.keys();
  }

  void writer_loop() {
    auto next = std::chrono::steady_clock::now();
    while (!stop_) {
      next += kWriterPeriod;
      LowCmd msg;
      bool have = false;
      {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (have_cmd_) {
          msg = pending_cmd_;
          have = true;
        }
      }
      if (have) {
        msg.mode_machine() = mode_machine_.load();
        msg.crc() = message_crc(msg);
        low_cmd_pub_->Write(msg);
      }
      std::this_thread::sleep_until(next);
    }
  }

  std::string nic_;
  std::vector<int> joint2motor_;
  int num_joints_;
  int num_motors_;

  std::shared_ptr<unitree::robot::ChannelPublisher<LowCmd>> low_cmd_pub_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> low_state_sub_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<WirelessControllerMsg>> wireless_sub_;

  std::mutex state_mutex_;
  RobotState state_;
  bool have_state_ = false;
  std::atomic<uint8_t> mode_machine_{0};
  uint64_t crc_errors_ = 0;

  std::mutex cmd_mutex_;
  LowCmd pending_cmd_;
  bool have_cmd_ = false;

  std::mutex wireless_mutex_;
  WirelessController wireless_;

  std::atomic<bool> stop_{false};
  std::thread writer_;
};

}  // namespace

bool unitree_sdk_available() { return true; }

std::unique_ptr<RobotInterface> make_unitree_interface(const RobotConfig& robot, int domain_id,
                                                       const std::string& network_interface) {
  const std::string nic = network_interface == "auto" ? detect_robot_interface() : network_interface;
  return std::make_unique<UnitreeInterface>(robot, domain_id, nic);
}

#else

bool unitree_sdk_available() { return false; }

std::unique_ptr<RobotInterface> make_unitree_interface(const RobotConfig&, int, const std::string&) {
  throw std::runtime_error(
      "This binary was built without Unitree SDK2 (HOLOSOMA_WITH_UNITREE_SDK=OFF); rebuild on Linux with the SDK to "
      "talk to a robot or the simulator bridge.");
}

#endif

}  // namespace holosoma
