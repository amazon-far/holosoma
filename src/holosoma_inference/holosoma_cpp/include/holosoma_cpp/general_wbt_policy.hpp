#pragma once

#include <array>
#include <optional>

#include "holosoma_cpp/holosoma_inference_core.hpp"
#include "holosoma_cpp/policy.hpp"

namespace holosoma {

constexpr int kNumOffsets = 5;  // The 0p2s preset; other exports declare their own offsets.

struct ReferenceFrame {
  std::vector<double> q;
  std::vector<double> dq;
  std::array<double, 4> quat{1, 0, 0, 0};  // wxyz
};

// Offset-major, canonical joint order. The caller keeps this alive until it is replaced.
// For exports without lookahead, supply one frame. For legacy exports whose current
// reference differs from the first future sample, set current explicitly.
struct ReferenceWindow {
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<std::array<double, 4>> quat;
  std::optional<ReferenceFrame> current;
};

// The public policy adapter uses the same metadata-driven implementation as live inference.
// Observation layout, history, gains, default pose and anchor FK come from the checkpoint.
class GeneralWBTPolicy : public BasePolicy {
 public:
  GeneralWBTPolicy(InferenceConfig config, PolicyContext& context);
  // Buffer lifetime only; a new clip/heading needs reset() before inference.
  void set_reference(const ReferenceWindow* window) { reference_ = window; }
  void handle_start_policy() override;
  std::vector<double> infer_observation(const std::vector<float>& obs) override;
  live::WbtPolicy& core();
  const live::WbtPolicy& core() const;

 protected:
  void setup_policy(ModelSlot& slot) override;
  std::set<std::string> available_terms() const override;
  void on_policy_switched() override;
  std::vector<double> rl_inference(const RobotState& state) override;

 private:
  const ReferenceWindow* reference_ = nullptr;
  std::vector<std::unique_ptr<live::WbtPolicy>> cores_;
};

// Build an inference-only configuration for a metadata-complete G1 checkpoint.
InferenceConfig live_policy_config(const std::string& checkpoint, bool use_gpu = false, int intra_op_threads = 2);

}  // namespace holosoma
