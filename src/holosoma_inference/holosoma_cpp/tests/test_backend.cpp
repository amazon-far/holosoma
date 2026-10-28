// The TensorRT backend must agree with ONNX Runtime, tensor for tensor, on the same graph.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "holosoma_cpp/onnx_model.hpp"

namespace holosoma {
namespace {

std::string model(const char* name) { return std::string(HOLOSOMA_LIVE_FIXTURE_DIR) + "/" + name; }

// Deterministic, non-degenerate input: zeros would pass even with a transposed or truncated copy.
void fill(OnnxModel& m, const std::vector<TensorInfo>& inputs) {
  for (const auto& info : inputs) {
    auto& buffer = m.input(info.name);
    for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = std::sin(0.37f * static_cast<float>(i) + 0.11f);
  }
}

TEST(Backend, NamesRoundTrip) {
  for (auto backend : {Backend::kCpu, Backend::kCuda, Backend::kTensorRt}) {
    EXPECT_EQ(parse_backend(backend_name(backend)), backend);
  }
  EXPECT_EQ(parse_backend("trt"), Backend::kTensorRt);
  EXPECT_THROW(parse_backend("metal"), std::runtime_error);
}

TEST(Backend, CpuIsAlwaysAvailable) { EXPECT_TRUE(backend_available(Backend::kCpu)); }

TEST(Backend, NamingAnAbsentBackendFailsLoudly) {
  if (backend_available(Backend::kTensorRt)) GTEST_SKIP() << "this build has TensorRT";
  // The point is that it throws rather than quietly running on the CPU, which would look like a
  // working GPU deployment in every log.
  EXPECT_THROW(OnnxModel(model("pelvis.onnx"), Backend::kTensorRt), std::runtime_error);
}

TEST(Backend, TensorRtMatchesOnnxRuntime) {
  if (!backend_available(Backend::kTensorRt)) GTEST_SKIP() << "built without TensorRT";
  OnnxModel cpu(model("pelvis.onnx"), Backend::kCpu);
  OnnxModel trt(model("pelvis.onnx"), Backend::kTensorRt);
  EXPECT_STREQ(trt.provider(), "TensorRT");

  // The signature comes from the .onnx either way, so the two cannot disagree about it.
  ASSERT_EQ(cpu.inputs().size(), trt.inputs().size());
  ASSERT_EQ(cpu.outputs().size(), trt.outputs().size());
  EXPECT_EQ(cpu.metadata(), trt.metadata());

  fill(cpu, cpu.inputs());
  fill(trt, trt.inputs());
  cpu.run();
  trt.run();
  for (const auto& info : cpu.outputs()) {
    const auto& expected = cpu.output(info.name);
    const auto& actual = trt.output(info.name);
    ASSERT_EQ(expected.size(), actual.size()) << info.name;
    for (size_t i = 0; i < expected.size(); ++i) {
      // FP32 throughout, so this is kernel-selection noise rather than a precision change.
      EXPECT_NEAR(expected[i], actual[i], 1e-4f) << info.name << "[" << i << "]";
    }
  }
}

TEST(Backend, TensorRtReusesACachedEngine) {
  if (!backend_available(Backend::kTensorRt)) GTEST_SKIP() << "built without TensorRT";
  // Second construction hits the cache written by the first; both must produce the same engine.
  OnnxModel first(model("torso.onnx"), Backend::kTensorRt);
  OnnxModel second(model("torso.onnx"), Backend::kTensorRt);
  fill(first, first.inputs());
  fill(second, second.inputs());
  first.run();
  second.run();
  for (const auto& info : first.outputs()) {
    EXPECT_EQ(first.output(info.name), second.output(info.name)) << info.name;
  }
}

}  // namespace
}  // namespace holosoma
