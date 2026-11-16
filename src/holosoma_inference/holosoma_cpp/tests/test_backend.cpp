// The TensorRT and ExecuTorch backends must agree with ONNX Runtime, tensor for tensor, on the same graph.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
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
  for (auto backend : {Backend::kCpu, Backend::kCuda, Backend::kTensorRt, Backend::kExecuTorch}) {
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

TEST(Backend, NamingAbsentExecuTorchFailsLoudly) {
  if (backend_available(Backend::kExecuTorch)) GTEST_SKIP() << "this build has ExecuTorch";
  EXPECT_THROW(OnnxModel(model("pelvis.onnx"), Backend::kExecuTorch), std::runtime_error);
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

#ifdef HOLOSOMA_EXECUTORCH_FIXTURE_DIR
std::string executorch_model(const char* name) { return std::string(HOLOSOMA_EXECUTORCH_FIXTURE_DIR) + "/" + name; }

// policy_*.pte hold the same weights as policy.onnx. The .pte brings its own names and metadata,
// so they are compared too: a policy reads its gains and joint order from there.
void expect_matches_onnx_runtime(const std::string& program, float tolerance) {
  OnnxModel ort(executorch_model("policy.onnx"), Backend::kCpu);
  OnnxModel et(program, Backend::kExecuTorch);
  ASSERT_EQ(ort.inputs().size(), et.inputs().size());
  for (size_t i = 0; i < ort.inputs().size(); ++i) {
    EXPECT_EQ(ort.inputs()[i].name, et.inputs()[i].name);
    EXPECT_EQ(ort.inputs()[i].shape, et.inputs()[i].shape);
  }
  ASSERT_EQ(ort.outputs().size(), et.outputs().size());
  EXPECT_EQ(ort.metadata(), et.metadata());

  fill(ort, ort.inputs());
  fill(et, et.inputs());
  ort.run();
  et.run();
  for (const auto& info : ort.outputs()) {
    const auto& expected = ort.output(info.name);
    const auto& actual = et.output(info.name);
    ASSERT_EQ(expected.size(), actual.size()) << info.name;
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance) << info.name << "[" << i << "]";
    }
  }
}

TEST(Backend, ExecuTorchXnnpackMatchesOnnxRuntime) {
  expect_matches_onnx_runtime(executorch_model("policy_xnnpack.pte"), 1e-5f);
}

TEST(Backend, ExecuTorchTensorRtMatchesOnnxRuntime) {
#ifndef HOLOSOMA_EXECUTORCH_TENSORRT
  GTEST_SKIP() << "built without ExecuTorch's TensorRT delegate";
#endif
  const std::string program = executorch_model("policy_tensorrt.pte");
  // The fixture is only written when a GPU was visible at build time.
  if (!std::filesystem::exists(program)) GTEST_SKIP() << "no GPU when the fixtures were exported";
  EXPECT_NE(std::string(OnnxModel(program, Backend::kExecuTorch).provider()).find("TensorRTBackend"),
            std::string::npos);
  expect_matches_onnx_runtime(program, 1e-4f);
}

// The message is checked, not just the throw: any load failure would also throw.
void expect_refused(const std::string& path, const std::string& reason) {
  try {
    OnnxModel model(path, Backend::kExecuTorch);
    ADD_FAILURE() << path << " was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
  }
}

TEST(Backend, ExecuTorchRefusesAProgramWithoutASignature) {
  expect_refused(executorch_model("no_signature.pte"), "has no 'signature' method");
}

// runtime.backend covers the secondary policy too, which the shipped presets keep as an .onnx.
TEST(Backend, ExecuTorchRefusesAnOnnxModel) { expect_refused(executorch_model("policy.onnx"), "is not one"); }
#endif

}  // namespace
}  // namespace holosoma
