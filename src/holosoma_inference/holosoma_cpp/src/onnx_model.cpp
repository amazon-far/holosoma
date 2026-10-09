#include "holosoma_cpp/onnx_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <stdexcept>

namespace holosoma {
namespace {

Ort::Env& ort_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "holosoma_inference");
  return env;
}

std::vector<int64_t> tensor_shape(const Ort::TypeInfo& info, const std::string& what) {
  if (info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw std::runtime_error(what + " is not a tensor");
  }
  const auto tensor = info.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(what + " must be float32");
  }
  return tensor.GetShape();
}

}  // namespace

int64_t TensorInfo::element_count() const {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d > 0 ? d : 1;
  }
  return count;
}

struct OnnxModel::Impl {
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::vector<Ort::Value> input_values;
  std::vector<Ort::Value> output_values;
  Ort::RunOptions run_options;
};

OnnxModel::OnnxModel(const std::string& path) : impl_(std::make_unique<Impl>()), path_(path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("ONNX model not found: " + path);
  }
  Ort::SessionOptions options;
  // A single thread: these MLPs are tiny and a pool only adds wake-up jitter.
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  try {
    impl_->session = std::make_unique<Ort::Session>(ort_env(), path.c_str(), options);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("Failed to load ONNX model " + path + ": " + e.what());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Session& session = *impl_->session;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    TensorInfo info;
    info.name = session.GetInputNameAllocated(i, allocator).get();
    info.shape = tensor_shape(session.GetInputTypeInfo(i), path + " input '" + info.name + "'");
    inputs_.push_back(info);
  }
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    TensorInfo info;
    info.name = session.GetOutputNameAllocated(i, allocator).get();
    info.shape = tensor_shape(session.GetOutputTypeInfo(i), path + " output '" + info.name + "'");
    outputs_.push_back(info);
  }

  const Ort::ModelMetadata meta = session.GetModelMetadata();
  for (const auto& key : meta.GetCustomMetadataMapKeysAllocated(allocator)) {
    const std::string k = key.get();
    metadata_[k] = meta.LookupCustomMetadataMapAllocated(k.c_str(), allocator).get();
  }

  // Preallocate every tensor once; run() reuses the same memory each tick.
  for (const auto& info : inputs_) {
    auto& buffer = input_buffers_[info.name];
    buffer.assign(static_cast<size_t>(info.element_count()), 0.0f);
    std::vector<int64_t> shape = info.shape;
    for (auto& d : shape) d = d > 0 ? d : 1;
    impl_->input_values.push_back(
        Ort::Value::CreateTensor<float>(impl_->memory_info, buffer.data(), buffer.size(), shape.data(), shape.size()));
  }
  for (const auto& info : outputs_) {
    auto& buffer = output_buffers_[info.name];
    buffer.assign(static_cast<size_t>(info.element_count()), 0.0f);
    std::vector<int64_t> shape = info.shape;
    for (auto& d : shape) d = d > 0 ? d : 1;
    impl_->output_values.push_back(
        Ort::Value::CreateTensor<float>(impl_->memory_info, buffer.data(), buffer.size(), shape.data(), shape.size()));
  }
  for (const auto& info : inputs_) impl_->input_names.push_back(info.name.c_str());
  for (const auto& info : outputs_) impl_->output_names.push_back(info.name.c_str());
}

OnnxModel::~OnnxModel() = default;

const TensorInfo* OnnxModel::find_input(const std::string& name) const {
  for (const auto& info : inputs_) {
    if (info.name == name) return &info;
  }
  return nullptr;
}

const TensorInfo* OnnxModel::find_output(const std::string& name) const {
  for (const auto& info : outputs_) {
    if (info.name == name) return &info;
  }
  return nullptr;
}

std::optional<std::string> OnnxModel::metadata_value(const std::string& key) const {
  const auto it = metadata_.find(key);
  if (it == metadata_.end()) return std::nullopt;
  return it->second;
}

std::vector<float>& OnnxModel::input(const std::string& name) {
  const auto it = input_buffers_.find(name);
  if (it == input_buffers_.end()) {
    throw std::runtime_error(path_ + " has no input named '" + name + "'");
  }
  return it->second;
}

void OnnxModel::run() {
  try {
    impl_->session->Run(impl_->run_options, impl_->input_names.data(), impl_->input_values.data(),
                        impl_->input_values.size(), impl_->output_names.data(), impl_->output_values.data(),
                        impl_->output_values.size());
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("ONNX inference failed for " + path_ + ": " + e.what());
  }
}

const std::vector<float>& OnnxModel::output(const std::string& name) const {
  const auto it = output_buffers_.find(name);
  if (it == output_buffers_.end()) {
    throw std::runtime_error(path_ + " has no output named '" + name + "'");
  }
  return it->second;
}

}  // namespace holosoma
