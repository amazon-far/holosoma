#include "holosoma_cpp/onnx_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "holosoma_cpp/cuda_provider.hpp"
#include "holosoma_cpp/model_signature.hpp"

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

ModelSignature describe(Ort::Session& session, const std::string& path) {
  ModelSignature signature;
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    TensorInfo info;
    info.name = session.GetInputNameAllocated(i, allocator).get();
    info.shape = tensor_shape(session.GetInputTypeInfo(i), path + " input '" + info.name + "'");
    signature.inputs.push_back(info);
  }
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    TensorInfo info;
    info.name = session.GetOutputNameAllocated(i, allocator).get();
    info.shape = tensor_shape(session.GetOutputTypeInfo(i), path + " output '" + info.name + "'");
    signature.outputs.push_back(info);
  }
  const Ort::ModelMetadata meta = session.GetModelMetadata();
  for (const auto& key : meta.GetCustomMetadataMapKeysAllocated(allocator)) {
    const std::string k = key.get();
    signature.metadata[k] = meta.LookupCustomMetadataMapAllocated(k.c_str(), allocator).get();
  }
  return signature;
}

// ONNX Runtime, on the CPU or on its CUDA execution provider.
class OrtBackend final : public ModelBackend {
 public:
  OrtBackend(const std::string& path, Backend backend, int intra_op_threads, const std::string& profile_prefix,
             ModelSignature& signature)
      : path_(path), backend_(backend), profiling_(!profile_prefix.empty()) {
    Ort::SessionOptions options;
    if (profiling_) options.EnableProfiling(profile_prefix.c_str());
    // A single thread: these MLPs are tiny and a pool only adds wake-up jitter.
    options.SetIntraOpNumThreads(intra_op_threads > 0 ? intra_op_threads : 1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
      if (backend == Backend::kCuda) enable_cuda(options);
      session_ = std::make_unique<Ort::Session>(ort_env(), path.c_str(), options);
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("Failed to load ONNX model " + path + ": " + e.what());
    }
    signature = describe(*session_, path);
  }

  void bind(const std::vector<std::pair<const TensorInfo*, float*>>& inputs,
            const std::vector<std::pair<const TensorInfo*, float*>>& outputs) override {
    const auto wrap = [this](const auto& tensors, std::vector<Ort::Value>& values, std::vector<const char*>& names) {
      for (const auto& [info, data] : tensors) {
        std::vector<int64_t> shape = info->shape;
        for (auto& d : shape) d = d > 0 ? d : 1;
        values.push_back(Ort::Value::CreateTensor<float>(memory_info_, data, static_cast<size_t>(info->element_count()),
                                                         shape.data(), shape.size()));
        names.push_back(info->name.c_str());
      }
    };
    wrap(inputs, input_values_, input_names_);
    wrap(outputs, output_values_, output_names_);
  }

  void run() override {
    try {
      session_->Run(run_options_, input_names_.data(), input_values_.data(), input_values_.size(), output_names_.data(),
                    output_values_.data(), output_values_.size());
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("ONNX inference failed for " + path_ + ": " + e.what());
    }
  }

  // What the session actually chose, not what was asked for: a CUDA request that silently lands
  // on the CPU is exactly the failure this is here to make visible.
  const char* provider() const override {
    if (backend_ != Backend::kCuda) return "CPUExecutionProvider";
    static const bool cuda = [] {
      const auto providers = Ort::GetAvailableProviders();
      return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
    }();
    return cuda ? "CUDAExecutionProvider" : "CPUExecutionProvider";
  }

  std::string finish_profile() {
    if (!profiling_) throw std::logic_error("Profiling was not enabled");
    Ort::AllocatorWithDefaultOptions allocator;
    return session_->EndProfilingAllocated(allocator).get();
  }

 private:
  std::string path_;
  Backend backend_;
  bool profiling_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<Ort::Value> input_values_;
  std::vector<Ort::Value> output_values_;
  Ort::RunOptions run_options_;
};

}  // namespace

int64_t TensorInfo::element_count() const {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d > 0 ? d : 1;
  }
  return count;
}

bool backend_available(Backend backend) {
  switch (backend) {
    case Backend::kTensorRt:
#ifdef HOLOSOMA_WITH_TENSORRT
      return true;
#else
      return false;
#endif
    case Backend::kCuda: {
      const auto providers = Ort::GetAvailableProviders();
      return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
    }
    default:
      return true;
  }
}

ModelSignature read_onnx_signature(const std::string& path) {
  ModelSignature signature;
  OrtBackend(path, Backend::kCpu, 1, "", signature);
  return signature;
}

OnnxModel::OnnxModel(const std::string& path, Backend backend, int intra_op_threads, const std::string& profile_prefix)
    : path_(path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("ONNX model not found: " + path);
  }
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("inference backend '") + backend_name(backend) +
                             "' is not available in this build");
  }
  ModelSignature signature;
  if (backend == Backend::kTensorRt) {
#ifdef HOLOSOMA_WITH_TENSORRT
    signature = read_onnx_signature(path);
    if (!profile_prefix.empty()) throw std::runtime_error("ONNX Runtime profiling needs the cpu or cuda backend");
    backend_ = make_tensorrt_backend(path, signature);
#endif
  } else {
    backend_ = std::make_unique<OrtBackend>(path, backend, intra_op_threads, profile_prefix, signature);
  }
  inputs_ = std::move(signature.inputs);
  outputs_ = std::move(signature.outputs);
  metadata_ = std::move(signature.metadata);

  // Preallocate every tensor once; run() reuses the same memory each tick.
  std::vector<std::pair<const TensorInfo*, float*>> input_views;
  std::vector<std::pair<const TensorInfo*, float*>> output_views;
  for (const auto& info : inputs_) {
    auto& buffer = input_buffers_[info.name];
    buffer.assign(static_cast<size_t>(info.element_count()), 0.0f);
    input_views.emplace_back(&info, buffer.data());
  }
  for (const auto& info : outputs_) {
    auto& buffer = output_buffers_[info.name];
    buffer.assign(static_cast<size_t>(info.element_count()), 0.0f);
    output_views.emplace_back(&info, buffer.data());
  }
  backend_->bind(input_views, output_views);
}

OnnxModel::~OnnxModel() = default;

std::string OnnxModel::finish_profile() {
  auto* ort = dynamic_cast<OrtBackend*>(backend_.get());
  if (!ort) throw std::logic_error("Profiling was not enabled");
  return ort->finish_profile();
}

const char* OnnxModel::provider() const { return backend_->provider(); }

std::string onnx_runtime_version() { return Ort::GetVersionString(); }
std::vector<std::string> onnx_available_providers() { return Ort::GetAvailableProviders(); }

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

void OnnxModel::run() { backend_->run(); }

const std::vector<float>& OnnxModel::output(const std::string& name) const {
  const auto it = output_buffers_.find(name);
  if (it == output_buffers_.end()) {
    throw std::runtime_error(path_ + " has no output named '" + name + "'");
  }
  return it->second;
}

}  // namespace holosoma
