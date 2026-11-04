// TensorRT execution of the same graphs ONNX Runtime runs, for hosts where the CPU is the budget.
//
// Only compiled when NvInfer is found. The engine is built once from the .onnx and cached, keyed
// on everything that would invalidate it -- the model's bytes, the TensorRT version and the
// device's compute capability -- because a plan is specific to all three and a stale one either
// fails to deserialize or, worse, does not.
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "holosoma_cpp/model_signature.hpp"

namespace holosoma {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
}

class Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[tensorrt] " << message << std::endl;
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

// Dynamic dimensions are pinned to 1, matching TensorInfo::element_count() and the single-robot
// batch every policy here runs at.
nvinfer1::Dims fixed_dims(const std::vector<int64_t>& shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) dims.d[i] = shape[i] > 0 ? static_cast<int64_t>(shape[i]) : 1;
  return dims;
}

std::filesystem::path cache_dir() {
  if (const char* explicit_dir = std::getenv("HOLOSOMA_TRT_CACHE_DIR")) return explicit_dir;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) return std::filesystem::path(xdg) / "holosoma/trt";
  if (const char* home = std::getenv("HOME")) return std::filesystem::path(home) / ".cache/holosoma/trt";
  return std::filesystem::temp_directory_path() / "holosoma-trt";
}

std::string cache_key(const std::string& path, const ModelSignature& signature) {
  int major = 0;
  int minor = 0;
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "cudaDeviceGetAttribute");
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), "cudaDeviceGetAttribute");
  std::ostringstream key;
  key << std::filesystem::canonical(path).string() << ":" << std::filesystem::file_size(path) << ":"
      << std::filesystem::last_write_time(path).time_since_epoch().count() << ":" << NV_TENSORRT_VERSION << ":sm"
      << major << minor;
  for (const auto& info : signature.inputs) {
    key << ":" << info.name;
    for (int64_t d : info.shape) key << "x" << (d > 0 ? d : 1);
  }
  std::ostringstream name;
  name << std::filesystem::path(path).stem().string() << "-" << std::hex << std::hash<std::string>{}(key.str())
       << ".engine";
  return name.str();
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto size = in.tellg();
  std::vector<char> bytes(static_cast<size_t>(size));
  in.seekg(0);
  in.read(bytes.data(), size);
  return in ? bytes : std::vector<char>{};
}

std::vector<char> build_plan(const std::string& path, const ModelSignature& signature) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
  if (!builder) throw std::runtime_error("could not create a TensorRT builder");
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
  if (!parser->parseFromFile(path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::ostringstream errors;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) errors << "\n  " << parser->getError(i)->desc();
    throw std::runtime_error("TensorRT could not parse " + path + ":" + errors.str());
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  // One profile, every dimension pinned: the plan is cached per shape anyway, and a fixed shape
  // is what lets TensorRT pick its fastest kernels.
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  for (const auto& info : signature.inputs) {
    const nvinfer1::Dims dims = fixed_dims(info.shape);
    for (auto selector :
         {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX}) {
      profile->setDimensions(info.name.c_str(), selector, dims);
    }
  }
  config->addOptimizationProfile(profile);
  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("TensorRT could not build an engine for " + path);
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

class TrtBackend final : public ModelBackend {
 public:
  TrtBackend(const std::string& path, const ModelSignature& signature) {
    const std::filesystem::path cached = cache_dir() / cache_key(path, signature);
    std::vector<char> plan = read_file(cached);
    if (plan.empty()) {
      std::cerr << "[tensorrt] building an engine for " << path << "; this takes a minute, then it is cached at "
                << cached << std::endl;
      plan = build_plan(path, signature);
      std::error_code ignored;
      std::filesystem::create_directories(cached.parent_path(), ignored);
      std::ofstream out(cached, std::ios::binary);
      out.write(plan.data(), static_cast<std::streamsize>(plan.size()));
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) throw std::runtime_error("could not deserialize a TensorRT engine for " + path);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("could not create a TensorRT execution context for " + path);
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  ~TrtBackend() override {
    for (auto& tensor : tensors_) cudaFree(tensor.device);
    if (stream_) cudaStreamDestroy(stream_);
  }

  void bind(const std::vector<std::pair<const TensorInfo*, float*>>& inputs,
            const std::vector<std::pair<const TensorInfo*, float*>>& outputs) override {
    for (const auto& [info, host] : inputs) {
      if (!context_->setInputShape(info->name.c_str(), fixed_dims(info->shape)))
        throw std::runtime_error("TensorRT rejected the shape of input '" + info->name + "'");
    }
    const auto add = [this](const auto& tensors, bool input) {
      for (const auto& [info, host] : tensors) {
        Tensor tensor{info->name, host, nullptr, static_cast<size_t>(info->element_count()) * sizeof(float), input};
        check_cuda(cudaMalloc(&tensor.device, tensor.bytes), "cudaMalloc");
        if (!context_->setTensorAddress(info->name.c_str(), tensor.device))
          throw std::runtime_error("TensorRT rejected the address of tensor '" + info->name + "'");
        tensors_.push_back(tensor);
      }
    };
    add(inputs, true);
    add(outputs, false);
    if (!context_->allInputDimensionsSpecified())
      throw std::runtime_error("TensorRT still has unspecified input dimensions after binding");
  }

  void run() override {
    for (const auto& tensor : tensors_) {
      if (tensor.input)
        check_cuda(cudaMemcpyAsync(tensor.device, tensor.host, tensor.bytes, cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync to device");
    }
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT inference failed");
    for (const auto& tensor : tensors_) {
      if (!tensor.input)
        check_cuda(cudaMemcpyAsync(tensor.host, tensor.device, tensor.bytes, cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync to host");
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

  const char* provider() const override { return "TensorRT"; }

 private:
  struct Tensor {
    std::string name;
    float* host = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    bool input = false;
  };

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  std::vector<Tensor> tensors_;
};

}  // namespace

std::unique_ptr<ModelBackend> make_tensorrt_backend(const std::string& path, const ModelSignature& signature) {
  return std::make_unique<TrtBackend>(path, signature);
}

}  // namespace holosoma
