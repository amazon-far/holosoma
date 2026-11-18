// ExecuTorch execution of a .pte program, as written by holosoma's export_policy_as_pte.
//
// Only compiled when the ExecuTorch package is found. A .pte carries its own signature, so unlike
// the TensorRT backend no .onnx is needed beside it: shapes come from the program, and tensor
// names and metadata from its `signature` method, a JSON string written at export time.
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool_guard.h>

#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "holosoma_cpp/model_signature.hpp"

namespace holosoma {
namespace {

using namespace executorch::extension;
using namespace executorch::runtime;

// executorch::runtime::TensorInfo, spelled out: holosoma::TensorInfo is the one in scope.
std::vector<int64_t> tensor_shape(Result<executorch::runtime::TensorInfo> meta, const std::string& what) {
  if (!meta.ok()) throw std::runtime_error(what + " is not a tensor");
  if (meta->scalar_type() != executorch::aten::ScalarType::Float) throw std::runtime_error(what + " must be float32");
  return {meta->sizes().begin(), meta->sizes().end()};
}

// ExecuTorch's CUDA backend keeps its weights in a data file, which the exporter puts beside the program.
std::vector<std::string> data_files(const std::string& path) {
  const auto data = std::filesystem::path(path).replace_extension(".ptd");
  if (std::filesystem::exists(data)) return {data.string()};
  return {};
}

class ExecuTorchBackend final : public ModelBackend {
 public:
  ExecuTorchBackend(const std::string& path, ModelSignature& signature) : path_(path), module_(path, data_files(path)) {
    // runtime.backend applies to every model in the process, so a secondary policy that is still an
    // .onnx lands here too; name that instead of failing inside the program loader.
    if (std::filesystem::path(path).extension() != ".pte")
      throw std::runtime_error(
          "the executorch backend runs .pte programs, and " + path +
          " is not one; every model this process loads, including a secondary policy, has to be exported");
    // ExecuTorch's thread pool spans every core and cannot be sized from here; on these tiny MLPs
    // it only adds wake-up jitter, so XNNPACK, which binds the pool at load, runs on the caller.
    threadpool::NoThreadPoolGuard single_threaded;
    if (const Error error = module_.load_forward(); error != Error::Ok) {
      throw std::runtime_error("Failed to load ExecuTorch program " + path + ": " + to_string(error) +
                               (error == Error::NotFound ? " (a delegate or data file it needs is missing)" : ""));
    }
    const auto meta = module_.method_meta("forward");
    const auto blob = module_.execute("signature");
    if (!blob.ok() || blob->size() != 1 || !blob->front().isString())
      throw std::runtime_error(path + " has no 'signature' method; export it with export_policy_as_pte");
    const std::string_view text = blob->front().toString();
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    try {
      const auto json = nlohmann::json::parse(text.begin(), text.end());
      input_names = json.at("inputs").get<std::vector<std::string>>();
      output_names = json.at("outputs").get<std::vector<std::string>>();
      signature.metadata = json.at("metadata").get<std::map<std::string, std::string>>();
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(path + ": malformed signature (" + e.what() + ")");
    }
    if (input_names.size() != meta->num_inputs() || output_names.size() != meta->num_outputs())
      throw std::runtime_error(path +
                               ": the signature names a different number of inputs or outputs than the program has");
    for (size_t i = 0; i < input_names.size(); ++i) {
      signature.inputs.push_back(
          {input_names[i], tensor_shape(meta->input_tensor_meta(i), path + " input '" + input_names[i] + "'")});
    }
    for (size_t i = 0; i < output_names.size(); ++i) {
      signature.outputs.push_back(
          {output_names[i], tensor_shape(meta->output_tensor_meta(i), path + " output '" + output_names[i] + "'")});
    }

    provider_ = "ExecuTorch";
    for (size_t i = 0; i < meta->num_backends(); ++i) {
      const auto name = meta->get_backend_name(i);
      if (name.ok()) provider_ += std::string(i == 0 ? " (" : ", ") + *name;
    }
    if (meta->num_backends() > 0) provider_ += ")";
  }

  void bind(const std::vector<std::pair<const TensorInfo*, float*>>& inputs,
            const std::vector<std::pair<const TensorInfo*, float*>>& outputs) override {
    for (const auto& [info, host] : inputs) {
      tensors_.push_back(
          make_tensor_ptr(std::vector<executorch::aten::SizesType>(info->shape.begin(), info->shape.end()), host));
      args_.emplace_back(tensors_.back());
    }
    for (const auto& [info, host] : outputs) outputs_.push_back({host, static_cast<size_t>(info->element_count())});
  }

  void run() override {
    threadpool::NoThreadPoolGuard single_threaded;
    const auto result = module_.forward(args_);
    if (!result.ok())
      throw std::runtime_error("ExecuTorch inference failed for " + path_ + ": " + to_string(result.error()));
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& tensor = result->at(i).toTensor();
      if (static_cast<size_t>(tensor.numel()) != outputs_[i].count)
        throw std::runtime_error("ExecuTorch output " + std::to_string(i) + " of " + path_ + " has an unexpected size");
      std::memcpy(outputs_[i].host, tensor.const_data_ptr<float>(), outputs_[i].count * sizeof(float));
    }
  }

  const char* provider() const override { return provider_.c_str(); }

 private:
  struct Output {
    float* host = nullptr;
    size_t count = 0;
  };

  std::string path_;
  Module module_;
  std::string provider_;
  std::vector<TensorPtr> tensors_;
  std::vector<EValue> args_;
  std::vector<Output> outputs_;
};

}  // namespace

std::unique_ptr<ModelBackend> make_executorch_backend(const std::string& path, ModelSignature& signature) {
  return std::make_unique<ExecuTorchBackend>(path, signature);
}

}  // namespace holosoma
