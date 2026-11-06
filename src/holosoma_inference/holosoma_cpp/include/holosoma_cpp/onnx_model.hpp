// ONNX Runtime session with preallocated float32 input/output buffers.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "holosoma_cpp/config.hpp"

namespace holosoma {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;     // -1 for dynamic dimensions
  int64_t element_count() const;  // dynamic dimensions count as 1 (batch size 1)
};

// Executes a loaded graph over buffers the model owns. One implementation per Backend; the model
// reads its signature and metadata through ONNX Runtime either way, so a TensorRT engine cannot
// disagree with the CPU path about what the graph's inputs are called.
class ModelBackend {
 public:
  virtual ~ModelBackend() = default;
  // Host buffers are allocated once and outlive the backend, so addresses may be retained.
  virtual void bind(const std::vector<std::pair<const TensorInfo*, float*>>& inputs,
                    const std::vector<std::pair<const TensorInfo*, float*>>& outputs) = 0;
  virtual void run() = 0;
  virtual const char* provider() const = 0;
};

// Whether this build can execute `backend` at all. A TensorRT build is opt-in at compile time and
// needs NvInfer, so a config naming it has to fail loudly rather than quietly run on the CPU.
bool backend_available(Backend backend);

class OnnxModel {
 public:
  explicit OnnxModel(const std::string& path, Backend backend = Backend::kCpu, int intra_op_threads = 1,
                     const std::string& profile_prefix = "");
  ~OnnxModel();
  OnnxModel(const OnnxModel&) = delete;
  OnnxModel& operator=(const OnnxModel&) = delete;

  const std::string& path() const { return path_; }
  const std::vector<TensorInfo>& inputs() const { return inputs_; }
  const std::vector<TensorInfo>& outputs() const { return outputs_; }
  const TensorInfo* find_input(const std::string& name) const;
  const TensorInfo* find_output(const std::string& name) const;

  // Raw custom metadata (values are usually JSON).
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  std::optional<std::string> metadata_value(const std::string& key) const;

  // Input buffer for `name`, sized to the input's element count. Fill before run().
  std::vector<float>& input(const std::string& name);

  // Runs the graph and fetches every output into its buffer. All tensors must be float32.
  void run();

  const std::vector<float>& output(const std::string& name) const;
  std::string finish_profile();
  const char* provider() const;

 private:
  std::unique_ptr<ModelBackend> backend_;
  std::string path_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
  std::map<std::string, std::string> metadata_;
  std::map<std::string, std::vector<float>> input_buffers_;
  std::map<std::string, std::vector<float>> output_buffers_;
};

std::string onnx_runtime_version();
std::vector<std::string> onnx_available_providers();

}  // namespace holosoma
