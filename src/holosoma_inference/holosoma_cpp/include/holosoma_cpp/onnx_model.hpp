// ONNX Runtime session with preallocated float32 input/output buffers.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace holosoma {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;     // -1 for dynamic dimensions
  int64_t element_count() const;  // dynamic dimensions count as 1 (batch size 1)
};

class OnnxModel {
 public:
  explicit OnnxModel(const std::string& path);
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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string path_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
  std::map<std::string, std::string> metadata_;
  std::map<std::string, std::vector<float>> input_buffers_;
  std::map<std::string, std::vector<float>> output_buffers_;
};

}  // namespace holosoma
