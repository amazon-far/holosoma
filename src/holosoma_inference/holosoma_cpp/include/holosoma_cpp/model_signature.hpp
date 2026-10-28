// What a graph's inputs, outputs and metadata are, read once at load time.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holosoma_cpp/onnx_model.hpp"

namespace holosoma {

struct ModelSignature {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::map<std::string, std::string> metadata;
};

// Reads the signature through a throwaway CPU session. Only for a backend that cannot read ONNX
// metadata itself -- a serialized TensorRT engine carries none, and gains and joint names live
// there, so they have to come from the .onnx regardless of what executes it.
ModelSignature read_onnx_signature(const std::string& path);

// Defined in trt_backend.cpp, which is only compiled when NvInfer is available.
std::unique_ptr<ModelBackend> make_tensorrt_backend(const std::string& path, const ModelSignature& signature);

}  // namespace holosoma
