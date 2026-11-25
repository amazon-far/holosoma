// File names for cached TensorRT engines. Kept apart from trt_backend.cpp, which needs TensorRT and a
// GPU, so the key can be tested on every build.
#pragma once

#include <string>

namespace holosoma {

// Names the cached engine for one model. `model_key` identifies the model, the TensorRT version and
// the device; `tf32_override` is the process's NVIDIA_TF32_OVERRIDE (nullptr when unset). That
// variable changes the precision TensorRT builds and executes with, and a plan built under one
// setting fails to create an execution context under another, so each setting gets its own file.
std::string trt_engine_file_name(const std::string& model_stem, const std::string& model_key,
                                 const char* tf32_override);

}  // namespace holosoma
