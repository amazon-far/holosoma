#pragma once

#include <onnxruntime_cxx_api.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <dlfcn.h>
#endif

namespace holosoma {
#ifdef __linux__
inline void* load_nvidia_driver_library(const char* name) {
  if (void* handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL)) return handle;
  for (const char* directory :
       {"/run/opengl-driver/lib/", "/usr/lib/x86_64-linux-gnu/", "/usr/lib64/", "/usr/local/nvidia/lib64/"}) {
    if (void* handle = dlopen((std::string(directory) + name).c_str(), RTLD_NOW | RTLD_GLOBAL)) return handle;
  }
  throw std::runtime_error(std::string("NVIDIA ") + name + " is unavailable; check the driver and GPU passthrough");
}
#endif

inline void enable_cuda(Ort::SessionOptions& options) {
#ifdef __linux__
  // Hermetic ELF loaders may not consult the host driver cache. Preload the
  // driver and PTX compiler before ORT caches an unsuccessful driver lookup.
  static void* compiler = load_nvidia_driver_library("libnvidia-ptxjitcompiler.so.1");
  static void* driver = load_nvidia_driver_library("libcuda.so.1");
  (void)compiler;
  (void)driver;
#endif
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  options.AppendExecutionProvider_CUDA(cuda_options);
}
}  // namespace holosoma
