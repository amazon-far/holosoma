# Finds TensorRT and the CUDA runtime it needs (the JetPack / CUDA toolkit layout).
#
# Hints: TENSORRT_ROOT, CUDAToolkit_ROOT (CMake or environment variables).
# Defines the imported target TensorRT::TensorRT, which carries NvInfer, the ONNX parser and cudart.
#
# Deliberately not find_package(CUDAToolkit): only the runtime library and its headers are needed,
# and requiring nvcc would make a host without the full toolkit unable to configure.

set(_trt_hints ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} /usr /usr/local/cuda /usr/local/tensorrt)
set(_cuda_hints ${CUDAToolkit_ROOT} $ENV{CUDAToolkit_ROOT} /usr/local/cuda /usr)

find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h
  HINTS ${_trt_hints} PATH_SUFFIXES include include/aarch64-linux-gnu include/x86_64-linux-gnu)
find_library(TensorRT_LIBRARY NAMES nvinfer HINTS ${_trt_hints} PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)
find_library(TensorRT_ONNX_PARSER NAMES nvonnxparser HINTS ${_trt_hints}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)
find_path(TensorRT_CUDA_INCLUDE_DIR NAMES cuda_runtime_api.h HINTS ${_cuda_hints} PATH_SUFFIXES include)
find_library(TensorRT_CUDART NAMES cudart HINTS ${_cuda_hints} PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu targets/aarch64-linux/lib)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  # TensorRT 10 defines NV_TENSORRT_MAJOR indirectly as TRT_MAJOR_ENTERPRISE, so read the numbers.
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_lines
    REGEX "#define TRT_(MAJOR|MINOR|PATCH)_ENTERPRISE +[0-9]+")
  set(TensorRT_VERSION "")
  foreach(_part MAJOR MINOR PATCH)
    foreach(_line ${_trt_version_lines})
      if(_line MATCHES "#define TRT_${_part}_ENTERPRISE +([0-9]+)")
        string(APPEND TensorRT_VERSION "${CMAKE_MATCH_1}.")
      endif()
    endforeach()
  endforeach()
  string(REGEX REPLACE "\\.$" "" TensorRT_VERSION "${TensorRT_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_ONNX_PARSER TensorRT_INCLUDE_DIR TensorRT_CUDART TensorRT_CUDA_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
  add_library(TensorRT::TensorRT INTERFACE IMPORTED)
  set_target_properties(TensorRT::TensorRT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR};${TensorRT_CUDA_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARY};${TensorRT_ONNX_PARSER};${TensorRT_CUDART}")
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_ONNX_PARSER TensorRT_CUDART TensorRT_CUDA_INCLUDE_DIR)
