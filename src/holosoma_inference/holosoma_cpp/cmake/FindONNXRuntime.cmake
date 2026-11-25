# Finds ONNX Runtime in a release tarball, native NuGet package, or system installation.
#
# Hints: ONNXRUNTIME_ROOT (CMake or environment variable), onnxruntime_ROOT.
# Defines the imported target ONNXRuntime::ONNXRuntime.

set(_ort_hints ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT} ${onnxruntime_ROOT} $ENV{onnxruntime_ROOT})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(_ort_runtime linux-x64)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_ort_runtime linux-arm64)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_ort_runtime osx-arm64)
endif()

find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ort_hints}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session build/native/include)
find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${_ort_hints}
  PATH_SUFFIXES lib lib64 runtimes/${_ort_runtime}/native)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
  get_filename_component(ONNXRuntime_LIBRARY_DIR "${ONNXRuntime_LIBRARY}" DIRECTORY)
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
