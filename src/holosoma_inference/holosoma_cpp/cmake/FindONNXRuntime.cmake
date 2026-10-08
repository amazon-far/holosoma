# Finds a prebuilt ONNX Runtime (the release tarball layout: include/ and lib/).
#
# Hints: ONNXRUNTIME_ROOT (CMake or environment variable), onnxruntime_ROOT.
# Defines the imported target ONNXRuntime::ONNXRuntime.

set(_ort_hints ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT} ${onnxruntime_ROOT} $ENV{onnxruntime_ROOT})

find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ort_hints}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${_ort_hints}
  PATH_SUFFIXES lib lib64)

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
