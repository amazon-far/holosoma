cmake_minimum_required(VERSION 3.16)
project(holosoma_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set(HOLOSOMA_ORIGIN "@loader_path")
else()
  set(HOLOSOMA_ORIGIN "$ORIGIN")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HOLOSOMA_BUILD_TESTS "Build the unit and Python-parity tests" ON)
option(HOLOSOMA_FETCH_DEPENDENCIES "Download dependencies that are not installed" ON)
option(HOLOSOMA_BUILD_PYTHON "Build the Python native extension" OFF)
option(HOLOSOMA_BUILD_RUNNER "Build the standalone robot runner" ON)
set(HOLOSOMA_WITH_UNITREE_SDK "AUTO" CACHE STRING "Build the Unitree SDK2 robot interface: AUTO (Linux), ON, OFF")
set_property(CACHE HOLOSOMA_WITH_UNITREE_SDK PROPERTY STRINGS AUTO ON OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HolosomaDependencies)

set(HOLOSOMA_PRESET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config/inference" CACHE PATH
    "Directory with the inference presets (inference:<name>)")

add_library(holosoma_cpp_core SHARED
  src/general_wbt_policy.cpp
  src/cli.cpp
  src/commands.cpp
  src/config.cpp
  src/controller.cpp
  src/input.cpp
  src/api.cpp
  src/holosoma_inference_core.cpp
  src/locomotion_policy.cpp
  src/logging.cpp
  src/observation.cpp
  src/onnx_model.cpp
  src/policy.cpp
  src/terminal.cpp
  src/timing.cpp
  src/urdf_kinematics.cpp
  src/wbt_policy.cpp)
add_library(holosoma::inference ALIAS holosoma_cpp_core)
target_include_directories(holosoma_cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(holosoma_cpp_core
  PUBLIC Threads::Threads Eigen3::Eigen
  PRIVATE ONNXRuntime::ONNXRuntime ${HOLOSOMA_YAML_TARGET} nlohmann_json::nlohmann_json ${HOLOSOMA_TINYXML2_TARGET} ${CMAKE_DL_LIBS})
target_compile_definitions(holosoma_cpp_core PRIVATE
  HOLOSOMA_CPP_DEFAULT_PRESET_DIR="${HOLOSOMA_PRESET_DIR}")
target_compile_options(holosoma_cpp_core PRIVATE -Wall -Wextra -Wpedantic)

if(HOLOSOMA_BUILD_RUNNER)
add_library(holosoma_robot_interface src/unitree_interface.cpp)
target_link_libraries(holosoma_robot_interface PUBLIC holosoma_cpp_core)
if(HOLOSOMA_UNITREE_SDK2_ENABLED)
  target_link_libraries(holosoma_robot_interface PRIVATE unitree_sdk2)
  target_compile_definitions(holosoma_robot_interface PRIVATE HOLOSOMA_WITH_UNITREE_SDK=1)
else()
  target_compile_definitions(holosoma_robot_interface PRIVATE HOLOSOMA_WITH_UNITREE_SDK=0)
endif()
target_compile_options(holosoma_robot_interface PRIVATE -Wall -Wextra)

add_executable(holosoma_run_policy apps/run_policy.cpp)
target_link_libraries(holosoma_run_policy PRIVATE holosoma_robot_interface)
target_compile_options(holosoma_run_policy PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(holosoma_run_policy PROPERTIES
  BUILD_RPATH "${ONNXRuntime_LIBRARY_DIR}"
  INSTALL_RPATH "\$ORIGIN/../lib;${ONNXRuntime_LIBRARY_DIR}")

install(TARGETS holosoma_run_policy RUNTIME DESTINATION bin)
endif()

add_executable(ExampleApp apps/ExampleApp.cpp)
target_link_libraries(ExampleApp PRIVATE holosoma_cpp_core ${HOLOSOMA_YAML_TARGET})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set_target_properties(holosoma_cpp_core PROPERTIES
  EXPORT_NAME inference
  BUILD_RPATH "${ONNXRuntime_LIBRARY_DIR}"
  INSTALL_RPATH "${HOLOSOMA_ORIGIN}")
install(TARGETS holosoma_cpp_core EXPORT holosomaInferenceTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT holosomaInferenceTargets NAMESPACE holosoma::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/holosomaInference)
configure_package_config_file(cmake/holosomaInferenceConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/holosomaInferenceConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/holosomaInference)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/holosomaInferenceConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/holosomaInference)

if(HOLOSOMA_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/python_bindings.cpp)
  target_link_libraries(_native PRIVATE holosoma_cpp_core ${HOLOSOMA_YAML_TARGET})
  set_target_properties(_native PROPERTIES INSTALL_RPATH "${HOLOSOMA_ORIGIN}/../${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS _native LIBRARY DESTINATION holosoma_inference)
endif()

# Keep installed libraries loadable after the build/download directory is gone.
# Only ONNX Runtime's own libraries (core plus execution providers): the directory
# can be a system lib dir such as /usr/lib/x86_64-linux-gnu.
install(DIRECTORY "${ONNXRuntime_LIBRARY_DIR}/" DESTINATION ${CMAKE_INSTALL_LIBDIR}
  FILES_MATCHING PATTERN "libonnxruntime*.so*" PATTERN "libonnxruntime*.dylib" PATTERN "onnxruntime*.dll")
foreach(dependency ${HOLOSOMA_YAML_TARGET} ${HOLOSOMA_TINYXML2_TARGET})
  get_target_property(dependency_type ${dependency} TYPE)
  if(dependency_type STREQUAL "SHARED_LIBRARY")
    install(FILES $<TARGET_FILE:${dependency}> $<TARGET_SONAME_FILE:${dependency}>
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endforeach()

if(HOLOSOMA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "holosoma_cpp: Unitree SDK2 interface ${HOLOSOMA_UNITREE_SDK2_ENABLED}, tests ${HOLOSOMA_BUILD_TESTS}")
