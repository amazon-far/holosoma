cmake_minimum_required(VERSION 3.16)
project(holosoma_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HOLOSOMA_BUILD_TESTS "Build the unit and Python-parity tests" ON)
option(HOLOSOMA_FETCH_DEPENDENCIES "Download dependencies that are not installed" ON)
set(HOLOSOMA_WITH_UNITREE_SDK "AUTO" CACHE STRING "Build the Unitree SDK2 robot interface: AUTO (Linux), ON, OFF")
set_property(CACHE HOLOSOMA_WITH_UNITREE_SDK PROPERTY STRINGS AUTO ON OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HolosomaDependencies)

set(HOLOSOMA_PRESET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config/inference" CACHE PATH
    "Directory with the inference presets (inference:<name>)")

add_library(holosoma_cpp_core
  src/cli.cpp
  src/commands.cpp
  src/config.cpp
  src/controller.cpp
  src/input.cpp
  src/locomotion_policy.cpp
  src/logging.cpp
  src/observation.cpp
  src/onnx_model.cpp
  src/policy.cpp
  src/terminal.cpp
  src/timing.cpp
  src/urdf_kinematics.cpp
  src/wbt_policy.cpp)
target_include_directories(holosoma_cpp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(holosoma_cpp_core
  PUBLIC ONNXRuntime::ONNXRuntime Threads::Threads
  PRIVATE ${HOLOSOMA_YAML_TARGET} nlohmann_json::nlohmann_json ${HOLOSOMA_TINYXML2_TARGET})
target_compile_definitions(holosoma_cpp_core PRIVATE
  HOLOSOMA_CPP_DEFAULT_PRESET_DIR="${HOLOSOMA_PRESET_DIR}")
target_compile_options(holosoma_cpp_core PRIVATE -Wall -Wextra -Wpedantic)

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

if(HOLOSOMA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "holosoma_cpp: Unitree SDK2 interface ${HOLOSOMA_UNITREE_SDK2_ENABLED}, tests ${HOLOSOMA_BUILD_TESTS}")
