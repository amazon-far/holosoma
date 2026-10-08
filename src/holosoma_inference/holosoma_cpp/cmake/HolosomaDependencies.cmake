# Third-party dependencies of holosoma_cpp.
#
# Each dependency is taken from the system (find_package) when available and
# otherwise downloaded at a pinned version with a checked SHA-256, unless
# HOLOSOMA_FETCH_DEPENDENCIES is OFF.

include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # honor ONNXRUNTIME_ROOT
endif()
# Some pinned dependencies declare cmake_minimum_required(VERSION < 3.5), which CMake 4 rejects.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

find_package(Threads REQUIRED)

function(_holosoma_require_fetch name)
  if(NOT HOLOSOMA_FETCH_DEPENDENCIES)
    message(FATAL_ERROR "${name} not found and HOLOSOMA_FETCH_DEPENDENCIES=OFF; install it or point CMake at it.")
  endif()
  message(STATUS "holosoma: downloading ${name}")
endfunction()

# --- yaml-cpp ----------------------------------------------------------------
find_package(yaml-cpp 0.6 CONFIG QUIET)
if(NOT yaml-cpp_FOUND)
  _holosoma_require_fetch(yaml-cpp)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
    URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16)
  FetchContent_MakeAvailable(yaml-cpp)
endif()
if(TARGET yaml-cpp::yaml-cpp)
  set(HOLOSOMA_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(HOLOSOMA_YAML_TARGET yaml-cpp)
endif()

# --- nlohmann_json -------------------------------------------------------------
find_package(nlohmann_json 3.2 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  _holosoma_require_fetch(nlohmann_json)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- tinyxml2 ------------------------------------------------------------------
find_package(tinyxml2 CONFIG QUIET)
if(NOT tinyxml2_FOUND)
  _holosoma_require_fetch(tinyxml2)
  set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(tinyxml2
    URL https://github.com/leethomason/tinyxml2/archive/refs/tags/10.0.0.tar.gz
    URL_HASH SHA256=3bdf15128ba16686e69bce256cc468e76c7b94ff2c7f391cc5ec09e40bff3839)
  FetchContent_MakeAvailable(tinyxml2)
endif()
if(TARGET tinyxml2::tinyxml2)
  set(HOLOSOMA_TINYXML2_TARGET tinyxml2::tinyxml2)
else()
  set(HOLOSOMA_TINYXML2_TARGET tinyxml2)
endif()

# --- ONNX Runtime --------------------------------------------------------------
set(HOLOSOMA_ONNXRUNTIME_VERSION "1.30.0" CACHE STRING "ONNX Runtime release downloaded when none is found")
find_package(ONNXRuntime QUIET)
if(NOT ONNXRuntime_FOUND)
  _holosoma_require_fetch("ONNX Runtime ${HOLOSOMA_ONNXRUNTIME_VERSION}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_ort_platform linux-x64)
    set(_ort_sha a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_ort_platform linux-aarch64)
    set(_ort_sha e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da)
  elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_ort_platform osx-arm64)
    set(_ort_sha 6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012)
  else()
    message(FATAL_ERROR "No prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; set ONNXRUNTIME_ROOT.")
  endif()
  set(_ort_url_hash)
  if(HOLOSOMA_ONNXRUNTIME_VERSION STREQUAL "1.30.0")
    set(_ort_url_hash URL_HASH SHA256=${_ort_sha})
  endif()
  FetchContent_Declare(onnxruntime_prebuilt
    URL https://github.com/microsoft/onnxruntime/releases/download/v${HOLOSOMA_ONNXRUNTIME_VERSION}/onnxruntime-${_ort_platform}-${HOLOSOMA_ONNXRUNTIME_VERSION}.tgz
    ${_ort_url_hash})
  # The release tarball has no CMakeLists.txt, so this only downloads and unpacks it.
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}" CACHE PATH "ONNX Runtime root" FORCE)
  find_package(ONNXRuntime REQUIRED)
endif()

# --- Unitree SDK2 (robot / simulator bridge transport) -------------------------
set(HOLOSOMA_UNITREE_SDK2_ENABLED OFF)
if(HOLOSOMA_WITH_UNITREE_SDK STREQUAL "AUTO")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|aarch64|arm64)$")
    set(_want_unitree ON)
  else()
    set(_want_unitree OFF)
    message(STATUS "holosoma: Unitree SDK2 needs Linux x86_64/aarch64; building without the robot interface")
  endif()
else()
  set(_want_unitree ${HOLOSOMA_WITH_UNITREE_SDK})
endif()

if(_want_unitree)
  find_package(unitree_sdk2 CONFIG QUIET PATHS /opt/unitree_robotics)
  if(NOT TARGET unitree_sdk2)
    _holosoma_require_fetch("Unitree SDK2")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_BINDING OFF CACHE BOOL "" FORCE)
    # amazon-far/unitree_sdk2 is the source of the far-unitree-sdk binding used by the Python runtime.
    FetchContent_Declare(unitree_sdk2
      URL https://github.com/amazon-far/unitree_sdk2/archive/9cf396bda17485febbacb2fadec27c52e0aff391.tar.gz
      URL_HASH SHA256=ecb231cce66e532ec1bb63e3a94427483421bce4cd9c1a47545f7753995e3c39)
    FetchContent_MakeAvailable(unitree_sdk2)
  endif()
  set(HOLOSOMA_UNITREE_SDK2_ENABLED ON)
endif()

# --- GoogleTest ----------------------------------------------------------------
if(HOLOSOMA_BUILD_TESTS)
  find_package(GTest 1.10 CONFIG QUIET)
  if(NOT GTest_FOUND)
    _holosoma_require_fetch(GoogleTest)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()
