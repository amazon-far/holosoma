add_executable(holosoma_cpp_tests
  test_config.cpp
  test_input.cpp
  test_math.cpp
  test_python_parity.cpp)
target_link_libraries(holosoma_cpp_tests PRIVATE
  holosoma_cpp_core nlohmann_json::nlohmann_json ${HOLOSOMA_YAML_TARGET} GTest::gtest_main)
get_filename_component(HOLOSOMA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
target_compile_definitions(holosoma_cpp_tests PRIVATE
  HOLOSOMA_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
  HOLOSOMA_TEST_PRESET_DIR="${HOLOSOMA_PRESET_DIR}"
  HOLOSOMA_TEST_REPO_ROOT="${HOLOSOMA_REPO_ROOT}")
set_target_properties(holosoma_cpp_tests PROPERTIES BUILD_RPATH "${ONNXRuntime_LIBRARY_DIR}")

include(GoogleTest)
gtest_discover_tests(holosoma_cpp_tests WORKING_DIRECTORY "${HOLOSOMA_REPO_ROOT}" DISCOVERY_TIMEOUT 60)
