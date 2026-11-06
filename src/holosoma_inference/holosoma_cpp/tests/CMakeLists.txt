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

find_package(Python COMPONENTS Interpreter REQUIRED)
set(LIVE_FIXTURES "${CMAKE_CURRENT_BINARY_DIR}/live-fixtures")
add_custom_command(OUTPUT "${LIVE_FIXTURES}/stamp"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_live_stub_models.py" --output-dir "${LIVE_FIXTURES}"
  COMMAND ${CMAKE_COMMAND} -E touch "${LIVE_FIXTURES}/stamp"
  DEPENDS gen_live_stub_models.py)
add_custom_target(holosoma_live_fixtures DEPENDS "${LIVE_FIXTURES}/stamp")
add_executable(holosoma_live_tests test_live_policy.cpp)
add_dependencies(holosoma_live_tests holosoma_live_fixtures)
target_link_libraries(holosoma_live_tests PRIVATE holosoma_cpp_core ONNXRuntime::ONNXRuntime GTest::gtest_main)
target_compile_definitions(holosoma_live_tests PRIVATE HOLOSOMA_LIVE_FIXTURE_DIR="${LIVE_FIXTURES}")
gtest_discover_tests(holosoma_live_tests DISCOVERY_TIMEOUT 60)

add_executable(holosoma_backend_tests test_backend.cpp)
add_dependencies(holosoma_backend_tests holosoma_live_fixtures)
target_link_libraries(holosoma_backend_tests PRIVATE holosoma_cpp_core GTest::gtest_main)
target_compile_definitions(holosoma_backend_tests PRIVATE HOLOSOMA_LIVE_FIXTURE_DIR="${LIVE_FIXTURES}")
set_target_properties(holosoma_backend_tests PROPERTIES BUILD_RPATH "${ONNXRuntime_LIBRARY_DIR}")
gtest_discover_tests(holosoma_backend_tests DISCOVERY_TIMEOUT 60)

add_executable(holosoma_api_tests test_api.cpp)
add_dependencies(holosoma_api_tests holosoma_live_fixtures)
target_link_libraries(holosoma_api_tests PRIVATE holosoma_cpp_core ${HOLOSOMA_YAML_TARGET} GTest::gtest_main)
target_compile_definitions(holosoma_api_tests PRIVATE
  HOLOSOMA_LIVE_FIXTURE_DIR="${LIVE_FIXTURES}"
  HOLOSOMA_TEST_PRESET_DIR="${HOLOSOMA_PRESET_DIR}")
gtest_discover_tests(holosoma_api_tests DISCOVERY_TIMEOUT 60)

set(CONTEXTUAL_FIXTURES "${CMAKE_CURRENT_BINARY_DIR}/contextual-fixtures")
add_custom_command(OUTPUT "${CONTEXTUAL_FIXTURES}/stamp"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_contextual_models.py" --output-dir "${CONTEXTUAL_FIXTURES}"
  COMMAND ${CMAKE_COMMAND} -E touch "${CONTEXTUAL_FIXTURES}/stamp"
  DEPENDS gen_contextual_models.py)
add_custom_target(holosoma_contextual_fixtures DEPENDS "${CONTEXTUAL_FIXTURES}/stamp")
add_executable(holosoma_contextual_tests test_contextual_api.cpp)
add_dependencies(holosoma_contextual_tests holosoma_contextual_fixtures)
target_link_libraries(holosoma_contextual_tests PRIVATE holosoma_cpp_core GTest::gtest_main nlohmann_json::nlohmann_json)
target_compile_definitions(holosoma_contextual_tests PRIVATE HOLOSOMA_CONTEXTUAL_FIXTURE_DIR="${CONTEXTUAL_FIXTURES}")
gtest_discover_tests(holosoma_contextual_tests DISCOVERY_TIMEOUT 60)
