# Tests for the injection plugin SDK (sdk/). Included from
# tests/unit-tests/CMakeLists.txt.
#
# The test plugins are real loadable modules, built the way plugin authors build
# them, and loaded with dlopen by the tests.

specfempp_add_injection_plugin(plugin_sdk_test_good_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/good_plugin.cpp)
specfempp_add_injection_plugin(plugin_sdk_test_c_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/c_plugin.c)
# Deliberately bypasses specfempp_add_injection_plugin so it leaks a symbol.
add_library(plugin_sdk_test_leaky_plugin MODULE
  ${CMAKE_CURRENT_LIST_DIR}/leaky_plugin.cpp)
target_link_libraries(plugin_sdk_test_leaky_plugin PRIVATE specfem::plugin_sdk)

specfem_add_test(plugin_sdk_tests
  LABELS      plugin_sdk
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/plugin_sdk_tests.cpp
  LIBRARIES   specfem::plugin_sdk
              gtest_main
              ${CMAKE_DL_LIBS}
  DEFINITIONS "SPFP_TEST_GOOD_PLUGIN=\"$<TARGET_FILE:plugin_sdk_test_good_plugin>\""
              "SPFP_TEST_C_PLUGIN=\"$<TARGET_FILE:plugin_sdk_test_c_plugin>\""
)
add_dependencies(plugin_sdk_tests
  plugin_sdk_test_good_plugin plugin_sdk_test_c_plugin)

foreach(_plugin good c leaky)
  add_test(NAME plugin_sdk_exports_${_plugin}
    COMMAND ${CMAKE_COMMAND}
      "-DPLUGIN_FILE=$<TARGET_FILE:plugin_sdk_test_${_plugin}_plugin>"
      "-DFORMAT=$<IF:$<PLATFORM_ID:Darwin>,macho,elf>"
      "-DNM=${CMAKE_NM}"
      -P "${CMAKE_SOURCE_DIR}/sdk/cmake/check_plugin_exports.cmake")
  set_tests_properties(plugin_sdk_exports_${_plugin} PROPERTIES
    LABELS plugin_sdk)
endforeach()
# The checker must reject the leaky plugin and name the extra symbol.
set_tests_properties(plugin_sdk_exports_leaky PROPERTIES
  PASS_REGULAR_EXPRESSION "must export exactly specfempp_injection_plugin, but exports:.*specfempp_leaked_symbol")

# Builds a plugin against the installed SDK only, as an external author would.
add_test(NAME plugin_sdk_consumer
  COMMAND ${CMAKE_COMMAND}
    "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
    "-DSOURCE_DIR=${CMAKE_CURRENT_LIST_DIR}/consumer"
    "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/plugin_sdk_consumer"
    "-DGENERATOR=${CMAKE_GENERATOR}"
    "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
    -P "${CMAKE_CURRENT_LIST_DIR}/run_consumer.cmake")
set_tests_properties(plugin_sdk_consumer PROPERTIES
  LABELS plugin_sdk
  TIMEOUT 300)
