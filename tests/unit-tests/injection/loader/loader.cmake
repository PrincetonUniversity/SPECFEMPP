# Tests for the injection plugin loader (core/specfem/injection). Included from
# tests/unit-tests/CMakeLists.txt.
#
# Each test plugin is a real loadable module; all but one are broken on
# purpose, in exactly one way the loader must catch.

specfempp_add_injection_plugin(injection_loader_test_good_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/good_plugin.cpp)
specfempp_add_injection_plugin(injection_loader_test_no_manifest_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/no_manifest.c)
specfempp_add_injection_plugin(injection_loader_test_bad_abi_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/bad_abi_manifest.c)
specfempp_add_injection_plugin(injection_loader_test_table_mismatch_plugin
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/table_mismatch.c)
# Bypasses specfempp_add_injection_plugin, whose export check would reject a
# module without the entry point.
add_library(injection_loader_test_missing_symbol_plugin MODULE
  ${CMAKE_CURRENT_LIST_DIR}/missing_symbol.c)
target_link_libraries(injection_loader_test_missing_symbol_plugin
  PRIVATE specfem::plugin_sdk)

set(_injection_loader_plugins
  good no_manifest bad_abi table_mismatch missing_symbol)
set(_injection_loader_definitions)
foreach(_plugin ${_injection_loader_plugins})
  string(TOUPPER ${_plugin} _upper)
  list(APPEND _injection_loader_definitions
    "SPFP_TEST_${_upper}_PLUGIN=\"$<TARGET_FILE:injection_loader_test_${_plugin}_plugin>\"")
endforeach()

specfem_add_test(injection_loader_tests
  LABELS      injection
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/loader_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/options_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/request_tests.cpp
  LIBRARIES   specfem::injection
              gmock_main
  DEFINITIONS ${_injection_loader_definitions}
)
foreach(_plugin ${_injection_loader_plugins})
  add_dependencies(injection_loader_tests
    injection_loader_test_${_plugin}_plugin)
endforeach()
