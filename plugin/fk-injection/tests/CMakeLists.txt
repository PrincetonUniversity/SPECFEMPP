# Tests of the FK plugin: the static library directly, and the module loaded
# with dlopen as SPECFEM++ loads it.

include(FetchContent)
# Same release as core's cmake/googletest.cmake.
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  EXCLUDE_FROM_ALL
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(fk_plugin_skeleton_tests
  runner.cpp
  skeleton_tests.cpp
)
target_link_libraries(fk_plugin_skeleton_tests PRIVATE
  specfem_fk_core
  GTest::gtest
)
if(FK_PLUGIN_DOUBLE)
  target_compile_definitions(fk_plugin_skeleton_tests PRIVATE
    FK_PLUGIN_EXPECT_DOUBLE)
endif()
gtest_discover_tests(fk_plugin_skeleton_tests)

# Loads the module only.
add_executable(fk_plugin_module_tests module_tests.cpp)
target_link_libraries(fk_plugin_module_tests PRIVATE
  specfem::plugin_sdk
  GTest::gtest_main
  ${CMAKE_DL_LIBS}
)
target_compile_definitions(fk_plugin_module_tests PRIVATE
  "FK_PLUGIN_MODULE=\"$<TARGET_FILE:specfem_fk_plugin>\"")
add_dependencies(fk_plugin_module_tests specfem_fk_plugin)
gtest_discover_tests(fk_plugin_module_tests)
