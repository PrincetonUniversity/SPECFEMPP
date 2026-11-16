# Tests of the FK plugin: the static library directly, and the module loaded
# with dlopen as SPECFEM++ loads it. Added from tests/unit-tests/CMakeLists.txt.

specfem_add_test(fk_plugin_skeleton_tests
  LABELS      fk_plugin
  SOURCES     runner.cpp
              skeleton_tests.cpp
  LIBRARIES   specfem_fk_core
              gtest
  DEFINITIONS $<$<BOOL:${FK_PLUGIN_DOUBLE}>:FK_PLUGIN_EXPECT_DOUBLE>
)

specfem_add_test(injection_math_tests
  LABELS      fk_plugin
  SOURCES     runner.cpp
              register_array_tests.cpp
              cubic_bspline_tests.cpp
              fft_tests.cpp
  LIBRARIES   specfem_fk_core
              gtest
)

# fk_plugin_module_tests embeds the module path, so with an external test
# directory the module has to live there too (see specfem_add_test).
if(SPECFEM_TEST_DATA_MODE STREQUAL "INSTALL")
  set_target_properties(specfem_fk_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}")
endif()

# Loads the module only.
specfem_add_test(fk_plugin_module_tests
  LABELS      fk_plugin
  SOURCES     module_tests.cpp
  LIBRARIES   specfem::plugin_sdk
              gtest_main
              ${CMAKE_DL_LIBS}
  DEFINITIONS "FK_PLUGIN_MODULE=\"$<TARGET_FILE:specfem_fk_plugin>\""
)
add_dependencies(fk_plugin_module_tests specfem_fk_plugin)
