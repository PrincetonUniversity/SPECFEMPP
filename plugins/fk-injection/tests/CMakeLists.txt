# Tests of the FK plugin: the static library directly, and the module loaded
# with dlopen as SPECFEM++ loads it. Added from tests/unit-tests/CMakeLists.txt.

specfem_add_test(fk_injection_skeleton_tests
  LABELS      fk_injection
  SOURCES     runner.cpp
              skeleton_tests.cpp
  LIBRARIES   fk_injection_impl
              gtest
  DEFINITIONS $<$<BOOL:${FK_INJECTION_DOUBLE}>:FK_INJECTION_EXPECT_DOUBLE>
)

specfem_add_test(injection_math_tests
  LABELS      fk_injection
  SOURCES     runner.cpp
              register_array_tests.cpp
              cubic_bspline_tests.cpp
              fft_tests.cpp
  LIBRARIES   fk_injection_impl
              gtest
)

# fk_injection_module_tests embeds the module path, so with an external test
# directory the module has to live there too (see specfem_add_test).
if(SPECFEM_TEST_DATA_MODE STREQUAL "INSTALL")
  set_target_properties(fk_injection PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}")
endif()

specfem_add_test(injection_types_tests
  LABELS      fk_injection
  SOURCES     runner.cpp
              layered_model_tests.cpp
              time_window_tests.cpp
              incident_wave_tests.cpp
              fk_result_tests.cpp
  LIBRARIES   fk_injection_impl
              gtest
)

# Loads the module only.
specfem_add_test(fk_injection_module_tests
  LABELS      fk_injection
  SOURCES     module_tests.cpp
  LIBRARIES   specfem::plugin_sdk
              gtest_main
              ${CMAKE_DL_LIBS}
  DEFINITIONS "FK_INJECTION_MODULE=\"$<TARGET_FILE:fk_injection>\""
)
add_dependencies(fk_injection_module_tests fk_injection)
