add_executable(
  displacement_newmark_2d_tests
  displacement_tests/Newmark/dim2/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_2d_tests
  specfem::quadrature
  specfem::mesh
  yaml-cpp
  specfem_environment
  specfem::assembly
  specfem::runtime_configuration
  timescheme
  point
  specfem::algorithms
  specfem::solver
  specfem::periodic_tasks
  ${BOOST_LIBS}
  -lpthread -lm
)

add_custom_command(TARGET displacement_newmark_2d_tests POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/dim2/tests.yaml
          ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2/tests.yaml
     COMMENT "Moving displacement_newmark_2d_tests data files to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2"
)

add_executable(
  displacement_newmark_3d_tests
  displacement_tests/Newmark/dim3/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_3d_tests
  specfem::quadrature
  specfem::mesh
  yaml-cpp
  specfem_environment
  specfem::assembly
  specfem::runtime_configuration
  timescheme
  point
  specfem::algorithms
  specfem::solver
  specfem::periodic_tasks
  ${BOOST_LIBS}
  -lpthread -lm
)

add_custom_command(TARGET displacement_newmark_3d_tests POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/dim3/tests.yaml
          ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3/tests.yaml
     COMMENT "Moving displacement_newmark_3d_tests data files to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3"
)

# Register serial tests for discovery
set(SERIAL_TEST_TARGETS
  displacement_newmark_2d_tests
  displacement_newmark_3d_tests
)

# Link test data directories for serial tests
set(SERIAL_LINK_DIRS
  displacement_tests
)

# Setup test script writer (called once for all targets)
specfem_write_copy_test_cmake_script()

# Register each test target for discovery with optional path-fix
foreach(test_target IN LISTS SERIAL_TEST_TARGETS)
    specfem_register_test_target(${test_target})
endforeach()
