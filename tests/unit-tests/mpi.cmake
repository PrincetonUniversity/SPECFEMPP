# MPI-specific test definitions and setup
# This file contains MPI-only test executables and their discovery registration.

# Include GoogleTest module for test discovery
include(GoogleTest)

# ==============================================================================
# Helper function to register an MPI test with configurable processor count
# and test properties
# ==============================================================================
# Usage: add_mpi_test(TARGET NUM_PROCESSES)
# Example: add_mpi_test(mesh_mpi_dim3_tests 4)
# Uses gtest_discover_tests with POST_BUILD discovery mode and
# CROSSCOMPILING_EMULATOR to run the binary via the MPI launcher.
function(add_mpi_test TARGET NUM_PROCESSES)
    # Copy test binary to TEST_OUTPUT_DIR after build
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${TARGET}>
            ${TEST_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET}>
        COMMENT "Moving ${TARGET} to ${TEST_OUTPUT_DIR}"
    )

    # CROSSCOMPILING_EMULATOR prefixes test execution with the MPI launcher.
    set_target_properties(${TARGET} PROPERTIES
        CROSSCOMPILING_EMULATOR "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${NUM_PROCESSES}"
    )

    # Discover tests from TEST_OUTPUT_DIR using POST_BUILD mode.
    # CROSSCOMPILING_EMULATOR is used to run the binary with the MPI launcher.
    gtest_discover_tests(${TARGET}
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 300
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
        PROPERTIES
            PROCESSORS ${NUM_PROCESSES}
            TIMEOUT 300
            RUN_SERIAL ON
    )

    # When TEST_OUTPUT_DIR is external, copy and fix the discovery .cmake files
    if(NOT SPECFEM_TESTDIR_DEFAULT)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DINPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}[1]_tests.cmake
                -DOUTPUT_FILE=${TEST_OUTPUT_DIR}/${TARGET}_tests.cmake
                -DOLD_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                -DNEW_PATH=${TEST_OUTPUT_DIR}
                -P ${CMAKE_CURRENT_BINARY_DIR}/copy_test_cmake.cmake
            COMMENT "Copying ${TARGET} test discovery files to ${TEST_OUTPUT_DIR}"
        )
    endif()
endfunction()

# MPI-only test executables

add_executable(
  mesh_mpi_dim3_tests
  mesh/mpi/dim3/adjacency_graph.cpp
  mesh/mpi/dim3/runner.cpp
)

target_link_libraries(
  mesh_mpi_dim3_tests
  specfem::mesh
  specfem_environment
  specfem::io
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  mesh_mpi_dim2_tests
  mesh/mpi/dim2/adjacency_graph.cpp
  mesh/mpi/dim2/runner.cpp
)

target_link_libraries(
  mesh_mpi_dim2_tests
  specfem::mesh
  specfem_environment
  specfem::io
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  io_mesh_mpi_tests
  io/mesh/dim3/read_mesh.cpp
  io/mesh/dim3/runner.cpp
)

target_link_libraries(
  io_mesh_mpi_tests
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  io_mesh_mpi_dim2_tests
  io/mesh/dim2/read_mesh.cpp
  io/mesh/dim2/runner.cpp
)

target_link_libraries(
  io_mesh_mpi_dim2_tests
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

# specfem::MPI class tests

add_executable(
  mpi_standard_tests
  mpi/mpi_standard_tests.cpp
)

target_link_libraries(
  mpi_standard_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_2of4_tests
  mpi/mpi_subset_2of4_tests.cpp
)

target_link_libraries(
  mpi_subset_2of4_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_1of4_tests
  mpi/mpi_subset_1of4_tests.cpp
)

target_link_libraries(
  mpi_subset_1of4_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_all_tests
  mpi/mpi_subset_all_tests.cpp
)

target_link_libraries(
  mpi_subset_all_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  algorithms_locate_point_mpi_dim2_tests
  algorithms/mpi/dim2/locate_point_test.cpp
)

target_link_libraries(
  algorithms_locate_point_mpi_dim2_tests
  specfem::algorithms
  specfem::assembly
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  algorithms_locate_point_mpi_dim3_tests
  algorithms/mpi/dim3/locate_point_test.cpp
)

target_link_libraries(
  algorithms_locate_point_mpi_dim3_tests
  specfem::algorithms
  specfem::assembly
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  assembly_mpi_dim3_tests
  assembly/mpi/dim3/fixture.cpp
  assembly/mpi/dim3/communication_group/communication_group.cpp
  assembly/mpi/dim3/communication_pattern/communication_pattern.cpp
  assembly/mpi/dim3/mass_matrix/mass_matrix.cpp
  assembly/mpi/dim3/reordering/reordering.cpp
  assembly/mpi/dim3/runner.cpp
)

target_compile_definitions(assembly_mpi_dim3_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})
set_target_properties(assembly_mpi_dim3_tests PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  assembly_mpi_dim3_tests
  specfem::mesh
  specfem::assembly
  specfem::quadrature
  specfem_environment
  specfem::io
  specfem::utilities
  specfem::enums
  specfem::element
  MPI::MPI_CXX
  Kokkos::kokkos
  gtest_main
  -lpthread -lm
)

# ==============================================================================
# 3D MPI Newmark displacement tests (one executable per process count)
# ==============================================================================
# A gtest executable runs under a single `mpirun -n`, so per-test process counts
# require one executable per size. The set of sizes is derived from the per-test
# core counts in tests_mpi.yaml (the single source of truth): one executable is
# built per distinct value. Adding a test at a new size needs only a yaml edit.
set(_tests_mpi_yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/mpi/dim3/tests_mpi.yaml)
# Re-run CMake configure when the yaml changes so new sizes are picked up.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_tests_mpi_yaml})
# Match "  <name>: <int>" lines under tests3d: and collect the distinct integers.
file(STRINGS ${_tests_mpi_yaml} _tests_mpi_lines
  REGEX "^[ \t]+[A-Za-z0-9_]+[ \t]*:[ \t]*[0-9]+[ \t]*$")
set(DISPLACEMENT_NEWMARK_3D_MPI_NPROCS "")
foreach(_line IN LISTS _tests_mpi_lines)
  string(REGEX REPLACE "^.*:[ \t]*([0-9]+)[ \t]*$" "\\1" _nproc "${_line}")
  list(APPEND DISPLACEMENT_NEWMARK_3D_MPI_NPROCS ${_nproc})
endforeach()
list(REMOVE_DUPLICATES DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
if(NOT DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  message(FATAL_ERROR
    "No 'name: <nproc>' entries found in ${_tests_mpi_yaml}")
endif()

set(DISPLACEMENT_MPI_TARGETS "")
foreach(nproc IN LISTS DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  set(_tgt displacement_newmark_3d_mpi${nproc}_tests)
  add_executable(${_tgt}
    displacement_tests/Newmark/mpi/dim3/newmark_tests.cpp
  )
  target_compile_definitions(${_tgt} PRIVATE SPECFEM_MPI_TEST_NPROC=${nproc})
  target_link_libraries(${_tgt}
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
    MPI::MPI_CXX
    ${BOOST_LIBS}
    -lpthread -lm
  )

  # Copy the (single) test list to TEST_OUTPUT_DIR so it is available when
  # gtest_discover_tests runs the binary (mirrors the serial.cmake pattern). Each
  # per-size executable reads the same file and filters to its own size.
  add_custom_command(TARGET ${_tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_tests_mpi_yaml}
        ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3/tests_mpi.yaml
    COMMENT "Moving ${_tgt} test list to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3"
  )

  list(APPEND DISPLACEMENT_MPI_TARGETS ${_tgt})
endforeach()

# MPI test targets (4 processes)
set(MPI_TEST_TARGETS_4PROCS
  mesh_mpi_dim3_tests
  mesh_mpi_dim2_tests
  io_mesh_mpi_tests
  io_mesh_mpi_dim2_tests
  mpi_standard_tests
  mpi_subset_2of4_tests
  mpi_subset_1of4_tests
  mpi_subset_all_tests
  algorithms_locate_point_mpi_dim2_tests
  algorithms_locate_point_mpi_dim3_tests
  assembly_mpi_dim3_tests
)

# Expose MPI test targets for use in CMakeLists.txt (ALL_TEST_TARGETS) and
# the registration loop below.
set(MPI_TEST_TARGETS ${MPI_TEST_TARGETS_4PROCS} ${DISPLACEMENT_MPI_TARGETS})

# Setup test script writer (needed for external TEST_OUTPUT_DIR path-fix)
specfem_write_copy_test_cmake_script()

# Register MPI tests using helper function
foreach(test_target IN LISTS MPI_TEST_TARGETS_4PROCS)
  add_mpi_test(${test_target} 4)
endforeach()

# Register the displacement MPI tests at their per-size process counts.
foreach(nproc IN LISTS DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  add_mpi_test(displacement_newmark_3d_mpi${nproc}_tests ${nproc})
endforeach()

# Note: CTestTestfile.cmake generation and data directories (data, mesh) are
# finalized by serial.cmake via specfem_finalize_test_targets, which covers
# both in-tree (symlinks) and external TEST_OUTPUT_DIR (install) cases.
