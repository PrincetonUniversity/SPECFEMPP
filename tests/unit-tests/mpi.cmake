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
  assembly_mpi_dim3_tests
  assembly/mpi/dim3/fixture.cpp
  assembly/mpi/dim3/communication_group.cpp
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
  assembly_mpi_dim3_tests
)

# Setup test script writer (needed for external TEST_OUTPUT_DIR path-fix)
specfem_write_copy_test_cmake_script()

# Register MPI tests using helper function
list(APPEND MPI_TEST_TARGETS assembly_mpi_dim3_tests)
foreach(test_target IN LISTS MPI_TEST_TARGETS)
  add_mpi_test(${test_target} 4)
endforeach()

# Note: CTestTestfile.cmake generation and data directories (data, mesh) are
# finalized by serial.cmake via specfem_finalize_test_targets, which covers
# both in-tree (symlinks) and external TEST_OUTPUT_DIR (install) cases.
