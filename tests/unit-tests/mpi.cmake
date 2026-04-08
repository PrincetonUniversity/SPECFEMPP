# MPI-specific test definitions and setup
# This file contains MPI-only test executables and their discovery registration.

# Include GoogleTest module for gtest_add_tests
include(GoogleTest)

# ==============================================================================
# Helper function to register an MPI test with configurable processor count
# and test properties
# ==============================================================================
# Usage: add_mpi_test(TARGET NUM_PROCESSES)
# Example: add_mpi_test(mesh_mpi_dim3_tests 4)
# Uses gtest_add_tests to scan source files at configure time (no binary execution),
# avoiding duplicate test registration from multiple MPI ranks.
function(add_mpi_test TARGET NUM_PROCESSES)
    message(STATUS "Registering MPI test: ${TARGET} with ${NUM_PROCESSES} processes")

    # CROSSCOMPILING_EMULATOR prefixes test execution with the MPI launcher.
    # gtest_add_tests scans sources at configure time so this is only used at run time.
    set_target_properties(${TARGET} PROPERTIES
        CROSSCOMPILING_EMULATOR "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${NUM_PROCESSES}"
    )

    # Scan source files to find Google Test cases and register them individually.
    # Unlike gtest_discover_tests, this never executes the binary for discovery.
    gtest_add_tests(
        TARGET ${TARGET}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests
        TEST_LIST DISCOVERED_TESTS
    )

    # Set properties on all discovered tests
    set_tests_properties(${DISCOVERED_TESTS} PROPERTIES
        PROCESSORS ${NUM_PROCESSES}
        TIMEOUT 300
        RUN_SERIAL ON
    )
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

# MPI test targets
set(MPI_TEST_TARGETS
  mesh_mpi_dim3_tests
  mesh_mpi_dim2_tests
  io_mesh_mpi_tests
  io_mesh_mpi_dim2_tests
)

# Register MPI tests using helper function
foreach(test_target IN LISTS MPI_TEST_TARGETS)
  add_mpi_test(${test_target} 4)
endforeach()

# Link test data directories for MPI tests
set(MPI_LINK_DIRS
  data
  mesh
)

foreach(dir_name IN LISTS MPI_LINK_DIRS)
    file(CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
        ${CMAKE_CURRENT_BINARY_DIR}/${dir_name}
        SYMBOLIC)
endforeach()
