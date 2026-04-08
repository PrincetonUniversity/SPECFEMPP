# ==============================================================================
# Helper function to register an MPI test with configurable processor count
# and test properties
# ==============================================================================
# Usage: add_mpi_test(TEST_NAME NUM_PROCESSES)
# Example: add_mpi_test(mesh_mpi_dim3_tests 4)
function(add_mpi_test TEST_NAME NUM_PROCESSES)
    message(STATUS "Registering MPI test: ${TEST_NAME} with ${NUM_PROCESSES} processes")

    # Create test command with MPI launcher
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCESSES}
            ${CMAKE_BINARY_DIR}/tests/unit-tests/${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests
    )

    # Set test properties for resource management and parallelization
    set_tests_properties(${TEST_NAME} PROPERTIES
        # Number of processors reserved for this test
        PROCESSORS ${NUM_PROCESSES}
        # Timeout (in seconds) - adjust as needed
        TIMEOUT 300
        # Run serially if multiple processes to avoid compute conflicts
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
  io_mesh_mpi_tests
  io/mesh/read_mesh.cpp
  io/mesh/runner.cpp
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

# Register MPI tests using helper function
# Format: add_mpi_test(test_name num_processes)
enable_testing()

add_mpi_test(mesh_mpi_dim3_tests 4)
add_mpi_test(io_mesh_mpi_tests 4)

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
