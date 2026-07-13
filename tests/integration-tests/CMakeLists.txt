cmake_minimum_required(VERSION 3.17.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration-tests)

include_directories(.)

include(${CMAKE_CURRENT_SOURCE_DIR}/serial.cmake)

if(SPECFEM_ENABLE_MPI)
  include(${CMAKE_CURRENT_SOURCE_DIR}/mpi.cmake)
endif()

# Data directories the tests read, relative to their working directory
specfem_add_test_data(
  displacement_tests
)
