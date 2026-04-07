# Common CMake configuration for unit tests
# This file contains shared setup, libraries, and utilities used by both
# serial (non-MPI) and MPI test builds.

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)

# Include the GoogleTest framework
include("${CMAKE_SOURCE_DIR}/cmake/googletest.cmake")

# Explicitly set binary output directory for tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests)

include_directories(.)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Some of the writing tests need to write somewhere and we don't want that
# to be in the source directory
set(TEST_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

enable_testing()

# Common library: Test environment setup
add_library(
  specfem_environment
  SPECFEM_Environment.cpp
)

target_link_libraries(
  specfem_environment
  gtest_main
  specfem_program
)

# Common library: Mesh utilities mapping
add_library(
  mesh_utilities_mapping
  mesh_utilities/mapping.cpp
)

target_link_libraries(
  mesh_utilities_mapping
  Kokkos::kokkos
)
