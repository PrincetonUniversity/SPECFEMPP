cmake_minimum_required(VERSION 3.17.5)

# Test binaries stay in the build tree. scripts/coverage.smk globs executables from
# this directory, so do not move it.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests)

include_directories(.)

# Common library: mesh utilities mapping
add_library(
  mesh_utilities_mapping
  mesh_utilities/mapping.cpp
)

target_link_libraries(
  mesh_utilities_mapping
  PUBLIC
  Kokkos::kokkos
)

include(${CMAKE_CURRENT_SOURCE_DIR}/serial.cmake)

if(SPECFEM_ENABLE_MPI)
  include(${CMAKE_CURRENT_SOURCE_DIR}/mpi.cmake)
endif()

# Data directories the tests read, relative to their working directory
specfem_add_test_data(
  algorithms
  assembly
  assembly_mesh
  data
  fortran_io
  io
  mesh
  policies
)

# Data subdirectories for test provenance
add_subdirectory(data/dim2/ncmarked_conforming_grid_curved/provenance)
add_subdirectory(data/dim2/ncmarked_conforming_grid_flat/provenance)
