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

# Injection plugin tests build real shared-library plugins, which need Linux or
# macOS.
if(UNIX)
  # Injection plugin SDK tests (sdk/)
  include(${CMAKE_CURRENT_SOURCE_DIR}/plugin_sdk/plugin_sdk.cmake)

  # Injection plugin loader tests (core/specfem/injection)
  include(${CMAKE_CURRENT_SOURCE_DIR}/injection/loader/loader.cmake)
endif()

# Injection provider session tests use a statically registered mock plugin, so
# they run on every platform.
include(${CMAKE_CURRENT_SOURCE_DIR}/injection/session/session.cmake)

# Injection surface and plugin points, built from meshed databases
include(${CMAKE_CURRENT_SOURCE_DIR}/injection/surface/surface.cmake)

# Injection lifecycle: configuration, method registry and a forward run
include(${CMAKE_CURRENT_SOURCE_DIR}/injection/lifecycle/lifecycle.cmake)

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
