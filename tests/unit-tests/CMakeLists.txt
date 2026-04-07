cmake_minimum_required(VERSION 3.17.5)

# Include common test configuration and utilities
# This handles C++ standard, GoogleTest setup, and shared test libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake)

# Include test definitions based on build configuration
if(SPECFEM_ENABLE_MPI)
  # Build MPI-only tests
  include(${CMAKE_CURRENT_SOURCE_DIR}/mpi.cmake)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/serial.cmake)

# Data subdirectories for test provenance
add_subdirectory(data/dim2/ncmarked_conforming_grid_curved/provenance)
add_subdirectory(data/dim2/ncmarked_conforming_grid_flat/provenance)
