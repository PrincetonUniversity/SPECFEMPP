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

# Combine serial and MPI test targets for finalization
if(SPECFEM_ENABLE_MPI)
  set(ALL_TEST_TARGETS ${SERIAL_TEST_TARGETS} ${MPI_TEST_TARGETS})
else()
  set(ALL_TEST_TARGETS ${SERIAL_TEST_TARGETS})
endif()

# Finalize test setup: generate CTestTestfile.cmake and setup data directories
specfem_finalize_test_targets(ALL_TEST_TARGETS SERIAL_LINK_DIRS)

# Data subdirectories for test provenance
add_subdirectory(data/dim2/ncmarked_conforming_grid_curved/provenance)
add_subdirectory(data/dim2/ncmarked_conforming_grid_flat/provenance)
