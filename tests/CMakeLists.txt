# Common CMake configuration for unit tests
# This file contains shared setup, libraries, and utilities used by both
# serial (non-MPI) and MPI test builds.

set(CMAKE_CXX_STANDARD 20)

# Include the GoogleTest framework
include("${CMAKE_SOURCE_DIR}/cmake/googletest.cmake")
include(GoogleTest)

# Explicitly set binary output directory for tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests)

include_directories(.)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Some of the writing tests need to write somewhere and we don't want that
# to be in the source directory
if (DEFINED SPECFEMPP_TEST_DIR)
  set(TEST_OUTPUT_DIR ${SPECFEMPP_TEST_DIR})
  set(SPECFEM_TESTDIR_DEFAULT FALSE CACHE BOOL "SPECFEM++ Test directory default flag" FORCE)
else()
  set(TEST_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(SPECFEM_TESTDIR_DEFAULT TRUE CACHE BOOL "SPECFEM++ Test directory default flag" FORCE)
endif()

enable_testing()

# Add test output directory to clean target
set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${TEST_OUTPUT_DIR}")

# Include the GoogleTest framework
include("${CMAKE_SOURCE_DIR}/cmake/googletest.cmake")
include(GoogleTest)

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

include(test_registration.cmake)

add_subdirectory(unit-tests)
add_subdirectory(integration-tests)
