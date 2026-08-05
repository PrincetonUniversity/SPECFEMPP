# Root of the SPECFEM++ test tree.
#
# Shared setup for every test suite. Each suite (unit-tests, integration-tests)
# declares its own targets with specfem_add_test() and its own data directories
# with specfem_add_test_data(); everything that is per-output-directory rather
# than per-suite lives here.

set(CMAKE_CXX_STANDARD 20)

# Include the GoogleTest framework
include("${CMAKE_SOURCE_DIR}/cmake/googletest.cmake")
include(GoogleTest)

include(test_registration.cmake)

enable_testing()

# tests/ holds test_macros.hpp and SPECFEM_Environment.hpp, included by tests in
# every suite below. Inherited by all subdirectories.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# The only place SPECFEMPP_TEST_DIR is read. Defines SPECFEM_TEST_OUTPUT_DIR,
# absolute, for every suite below.
specfem_init_tests()

# Common library: test environment setup. Links gtest_main publicly, so tests that
# link specfem_environment do not need to repeat it.
add_library(
  specfem_environment
  SPECFEM_Environment.cpp
)

target_include_directories(specfem_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  specfem_environment
  PUBLIC
  gtest_main
  specfem_program
)

add_subdirectory(unit-tests)
add_subdirectory(integration-tests)

# Must come after every suite: emits the single CTestTestfile.cmake and the single
# install_test_data target covering all of them.
specfem_finalize_tests()
