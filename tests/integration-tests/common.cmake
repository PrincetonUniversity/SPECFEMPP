
# Explicitly set binary output directory for tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration-tests)

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
