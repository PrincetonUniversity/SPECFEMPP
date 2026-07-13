# ==============================================================================
# 3D MPI Newmark displacement tests (one executable per process count)
# ==============================================================================
# A gtest executable runs under a single `mpirun -n`, so per-test process counts
# require one executable per size. The set of sizes is derived from the per-test
# core counts in tests_mpi.yaml (the single source of truth): one executable is
# built per distinct value. Adding a test at a new size needs only a yaml edit.
set(_tests_mpi_yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/mpi/dim3/tests_mpi.yaml)
# Re-run CMake configure when the yaml changes so new sizes are picked up.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_tests_mpi_yaml})
# Match "  <name>: <int>" lines under tests3d: and collect the distinct integers.
file(STRINGS ${_tests_mpi_yaml} _tests_mpi_lines
  REGEX "^[ \t]+[A-Za-z0-9_]+[ \t]*:[ \t]*[0-9]+[ \t]*$")
set(DISPLACEMENT_NEWMARK_3D_MPI_NPROCS "")
foreach(_line IN LISTS _tests_mpi_lines)
  string(REGEX REPLACE "^.*:[ \t]*([0-9]+)[ \t]*$" "\\1" _nproc "${_line}")
  list(APPEND DISPLACEMENT_NEWMARK_3D_MPI_NPROCS ${_nproc})
endforeach()
list(REMOVE_DUPLICATES DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
if(NOT DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  message(FATAL_ERROR
    "No 'name: <nproc>' entries found in ${_tests_mpi_yaml}")
endif()

# Each per-size executable reads the same test list and filters to its own size.
foreach(nproc IN LISTS DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  specfem_add_test(displacement_newmark_3d_mpi${nproc}_tests
    MPI_RANKS   ${nproc}
    SOURCES     displacement_tests/Newmark/mpi/dim3/newmark_tests.cpp
    DEFINITIONS SPECFEM_MPI_TEST_NPROC=${nproc}
    LIBRARIES   ${DISPLACEMENT_TEST_LIBS} MPI::MPI_CXX
  )
endforeach()
