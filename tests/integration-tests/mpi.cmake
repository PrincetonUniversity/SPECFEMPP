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

set(DISPLACEMENT_MPI_TARGETS "")
foreach(nproc IN LISTS DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  set(_tgt displacement_newmark_3d_mpi${nproc}_tests)
  add_executable(${_tgt}
    displacement_tests/Newmark/mpi/dim3/newmark_tests.cpp
  )
  target_compile_definitions(${_tgt} PRIVATE SPECFEM_MPI_TEST_NPROC=${nproc})
  target_link_libraries(${_tgt}
    specfem::quadrature
    specfem::mesh
    yaml-cpp
    specfem_environment
    specfem::assembly
    specfem::runtime_configuration
    timescheme
    point
    specfem::algorithms
    specfem::solver
    specfem::periodic_tasks
    MPI::MPI_CXX
    ${BOOST_LIBS}
    -lpthread -lm
  )

  # Copy the (single) test list to TEST_OUTPUT_DIR so it is available when
  # gtest_discover_tests runs the binary (mirrors the serial.cmake pattern). Each
  # per-size executable reads the same file and filters to its own size.
  add_custom_command(TARGET ${_tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_tests_mpi_yaml}
        ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3/tests_mpi.yaml
    COMMENT "Moving ${_tgt} test list to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/mpi/dim3"
  )

  list(APPEND DISPLACEMENT_MPI_TARGETS ${_tgt})
endforeach()

set(MPI_TEST_TARGETS ${MPI_TEST_TARGETS_4PROCS} ${DISPLACEMENT_MPI_TARGETS})

# Register the displacement MPI tests at their per-size process counts.
foreach(nproc IN LISTS DISPLACEMENT_NEWMARK_3D_MPI_NPROCS)
  add_mpi_test(displacement_newmark_3d_mpi${nproc}_tests ${nproc})
endforeach()

# Note: CTestTestfile.cmake generation and data directories (data, mesh) are
# finalized by serial.cmake via specfem_finalize_test_targets, which covers
# both in-tree (symlinks) and external TEST_OUTPUT_DIR (install) cases.
