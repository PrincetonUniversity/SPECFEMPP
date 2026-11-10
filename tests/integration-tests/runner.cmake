# Runner executables for the 3D simulations listed in configurations/tests.yaml
# (see runner.cpp). A gtest executable runs under a single `mpirun -n`, so every
# distinct process count needs its own executable; the counts are derived from
# the yaml, so a case at a new size needs only a yaml edit.
# Included from tests/integration-tests/CMakeLists.txt.

set(_tests_yaml ${CMAKE_CURRENT_SOURCE_DIR}/configurations/tests.yaml)
# Re-run CMake configure when the yaml changes so new cases and sizes are picked up.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_tests_yaml})

# Lines are either "<section>:" or "  <case>: <nproc>". The case names are what
# makes each case its own CTest entry for the MPI executables (PARAM_CASES below);
# registering the TEST_P alone would collapse every case at a given size into one
# entry, so a failure could not be attributed or re-run.
file(STRINGS ${_tests_yaml} _tests_yaml_lines)
set(RUNNER_NPROCS "")
set(_section "")
foreach(_line IN LISTS _tests_yaml_lines)
  if(_line MATCHES "^([A-Za-z0-9_]+):[ \t]*$")
    set(_section "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^[ \t]+([A-Za-z0-9_]+)[ \t]*:[ \t]*([0-9]+)[ \t]*$")
    if(_section STREQUAL "explicit_solver")
      list(APPEND RUNNER_NPROCS ${CMAKE_MATCH_2})
      list(APPEND _runner_cases_${CMAKE_MATCH_2} ${CMAKE_MATCH_1})
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES RUNNER_NPROCS)
if(NOT RUNNER_NPROCS)
  message(FATAL_ERROR "No '<case>: <nproc>' entries found in ${_tests_yaml}")
endif()

set(_runner_libs
  specfem::program
  specfem::injection_methods
  specfem::runtime_configuration
  specfem::io
  ${DISPLACEMENT_TEST_LIBS}
)

foreach(nproc IN LISTS RUNNER_NPROCS)
  if(nproc EQUAL 1)
    specfem_add_test(integration_tests
      NO_UNITY
      SOURCES     runner.cpp common.cpp
      DEFINITIONS SPECFEM_TEST_NPROC=1
      LIBRARIES   ${_runner_libs}
    )
  elseif(SPECFEM_ENABLE_MPI)
    # The fixture and test names are spelled out here rather than derived, so that
    # renaming them fails the configure (specfem_add_test checks PARAM_CASES against
    # the TEST_P declaration it greps) instead of leaving the cases unregistered.
    set(_param_cases "")
    foreach(_case IN LISTS _runner_cases_${nproc})
      list(APPEND _param_cases "Runner.Run/${_case}")
    endforeach()
    specfem_add_test(integration_tests_mpi${nproc}
      NO_UNITY
      MPI_RANKS   ${nproc}
      SOURCES     runner.cpp common.cpp
      DEFINITIONS SPECFEM_TEST_NPROC=${nproc}
      PARAM_CASES ${_param_cases}
      LIBRARIES   ${_runner_libs} MPI::MPI_CXX
    )
  else()
    message(STATUS "Integration tests on ${nproc} processes skipped: MPI is disabled")
  endif()
endforeach()
