# Runner executables for the 3D simulations listed in configurations/tests.yaml
# (see runner.cpp). A gtest executable runs under a single `mpirun -n`, so every
# distinct process count of a section needs its own executable; the counts are
# derived from the yaml, so a case at a new size needs only a yaml edit.
# Included from tests/integration-tests/CMakeLists.txt.

# The injection section runs the cases with the plane-wave example plugin. They
# need a built plugin, so they are registered only when this names one; the Jenkins
# plugin pipeline (.jenkins/plugin_checks.gvy) sets it.
set(SPECFEM_INJECTION_TEST_PLUGIN "" CACHE FILEPATH
  "Built plane-wave injection plugin (libplane_wave.so); enables the injection tests")

set(_tests_yaml ${CMAKE_CURRENT_SOURCE_DIR}/configurations/tests.yaml)
# Re-run CMake configure when the yaml changes so new cases and sizes are picked up.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_tests_yaml})

# Lines are either "<section>:" or "  <case>: <nproc>". The case names are what
# makes each case its own CTest entry for the MPI executables (PARAM_CASES below);
# registering the TEST_P alone would collapse every case at a given size into one
# entry, so a failure could not be attributed or re-run.
file(STRINGS ${_tests_yaml} _tests_yaml_lines)
set(_runner_sections "")
set(_section "")
foreach(_line IN LISTS _tests_yaml_lines)
  if(_line MATCHES "^([A-Za-z0-9_]+):[ \t]*$")
    set(_section "${CMAKE_MATCH_1}")
    list(APPEND _runner_sections ${_section})
  elseif(_line MATCHES "^[ \t]+([A-Za-z0-9_]+)[ \t]*:[ \t]*([0-9]+)[ \t]*$")
    list(APPEND _runner_nprocs_${_section} ${CMAKE_MATCH_2})
    list(APPEND _runner_cases_${_section}_${CMAKE_MATCH_2} ${CMAKE_MATCH_1})
  endif()
endforeach()
if(NOT _runner_sections)
  message(FATAL_ERROR "No '<section>:' found in ${_tests_yaml}")
endif()

set(_runner_libs
  specfem::program
  specfem::injection_methods
  specfem::runtime_configuration
  specfem::io
  ${DISPLACEMENT_TEST_LIBS}
)

foreach(_section IN LISTS _runner_sections)
  set(_section_definitions "")
  set(_section_options "")
  set(_section_enabled TRUE)

  if(_section STREQUAL "injection")
    if(NOT SPECFEM_INJECTION_TEST_PLUGIN)
      message(STATUS "Injection tests skipped: SPECFEM_INJECTION_TEST_PLUGIN is not set")
      set(_section_enabled FALSE)
    elseif(NOT EXISTS "${SPECFEM_INJECTION_TEST_PLUGIN}")
      message(FATAL_ERROR
        "SPECFEM_INJECTION_TEST_PLUGIN=${SPECFEM_INJECTION_TEST_PLUGIN} does not exist")
    else()
      list(APPEND _section_definitions
        "SPECFEM_INJECTION_TEST_PLUGIN=\"${SPECFEM_INJECTION_TEST_PLUGIN}\"")
      # One label: gtest_discover_tests() flattens a list-valued LABELS, so a
      # second label would be silently dropped.
      set(_section_options LABELS injection_e2e TIMEOUT 1200)
    endif()
  endif()

  if(_section_enabled)
    if(NOT _runner_nprocs_${_section})
      message(FATAL_ERROR "No '<case>: <nproc>' entries for '${_section}' in ${_tests_yaml}")
    endif()
    list(REMOVE_DUPLICATES _runner_nprocs_${_section})

    foreach(nproc IN LISTS _runner_nprocs_${_section})
      set(_definitions
        SPECFEM_TEST_NPROC=${nproc}
        "SPECFEM_TEST_SECTION=\"${_section}\""
        ${_section_definitions})

      if(nproc EQUAL 1)
        specfem_add_test(${_section}_tests
          NO_UNITY
          ${_section_options}
          SOURCES     runner.cpp common.cpp
          DEFINITIONS ${_definitions}
          LIBRARIES   ${_runner_libs}
        )
      elseif(SPECFEM_ENABLE_MPI)
        # The fixture and test names are spelled out here rather than derived, so
        # that renaming them fails the configure (specfem_add_test checks PARAM_CASES
        # against the TEST_P declaration it greps) instead of leaving the cases
        # unregistered.
        set(_param_cases "")
        foreach(_case IN LISTS _runner_cases_${_section}_${nproc})
          list(APPEND _param_cases "Runner.Run/${_case}")
        endforeach()
        specfem_add_test(${_section}_tests_mpi${nproc}
          NO_UNITY
          MPI_RANKS   ${nproc}
          ${_section_options}
          SOURCES     runner.cpp common.cpp
          DEFINITIONS ${_definitions}
          PARAM_CASES ${_param_cases}
          LIBRARIES   ${_runner_libs} MPI::MPI_CXX
        )
      else()
        message(STATUS "${_section} tests on ${nproc} processes skipped: MPI is disabled")
      endif()
    endforeach()
  endif()
endforeach()
