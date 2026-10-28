# End-to-end wavefield-injection tests: specfem 3d with the plane-wave example
# plugin on an all-Stacey box (see injection/README.md). They need a built
# plugin, so they are registered only when SPECFEM_INJECTION_TEST_PLUGIN names
# one; the Jenkins plugin pipeline (.jenkins/plugin_checks.gvy) sets it.
# Included from tests/integration-tests/CMakeLists.txt.

set(SPECFEM_INJECTION_TEST_PLUGIN "" CACHE FILEPATH
  "Built plane-wave injection plugin (libplane_wave.so); enables the end-to-end injection tests")

if(NOT SPECFEM_INJECTION_TEST_PLUGIN)
  message(STATUS "End-to-end injection tests skipped: SPECFEM_INJECTION_TEST_PLUGIN is not set")
elseif(NOT EXISTS "${SPECFEM_INJECTION_TEST_PLUGIN}")
  message(FATAL_ERROR
    "SPECFEM_INJECTION_TEST_PLUGIN=${SPECFEM_INJECTION_TEST_PLUGIN} does not exist")
else()
  set(_injection_e2e_libs
    specfem::program
    specfem::injection_methods
    specfem::runtime_configuration
    specfem::io
    ${DISPLACEMENT_TEST_LIBS}
  )

  specfem_add_test(injection_e2e_tests
    NO_UNITY
    LABELS      injection_e2e
    SOURCES     injection/injection_tests.cpp
    DEFINITIONS "SPECFEM_INJECTION_TEST_PLUGIN=\"${SPECFEM_INJECTION_TEST_PLUGIN}\""
    LIBRARIES   ${_injection_e2e_libs}
    TIMEOUT     1200
  )

  if(SPECFEM_ENABLE_MPI)
    # One executable per rank count; the cases per count are those with that
    # nproc in injection/tests.yaml.
    set(_injection_e2e_cases_2 p_wave_mpi2)
    set(_injection_e2e_cases_4 p_wave_mpi4 sv_wave_mpi4)
    foreach(nproc 2 4)
      set(_param_cases "")
      foreach(_case IN LISTS _injection_e2e_cases_${nproc})
        list(APPEND _param_cases "InjectionEndToEnd.Run/${_case}")
      endforeach()
      specfem_add_test(injection_e2e_mpi${nproc}_tests
        NO_UNITY
        MPI_RANKS   ${nproc}
        LABELS      injection_e2e
        SOURCES     injection/injection_tests.cpp
        DEFINITIONS "SPECFEM_INJECTION_TEST_PLUGIN=\"${SPECFEM_INJECTION_TEST_PLUGIN}\""
                    SPECFEM_INJECTION_TEST_NPROC=${nproc}
        PARAM_CASES ${_param_cases}
        LIBRARIES   ${_injection_e2e_libs} MPI::MPI_CXX
        TIMEOUT     1200
      )
    endforeach()
  endif()

  specfem_add_test_data(injection)
endif()
