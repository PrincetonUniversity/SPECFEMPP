# Tests for the injection provider session (core/specfem/injection/session).
# Included from tests/unit-tests/CMakeLists.txt.
#
# The session drives a mock plugin registered through the static path, so no
# shared library, mesh, or kernel is involved.

specfem_add_test(injection_session_tests
  LABELS      injection
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/session_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/mock_plugin.cpp
              ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  LIBRARIES   specfem::injection_session
              specfem_environment
              gmock
              Kokkos::kokkos
)

if(SPECFEM_ENABLE_MPI)
  specfem_add_test(injection_session_mpi_tests
    MPI_RANKS   2
    LABELS      injection
    SOURCES     ${CMAKE_CURRENT_LIST_DIR}/mpi_session_tests.cpp
                ${CMAKE_CURRENT_LIST_DIR}/mock_plugin.cpp
                ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
    LIBRARIES   specfem::injection_session
                specfem_environment
                gmock
                MPI::MPI_CXX
                Kokkos::kokkos
  )
endif()
