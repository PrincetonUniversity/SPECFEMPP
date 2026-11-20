# Tests for the injection lifecycle: the parameters.injection section, the
# method registry, and a forward 3D run driving a mock plugin through a
# recording method. Included from tests/unit-tests/CMakeLists.txt.

specfem_add_test(injection_lifecycle_tests
  NO_UNITY
  LABELS      injection
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/config_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/program_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/../session/mock_plugin.cpp
              ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  LIBRARIES   specfem::program
              specfem::injection_methods
              specfem::runtime_configuration
              specfem_environment
              gmock
              Kokkos::kokkos
)
