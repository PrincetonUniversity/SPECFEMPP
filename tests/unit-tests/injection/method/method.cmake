# Tests for the pointwise injection framework (core/specfem/injection/method):
# quantity lists, the point harness, and the injection kernel driven by the
# session's mock plugin. Included from tests/unit-tests/CMakeLists.txt.

specfem_add_test(injection_method_tests
  NO_UNITY
  LABELS      injection
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/quantity_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/requirement_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/point_harness_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/kernel_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/../session/mock_plugin.cpp
              ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  LIBRARIES   specfem::injection_methods
              specfem::io
              specfem::mesh
              specfem::quadrature
              specfem_environment
              gmock
              Kokkos::kokkos
)
