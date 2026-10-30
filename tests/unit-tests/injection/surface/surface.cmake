# Tests for the assembled injection surface (core/specfem/assembly) and the
# points built from it for an injection plugin (core/specfem/injection/points).
# Included from tests/unit-tests/CMakeLists.txt.
#
# The databases come from xmeshfem3D with injection faces declared; the
# provider-session test reuses the session's mock plugin.

specfem_add_test(injection_surface_tests
  NO_UNITY
  LABELS      injection
  SOURCES     ${CMAKE_CURRENT_LIST_DIR}/injection_surface_tests.cpp
              ${CMAKE_CURRENT_LIST_DIR}/../session/mock_plugin.cpp
              ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  LIBRARIES   specfem::injection_points
              specfem::io
              specfem::mesh
              specfem::quadrature
              specfem_environment
              gmock
              Kokkos::kokkos
)
