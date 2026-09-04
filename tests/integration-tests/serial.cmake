# Libraries shared by every Newmark displacement test, serial and MPI alike.
set(DISPLACEMENT_TEST_LIBS
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
  ${BOOST_LIBS}
  -lpthread -lm
)

specfem_add_test(displacement_newmark_2d_tests
  SOURCES   displacement_tests/Newmark/dim2/newmark_tests.cpp
  LIBRARIES ${DISPLACEMENT_TEST_LIBS}
)

specfem_add_test(displacement_newmark_3d_tests
  SOURCES   displacement_tests/Newmark/dim3/newmark_tests.cpp
  LIBRARIES ${DISPLACEMENT_TEST_LIBS}
)

# Multi-minute wall clock (full explicit reference runs + implicit solves), and the
# whole body is behind #ifdef SPECFEM_ENABLE_TRILINOS -- see the note in
# tests/unit-tests/serial.cmake. Exclude locally with `ctest -LE TRILINOS`, which
# replaces the `long` label this used to carry: this was the only test with it, and
# it is Trilinos-only, so the two selected exactly the same test.
#
# Keep this to ONE label. gtest_discover_tests() flattens a list-valued property when
# it generates the discovery file, so `LABELS a b` reaches set_tests_properties() as
# two key/value pairs -- binding LABELS to `a` and treating `b` as a property name.
# The second label is silently dropped, with no CMake or CTest diagnostic.
specfem_add_test(implicit_newmark_3d_tests
  SOURCES   displacement_tests/ImplicitNewmark/dim3/implicit_newmark_tests.cpp
  LIBRARIES ${DISPLACEMENT_TEST_LIBS} specfem::linear_system
  TIMEOUT   1800
  LABELS    TRILINOS
)
