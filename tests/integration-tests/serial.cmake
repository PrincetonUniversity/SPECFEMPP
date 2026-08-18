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
