# MPI-only unit test executables.
#
# MPI_RANKS is what makes a test run under `mpirun -n <ranks>`, and it is declared
# alongside the target rather than in a separate list far below -- the previous
# rank lists were consumed before they were defined, so none of the 4-rank tests
# were ever registered with CTest.

# Libraries shared by the specfem::MPI class tests
set(MPI_CLASS_TEST_LIBS
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

# Libraries shared by the 3D distributed-assembly tests
set(ASSEMBLY_MPI_TEST_LIBS
  specfem::mesh
  specfem::assembly
  specfem::quadrature
  specfem_environment
  specfem::io
  specfem::utilities
  specfem::enums
  specfem::element
  MPI::MPI_CXX
  Kokkos::kokkos
  gtest_main
  -lpthread -lm
)

specfem_add_test(mesh_mpi_dim3_tests
  MPI_RANKS 4
  SOURCES   mesh/mpi/dim3/adjacency_graph.cpp
            mesh/mpi/dim3/runner.cpp
  LIBRARIES specfem::mesh
            specfem_environment
            specfem::io
            specfem::utilities
            MPI::MPI_CXX
            -lpthread -lm
)

specfem_add_test(mesh_mpi_dim2_tests
  MPI_RANKS 4
  SOURCES   mesh/mpi/dim2/adjacency_graph.cpp
            mesh/mpi/dim2/runner.cpp
  LIBRARIES specfem::mesh
            specfem_environment
            specfem::io
            specfem::utilities
            MPI::MPI_CXX
            -lpthread -lm
)

specfem_add_test(io_mesh_mpi_tests
  MPI_RANKS 4
  SOURCES   io/mesh/dim3/read_mesh.cpp
            io/mesh/dim3/runner.cpp
  LIBRARIES specfem::io
            specfem::mesh
            specfem_environment
            specfem::utilities
            MPI::MPI_CXX
            -lpthread -lm
)

specfem_add_test(io_mesh_mpi_dim2_tests
  MPI_RANKS 4
  SOURCES   io/mesh/dim2/read_mesh.cpp
            io/mesh/dim2/runner.cpp
  LIBRARIES specfem::io
            specfem::mesh
            specfem_environment
            specfem::utilities
            MPI::MPI_CXX
            -lpthread -lm
)

# specfem::MPI class tests

specfem_add_test(mpi_standard_tests
  MPI_RANKS 4
  SOURCES   mpi/mpi_standard_tests.cpp
  LIBRARIES ${MPI_CLASS_TEST_LIBS}
)

specfem_add_test(mpi_subset_2of4_tests
  MPI_RANKS 4
  SOURCES   mpi/mpi_subset_2of4_tests.cpp
  LIBRARIES ${MPI_CLASS_TEST_LIBS}
)

specfem_add_test(mpi_subset_1of4_tests
  MPI_RANKS 4
  SOURCES   mpi/mpi_subset_1of4_tests.cpp
  LIBRARIES ${MPI_CLASS_TEST_LIBS}
)

specfem_add_test(mpi_subset_all_tests
  MPI_RANKS 4
  SOURCES   mpi/mpi_subset_all_tests.cpp
  LIBRARIES ${MPI_CLASS_TEST_LIBS}
)

specfem_add_test(algorithms_locate_point_mpi_dim2_tests
  MPI_RANKS 4
  SOURCES   algorithms/mpi/dim2/locate_point_test.cpp
  LIBRARIES specfem::algorithms
            specfem::assembly
            specfem::io
            specfem::mesh
            specfem_environment
            specfem::utilities
            MPI::MPI_CXX
            Kokkos::kokkos
            -lpthread -lm
)

specfem_add_test(algorithms_locate_point_mpi_dim3_tests
  MPI_RANKS 4
  SOURCES   algorithms/mpi/dim3/locate_point_test.cpp
  LIBRARIES specfem::algorithms
            specfem::assembly
            specfem::io
            specfem::mesh
            specfem_environment
            specfem::utilities
            MPI::MPI_CXX
            Kokkos::kokkos
            -lpthread -lm
)

specfem_add_test(assembly_mpi_dim3_tests
  NO_UNITY
  MPI_RANKS 4
  SOURCES   assembly/mpi/dim3/fixture.cpp
            assembly/mpi/dim3/communication_group/communication_group.cpp
            assembly/mpi/dim3/communication_pattern/communication_pattern.cpp
            assembly/mpi/dim3/mass_matrix/mass_matrix.cpp
            assembly/mpi/dim3/reordering/reordering.cpp
            assembly/mpi/dim3/runner.cpp
  LIBRARIES ${ASSEMBLY_MPI_TEST_LIBS}
)

# 8-rank assembly MPI tests. Uses the HomogeneousElasticMPI2x2x2 fixture (a
# METIS-decomposed cube) to exercise the general MPI interface set -- top/bottom
# faces, horizontal edges, and single-node corner connections -- that the
# structured 4-rank fixtures cannot. Guards the connection-ordering fix in
# specfem::assembly::mpi<dim3>.
specfem_add_test(assembly_mpi_dim3_8proc_tests
  NO_UNITY
  MPI_RANKS 8
  SOURCES   assembly/mpi/dim3/fixture.cpp
            assembly/mpi/dim3/communication_pattern/communication_pattern.cpp
            assembly/mpi/dim3/reordering/reordering.cpp
            assembly/mpi/dim3/runner_8proc.cpp
  LIBRARIES ${ASSEMBLY_MPI_TEST_LIBS}
            $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
)
