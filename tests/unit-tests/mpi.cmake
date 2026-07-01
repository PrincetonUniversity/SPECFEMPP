# MPI-specific test definitions and setup
# This file contains MPI-only test executables and their discovery registration.

# Include GoogleTest module for test discovery
include(GoogleTest)

# MPI-only test executables

add_executable(
  mesh_mpi_dim3_tests
  mesh/mpi/dim3/adjacency_graph.cpp
  mesh/mpi/dim3/runner.cpp
)

target_link_libraries(
  mesh_mpi_dim3_tests
  specfem::mesh
  specfem_environment
  specfem::io
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  mesh_mpi_dim2_tests
  mesh/mpi/dim2/adjacency_graph.cpp
  mesh/mpi/dim2/runner.cpp
)

target_link_libraries(
  mesh_mpi_dim2_tests
  specfem::mesh
  specfem_environment
  specfem::io
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  io_mesh_mpi_tests
  io/mesh/dim3/read_mesh.cpp
  io/mesh/dim3/runner.cpp
)

target_link_libraries(
  io_mesh_mpi_tests
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

add_executable(
  io_mesh_mpi_dim2_tests
  io/mesh/dim2/read_mesh.cpp
  io/mesh/dim2/runner.cpp
)

target_link_libraries(
  io_mesh_mpi_dim2_tests
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  -lpthread -lm
)

# specfem::MPI class tests

add_executable(
  mpi_standard_tests
  mpi/mpi_standard_tests.cpp
)

target_link_libraries(
  mpi_standard_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_2of4_tests
  mpi/mpi_subset_2of4_tests.cpp
)

target_link_libraries(
  mpi_subset_2of4_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_1of4_tests
  mpi/mpi_subset_1of4_tests.cpp
)

target_link_libraries(
  mpi_subset_1of4_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  mpi_subset_all_tests
  mpi/mpi_subset_all_tests.cpp
)

target_link_libraries(
  mpi_subset_all_tests
  specfem::program
  specfem::mpi
  specfem_environment
  MPI::MPI_CXX
  gtest_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  algorithms_locate_point_mpi_dim2_tests
  algorithms/mpi/dim2/locate_point_test.cpp
)

target_link_libraries(
  algorithms_locate_point_mpi_dim2_tests
  specfem::algorithms
  specfem::assembly
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  algorithms_locate_point_mpi_dim3_tests
  algorithms/mpi/dim3/locate_point_test.cpp
)

target_link_libraries(
  algorithms_locate_point_mpi_dim3_tests
  specfem::algorithms
  specfem::assembly
  specfem::io
  specfem::mesh
  specfem_environment
  specfem::utilities
  MPI::MPI_CXX
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  assembly_mpi_dim3_tests
  assembly/mpi/dim3/fixture.cpp
  assembly/mpi/dim3/communication_group/communication_group.cpp
  assembly/mpi/dim3/communication_pattern/communication_pattern.cpp
  assembly/mpi/dim3/mass_matrix/mass_matrix.cpp
  assembly/mpi/dim3/reordering/reordering.cpp
  assembly/mpi/dim3/runner.cpp
)

target_compile_definitions(assembly_mpi_dim3_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})
set_target_properties(assembly_mpi_dim3_tests PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  assembly_mpi_dim3_tests
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

# Register MPI tests using helper function
foreach(test_target IN LISTS MPI_TEST_TARGETS_4PROCS)
  add_mpi_test(${test_target} 4)
endforeach()

# 8-rank assembly MPI tests. Uses the HomogeneousElasticMPI2x2x2 fixture (a
# METIS-decomposed cube) to exercise the general MPI interface set -- top/bottom
# faces, horizontal edges, and single-node corner connections -- that the
# structured 4-rank fixtures cannot. Guards the connection-ordering fix in
# specfem::assembly::mpi<dim3>.
add_executable(
  assembly_mpi_dim3_8proc_tests
  assembly/mpi/dim3/fixture.cpp
  assembly/mpi/dim3/communication_pattern/communication_pattern.cpp
  assembly/mpi/dim3/reordering/reordering.cpp
  assembly/mpi/dim3/runner_8proc.cpp
)

target_compile_definitions(assembly_mpi_dim3_8proc_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})
set_target_properties(assembly_mpi_dim3_8proc_tests PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  assembly_mpi_dim3_8proc_tests
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
  $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
  -lpthread -lm
)

# MPI test targets (4 processes)
set(MPI_TEST_TARGETS_4PROCS
  mesh_mpi_dim3_tests
  mesh_mpi_dim2_tests
  io_mesh_mpi_tests
  io_mesh_mpi_dim2_tests
  mpi_standard_tests
  mpi_subset_2of4_tests
  mpi_subset_1of4_tests
  mpi_subset_all_tests
  algorithms_locate_point_mpi_dim2_tests
  algorithms_locate_point_mpi_dim3_tests
  assembly_mpi_dim3_tests
)

# MPI test targets (8 processes)
set(MPI_TEST_TARGETS_8PROCS
  assembly_mpi_dim3_8proc_tests
)

set(MPI_TEST_TARGETS ${MPI_TEST_TARGETS_4PROCS} ${MPI_TEST_TARGETS_8PROCS})

# Setup test script writer (needed for external TEST_OUTPUT_DIR path-fix)
specfem_write_copy_test_cmake_script()

foreach(test_target IN LISTS MPI_TEST_TARGETS_8PROCS)
  add_mpi_test(${test_target} 8)
endforeach()
