cmake_minimum_required(VERSION 3.17.5)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)

# Include the GoogleTest framework
include("${CMAKE_SOURCE_DIR}/cmake/googletest.cmake")

# Explicitly set binary output directory for tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests)

include_directories(.)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Some of the writing test need to write somewhere and we don't want that
# to be in the source directory
set(TEST_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

enable_testing()

add_library(
  kokkos_environment
  Kokkos_Environment.cpp
)

target_link_libraries(
  kokkos_environment
  Kokkos::kokkos
  gtest_main
)

add_library(
  mpi_environment
  MPI_environment.cpp
)

target_link_libraries(
  mpi_environment
  gtest_main
  specfem_mpi
)


add_library(
  compare_arrays
  utilities/src/compare_array.cpp
)

target_link_libraries(
  compare_arrays
  Kokkos::kokkos
  io
)

add_library(
  mesh_utilities_mapping
  mesh_utilities/mapping.cpp
)

target_link_libraries(
  mesh_utilities_mapping
  Kokkos::kokkos
)

add_executable(
  test_mesh_utilities_mapping_2d
  mesh_utilities/test_mapping_2d.cpp
)

target_link_libraries(
  test_mesh_utilities_mapping_2d
  mesh_utilities_mapping
  utilities
  gtest_main
)

add_executable(
  test_mesh_utilities_mapping_3d
  mesh_utilities/test_mapping_3d.cpp
)

target_link_libraries(
  test_mesh_utilities_mapping_3d
  mesh_utilities_mapping
  utilities
  gtest_main
)

add_executable(
  io_framework_tests
  io/io_framework_tests.cpp
)

target_link_libraries(io_framework_tests PRIVATE
  io
  ${BOOST_LIBS}
  gtest_main
  Kokkos::kokkos
  kokkos_environment
)

if (NOT SPECFEM_ENABLE_HDF5)
    target_compile_definitions(
        io_framework_tests
        PUBLIC -DNO_HDF5
    )
endif()

if (NOT SPECFEM_ENABLE_ADIOS2)
    target_compile_definitions(
        io_framework_tests
        PUBLIC -DNO_ADIOS2
    )
endif()


add_executable(
  is_close_tests
  utilities/is_close_tests.cpp
)

target_link_libraries(
  is_close_tests
  utilities
  gtest_main
  Kokkos::kokkos
)

add_executable(
  gll_tests
  gll/gll_tests.cpp
)

target_link_libraries(
  gll_tests
  gtest_main
  quadrature
  kokkos_environment
  point
  -lpthread -lm
)

add_executable(
  lagrange_tests
  lagrange/Lagrange_tests.cpp
)

target_link_libraries(
  lagrange_tests
  gtest_main
  quadrature
  kokkos_environment
  -lpthread -lm
)

add_executable(
  jacobian_tests
  jacobian/jacobian_tests.cpp
  jacobian/dim2/shape_functions_tests.cpp
  jacobian/dim2/compute_locations_tests.cpp
  jacobian/dim2/compute_jacobian_tests.cpp
  jacobian/dim3/shape_functions_tests.cpp
  jacobian/dim3/compute_locations_tests.cpp
  jacobian/dim3/compute_jacobian_tests.cpp
)

target_link_libraries(
  jacobian_tests
  jacobian
  mpi_environment
  kokkos_environment
  gtest_main
  -lpthread -lm
)

add_executable(
  simd_tests
  datatypes/simd_tests.cpp
)
target_link_libraries(
  simd_tests
  gtest_main
  gmock_main
  Kokkos::kokkos
  -lpthread -lm
)

add_executable(
  fortranio_test
  fortran_io/fortranio_tests.cpp
)

target_link_libraries(
  fortranio_test
  gtest_main
  gmock_main
  io
  -lpthread -lm
)


add_executable(
  point_tests
  point/index_tests.cpp
  point/coordinates_tests.cpp
  point/boundary_tests.cpp
  point/jacobian_matrix_tests.cpp
  point/source_tests.cpp
  point/stress_integrand_tests.cpp
  point/stress_tests.cpp
  # Kernels
  # Dim 2
  point/kernels/dim2/acoustic_isotropic.cpp
  point/kernels/dim2/elastic_isotropic.cpp
  point/kernels/dim2/elastic_anisotropic.cpp
  # point/kernels/dim2/poroelastic_isotropic.cpp
  # Dim 3
  point/kernels/dim3/elastic_isotropic.cpp
  # Properties
  # Dim 2
  point/properties/dim2/elastic_isotropic.cpp
  point/properties/dim2/elastic_anisotropic.cpp
  point/properties/dim2/acoustic_isotropic.cpp
  point/properties/dim2/electromagnetic_isotropic.cpp
  point/properties/dim2/poroelastic_isotropic.cpp
  # Dim 3
  # point/properties/dim3/elastic_isotropic.cpp

)

target_link_libraries(
  point_tests
  point
  gtest_main
  gmock_main
)

add_executable(
  receivers_tests
  receivers/dim2/receiver_tests.cpp
)

target_link_libraries(
  receivers_tests
  gtest_main
  specfem::receivers
  mpi_environment
  kokkos_environment
  yaml-cpp
  Boost::filesystem
  -lpthread -lm
)

add_executable(
  mesh_tests
  mesh/test_fixture/test_fixture.cpp
  mesh/materials/materials.cpp
  mesh/materials/properties.cpp
  mesh/runner.cpp
  mesh/adjacency_graph/adjacency_graph.cpp
  mesh/adjacency_graph/adjacency_graph_regular_mesh.cpp
  mesh/adjacency_graph/adjacency_graph_irregular_mesh.cpp
)

target_link_libraries(
  mesh_tests
  gtest_main
  mesh
  mpi_environment
  kokkos_environment
  yaml-cpp
  io
  utilities
  # material_class
  -lpthread -lm
)

add_executable(
  compute_jacobian_matrix_tests
  assembly_mesh/jacobian_matrix/compute_jacobian_matrix_tests.cpp
)

target_link_libraries(
  compute_jacobian_matrix_tests
  mesh
  assembly
  quadrature
  mpi_environment
  kokkos_environment
  yaml-cpp
  compare_arrays
  Boost::filesystem
  # material_class
  -lpthread -lm
)

# add_executable(
#   compute_elastic_tests
#   assembly_mesh/elastic/compute_properties_tests.cpp
# )

# target_link_libraries(
#   compute_elastic_tests
#   mesh
#   assembly
#   quadrature
#   mpi_environment
#   kokkos_environment
#   yaml-cpp
#   compare_arrays
#   Boost::filesystem
#   # material_class
#   -lpthread -lm
# )

# add_executable(
#   compute_acoustic_tests
#   assembly_mesh/acoustic/compute_properties_tests.cpp
# )

# target_link_libraries(
#   compute_acoustic_tests
#   mesh
#   assembly
#   quadrature
#   mpi_environment
#   kokkos_environment
#   yaml-cpp
#   compare_arrays
#   # material_class
#   -lpthread -lm
# )

add_executable(
  compute_tests
  assembly_mesh/index/compute_tests.cpp
)

target_link_libraries(
  compute_tests
  mesh
  assembly
  quadrature
  mpi_environment
  kokkos_environment
  yaml-cpp
  compare_arrays
  io
  Boost::filesystem
  # material_class
  -lpthread -lm
)

add_executable(
  assembly_tests
  assembly/test_fixture/test_fixture.cpp
  assembly/runner.cpp
  assembly/kernels/kernels.cpp
  assembly/properties/properties.cpp
  assembly/compute_wavefield/compute_wavefield.cpp
  assembly/sources/sources.cpp
  assembly/check_jacobian/check_jacobian.cpp
  assembly/locate/locate_point.cpp
  assembly/mesh/utilities.cpp
  assembly/sources/locate_sources.cpp
  assembly/compute_source_array/dim2/compute_source_array_from_vector.cpp
  assembly/compute_source_array/dim2/compute_source_array_from_tensor.cpp
  assembly/compute_source_array/dim3/compute_source_array_from_vector.cpp
)

target_compile_definitions(assembly_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})

target_link_libraries(
  assembly_tests
  reader
  writer
  mesh
  assembly
  quadrature
  mpi_environment
  io
  kokkos_environment
  yaml-cpp
  utilities
  Boost::filesystem
  -lpthread -lm
  gtest_main
)

add_executable(
  io_tests
  io/sources/test_read_sources.cpp
)

target_link_libraries(
  io_tests
  io
  gtest_main
  kokkos_environment
  yaml-cpp
  enumerations
  Boost::filesystem
)

add_executable(
  interpolate_function
  algorithms/interpolate_function.cpp
)

target_link_libraries(
  interpolate_function
  mesh
  assembly
  quadrature
  mpi_environment
  kokkos_environment
  algorithms
  io
  Boost::filesystem
  point
)

add_executable(
  locate_point_fixture_2d
  algorithms/dim2/locate_point_fixture.cpp
)

target_link_libraries(
  locate_point_fixture_2d
  mesh_utilities_mapping
  algorithms
  jacobian
  point
  utilities
  mpi_environment
  kokkos_environment
  gtest_main
)


add_executable(
  locate_point_fixture_3d
  algorithms/dim3/locate_point_fixture.cpp
)

target_link_libraries(
  locate_point_fixture_3d
  mesh_utilities_mapping
  algorithms
  jacobian
  point
  utilities
  mpi_environment
  kokkos_environment
  gtest_main
)

add_executable(
  policies_tests
  policies/policies.cpp
)

target_link_libraries(
  policies_tests
  mesh
  source_class
  specfem::receivers
  kokkos_environment
  mpi_environment
  yaml-cpp
  Boost::filesystem
  -lpthread -lm
)

add_executable(
  mass_matrix_tests
  mass_matrix/main.cpp
  # 2D mass matrix tests
  mass_matrix/dim2/elastic_isotropic.cpp
  mass_matrix/dim2/elastic_anisotropic.cpp
  mass_matrix/dim2/acoustic.cpp
  mass_matrix/dim2/poroelastic.cpp
  # 3D mass matrix tests
  mass_matrix/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  mass_matrix_tests
  point
  gtest_main
)

add_executable(
  stress_tests
  stress/main.cpp
  # 2D stress tests
  stress/dim2/acoustic.cpp
  stress/dim2/elastic_isotropic.cpp
  stress/dim2/elastic_anisotropic.cpp
  stress/dim2/elastic_isotropic_cosserat.cpp
  stress/dim2/poroelastic_isotropic.cpp
  # 3D stress tests
  stress/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  stress_tests
  point
  gtest_main
)

add_executable(
  source_class_tests
  source/source.cpp
  source/base_source_tests.cpp
  source/typed_source_tests.cpp
  # source/dim2/vector_sources/force_source.cpp
  # source/dim2/vector_sources/cosserat_force_source.cpp
  # source/dim2/vector_sources/adjoint_source.cpp
  # source/dim2/vector_sources/external_source.cpp
  # source/dim2/tensor_sources/moment_tensor_source.cpp
  # source/dim3/vector_sources/force_source.cpp
  # source/dim3/tensor_sources/moment_tensor_source.cpp
)
target_link_libraries(
  source_class_tests
  source_class
  source_time_function
  mpi_environment
  kokkos_environment
  -lpthread -lm
)

# # add_executable(
# #   source_location_tests
# #   source/source_location_tests.cpp
# # )

# # target_link_libraries(
# #   source_location_tests
# #   material_class
# #   mpi_environment
# #   kokkos_environment
# #   yaml-cpp
# #   mesh
# #   quadrature
# #   assembly
# #   source_class
# #   -lpthread -lm
# # )

add_executable(
  displacement_newmark_tests
  displacement_tests/Newmark/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_tests
  quadrature
  mesh
  # material_class
  yaml-cpp
  kokkos_environment
  mpi_environment
  assembly
  parameter_reader
  compare_arrays
  timescheme
  point
  edge
  algorithms
  coupled_interface
  kokkos_kernels
  solver
  periodic_tasks
  Boost::filesystem
  -lpthread -lm
)

# add_executable(
#   seismogram_elastic_tests
#   seismogram/elastic/seismogram_tests.cpp
# )

# target_link_libraries(
#   seismogram_elastic_tests
#   quadrature
#   mesh
#   yaml-cpp
#   kokkos_environment
#   mpi_environment
#   assembly
#   parameter_reader
#   writer
#   periodic_tasks
#   domain
#   coupled_interface
#   solver
#   -lpthread -lm
# )

# add_executable(
#   seismogram_acoustic_tests
#   seismogram/acoustic/seismogram_tests.cpp
# )

# target_link_libraries(
#   seismogram_acoustic_tests
#   quadrature
#   mesh
#   yaml-cpp
#   kokkos_environment
#   mpi_environment
#   assembly
#   parameter_reader
#   writer
#   periodic_tasks
#   domain
#   coupled_interface
#   solver
#   -lpthread -lm
# )

# add_executable(
#   compute_coupled_interfaces_tests
#   assembly_mesh/coupled_interfaces/coupled_interfaces_tests.cpp
# )

# target_link_libraries(
#   compute_coupled_interfaces_tests
#   quadrature
#   mesh
#   yaml-cpp
#   kokkos_environment
#   mpi_environment
#   assembly
#   compare_arrays
#   point
#   edge
#   Boost::filesystem
#   -lpthread -lm
# )




# Link to gtest only if MPI is disabled
if (NOT MPI_PARALLEL)
  include(GoogleTest)
  set(TEST_TARGETS
    assembly_tests
    # compute_acoustic_tests
    # compute_coupled_interfaces_tests
    # compute_elastic_tests
    compute_jacobian_matrix_tests
    compute_tests
    displacement_newmark_tests
    fortranio_test
    gll_tests
    interpolate_function
    io_framework_tests
    io_tests
    is_close_tests
    jacobian_tests
    lagrange_tests
    locate_point_fixture_2d
    locate_point_fixture_3d
    mass_matrix_tests
    mesh_tests
    point_tests
    policies_tests
    receivers_tests
    simd_tests
    source_class_tests
    stress_tests
    test_mesh_utilities_mapping_2d
    test_mesh_utilities_mapping_3d
  )
  foreach(test_target IN LISTS TEST_TARGETS)
      gtest_discover_tests(${test_target}
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 300
      )
  endforeach()

  # For tests that require data files from the tests directory,
  # create symbolic links to the test data directory so that
  # the tests can access them from the build folder.

  set(LINK_DIRS
    algorithms
    assembly
    assembly_mesh
    data
    displacement_tests
    domain
    fortran_io
    io
    mesh
    policies
    mortar
  )

  foreach(dir_name IN LISTS LINK_DIRS)
      file(CREATE_LINK
          ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
          ${CMAKE_CURRENT_BINARY_DIR}/${dir_name}
          SYMBOLIC)
  endforeach()


  # gtest_discover_tests(seismogram_elastic_tests)
  # gtest_discover_tests(seismogram_acoustic_tests)
endif(NOT MPI_PARALLEL)
