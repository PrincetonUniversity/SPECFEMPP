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
  specfem_environment
  SPECFEM_Environment.cpp
)

target_link_libraries(
  specfem_environment
  gtest_main
  specfem_program
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
  specfem::utilities
  gtest_main
)

add_executable(
  test_mesh_utilities_mapping_3d
  mesh_utilities/test_mapping_3d.cpp
)

target_link_libraries(
  test_mesh_utilities_mapping_3d
  mesh_utilities_mapping
  specfem::utilities
  gtest_main
)

add_executable(
  io_framework_tests
  io/io_framework_tests.cpp
)

target_link_libraries(io_framework_tests PRIVATE
  io
  boost
  gtest_main
  Kokkos::kokkos
  specfem_environment
  $<$<BOOL:${SPECFEM_ENABLE_NPZ}>:zlib>
  $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
  $<$<BOOL:${SPECFEM_ENABLE_ADIOS2}>:adios2>
)

target_compile_definitions(
    io_framework_tests
    PUBLIC
    $<$<NOT:$<BOOL:${SPECFEM_ENABLE_NPZ}>>:-DNO_NPZ>
    $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:-DNO_HDF5>
    $<$<NOT:$<BOOL:${SPECFEM_ENABLE_ADIOS2}>>:-DNO_ADIOS2>
)

add_executable(
  abort_tests
  program/abort_tests.cpp
)

target_link_libraries(
  abort_tests
  specfem::program
  gtest_main
  Kokkos::kokkos
)

add_executable(
  is_close_tests
  utilities/is_close_tests.cpp
)

target_link_libraries(
  is_close_tests
  specfem::utilities
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
  specfem::quadrature
  specfem_environment
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
  specfem::quadrature
  specfem_environment
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
  specfem_environment
  gtest_main
  -lpthread -lm
)

add_executable(
  enumerations_tests
  enumerations/dim2/mesh_entity.cpp
  enumerations/dim2/connections.cpp
  enumerations/dim3/mesh_entity.cpp
  enumerations/dim3/connections.cpp
  enumerations/runner.cpp
)

target_link_libraries(
  enumerations_tests
  enumerations
  specfem::quadrature
  shape_functions
  gtest_main
  specfem_environment
  specfem::utilities
  -lpthread -lm
)

set_target_properties(enumerations_tests PROPERTIES UNITY_BUILD OFF)

add_executable(
  simd_tests
  datatype/simd_tests.cpp
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
  receivers/dim3/receiver_tests.cpp
)

target_link_libraries(
  receivers_tests
  gtest_main
  specfem::receivers
  specfem_environment
  yaml-cpp
  boost
  -lpthread -lm
)

add_executable(
  mesh_dim2_tests
  mesh/dim2/test_fixture/test_fixture.cpp
  mesh/dim2/materials/materials.cpp
  mesh/dim2/materials/properties.cpp
  mesh/dim2/runner.cpp
  mesh/dim2/adjacency_graph/adjacency_graph.cpp
  mesh/dim2/adjacency_graph/adjacency_graph_regular_mesh.cpp
  mesh/dim2/adjacency_graph/adjacency_graph_irregular_mesh.cpp
)

target_link_libraries(
  mesh_dim2_tests
  gtest_main
  specfem::mesh
  specfem_environment
  yaml-cpp
  io
  specfem::utilities

  # material_class
  -lpthread -lm
)

add_executable(
  mesh_dim3_tests
  mesh/dim3/mesh.cpp
  mesh/dim3/control_nodes.cpp
  mesh/dim3/materials.cpp
  mesh/dim3/boundaries.cpp
  mesh/dim3/adjacency_graph.cpp
  mesh/dim3/tags.cpp
  mesh/dim3/test.cpp
)

target_link_libraries(
  mesh_dim3_tests
  gtest_main
  specfem::mesh
  specfem_environment
  io
  specfem::utilities

  -lpthread -lm
)

add_executable(
  nonconforming_tests
  nonconforming/reparameterizations/compute_intersection_test.cpp
  nonconforming/reparameterizations/set_transfer_functions_test.cpp
  nonconforming/runner.cpp
)

target_link_libraries(
  nonconforming_tests
  specfem::mesh
  assembly
  specfem::quadrature
  specfem_environment
  yaml-cpp
  specfem::utilities
  boost
  -lpthread -lm
  gtest_main
)

# add_executable(
#   compute_jacobian_matrix_tests
#   assembly_mesh/jacobian_matrix/compute_jacobian_matrix_tests.cpp
# )

# target_link_libraries(
#   compute_jacobian_matrix_tests
#   mesh
#   assembly
#   quadrature
#   yaml-cpp
#   compare_arrays
#   boost
#   specfem_environment
#   # material_class
#   -lpthread -lm
# )

# add_executable(
#   compute_elastic_tests
#   assembly_mesh/elastic/compute_properties_tests.cpp
# )

# target_link_libraries(
#   compute_elastic_tests
#   mesh
#   assembly
#   quadrature
#   specfem_environment
#
#   yaml-cpp
#   compare_arrays
#   boost
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
#   specfem_environment
#   yaml-cpp
#   compare_arrays
#   # material_class
#   -lpthread -lm
# )

# add_executable(
#   compute_tests
#   assembly_mesh/index/compute_tests.cpp
# )

# target_link_libraries(
#   compute_tests
#   mesh
#   assembly
#   quadrature
#   specfem_environment
#   yaml-cpp
#   compare_arrays
#   io
#   boost
#   # material_class
#   -lpthread -lm
# )

# target_compile_definitions(
#         compute_tests
#         PUBLIC
#         $<$<BOOL:${__APPLE__}>:-DSPECFEMPP_APPLE_BUILD>
#         $<$<BOOL:${__GNU__}>:-DSPECFEMPP_GNU_BUILD>
#         $<$<BOOL:${__MSVC__}>:-DSPECFEMPP_MSVC_BUILD>
#         $<$<BOOL:${__INTEL_LLVM__}>:-DSPECFEMPP_INTEL_LLVM_BUILD>
# )

add_executable(
  nonconforming_assembly_tests
  assembly/runner.cpp
  assembly/nonconforming_interfaces/container_init_test.cpp
)
target_link_libraries(
  nonconforming_assembly_tests
  reader
  writer
  specfem::mesh
  assembly
  specfem::quadrature
  specfem_environment
  io
  yaml-cpp
  specfem::utilities
  boost

  -lpthread -lm
  gtest_main
)


add_executable(
  assembly_tests
  assembly/test_fixture/test_fixture.cpp
  assembly/runner.cpp
  assembly/kernels/kernels.cpp
  assembly/properties/properties.cpp
  assembly/compute_wavefield/compute_wavefield.cpp
  assembly/sources/sources.cpp
  assembly/check_jacobian/dim2/check_jacobian.cpp
  assembly/locate/locate_point.cpp
  assembly/locate/locate_point_on_edge.cpp
  assembly/mesh/utilities.cpp
  assembly/sources/locate_sources.cpp
  assembly/compute_source_array/dim2/compute_source_array_from_vector.cpp
  assembly/compute_source_array/dim2/compute_source_array_from_tensor.cpp
  assembly/compute_source_array/dim3/compute_source_array_from_vector.cpp
  assembly/compute_source_array/dim3/compute_source_array_from_tensor.cpp

  # New dim3 tests
  assembly/dim3/mesh/shape_functions.cpp
  assembly/dim3/mesh/points.cpp
  assembly/dim3/mesh/control_nodes.cpp
  assembly/dim3/jacobian_matrix/jacobian_matrix.cpp
  assembly/dim3/properties/properties.cpp
)

#Set unity build off
target_compile_definitions(assembly_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})
set_target_properties(assembly_tests PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  assembly_tests
  reader
  writer
  specfem::mesh
  assembly
  specfem::quadrature
  specfem_environment
  io

  yaml-cpp
  specfem::utilities
  enumerations
  shape_functions
  specfem::quadrature
  boost

  -lpthread -lm
  gtest_main
  ${BOOST_LIBS}
)

add_executable(
  assembly_receivers_tests
  assembly/receivers/receivers_tests.cpp
  assembly/receivers/dim2/receivers_tests.cpp
  assembly/receivers/impl/receiver_iterator_tests.cpp
  assembly/receivers/impl/dim2/seismogram_iterator_tests.cpp
)

target_link_libraries(
  assembly_receivers_tests
  specfem::mesh
  assembly
  specfem::quadrature
  specfem_environment
  yaml-cpp
  io
  boost

  gtest
  gtest_main
  -lpthread -lm
)

add_executable(
  io_tests
  io/sources/test_read_sources_file.cpp
  io/sources/test_read_sources_yaml.cpp
  io/sources/test_source_solutions.cpp
  io/receivers/test_receiver_solutions.cpp
  io/receivers/test_read_stations_file.cpp
  io/receivers/test_read_yaml.cpp
)

target_link_libraries(
  io_tests
  io
  gtest_main

  yaml-cpp
  enumerations
  boost
)

add_executable(
  interpolate_function
  algorithms/interpolate_function/dim2/interpolate_function.cpp
  algorithms/interpolate_function/dim3/interpolate_function.cpp
  algorithms/interpolate_function/runner.cpp
)

target_link_libraries(
  interpolate_function
  specfem::quadrature
  specfem_environment
  specfem::algorithms
  io
  boost
  point
)

add_executable(
  locate_point_fixture_2d
  algorithms/dim2/locate_point_fixture.cpp
  algorithms/dim2/locate_point_test.cpp
  algorithms/dim2/locate_point_on_edge_test.cpp
)

target_link_libraries(
  locate_point_fixture_2d
  mesh_utilities_mapping
  specfem::algorithms
  jacobian
  point
  specfem::utilities
  specfem_environment
  gtest_main
)


add_executable(
  locate_point_fixture_3d
  algorithms/dim3/locate_point_fixture.cpp
)

target_link_libraries(
  locate_point_fixture_3d
  mesh_utilities_mapping
  specfem::algorithms
  jacobian
  point
  specfem::utilities
  specfem_environment
  gtest_main
)

add_executable(
  gradient_tests
  algorithms/dim2/gradient.cpp
  algorithms/dim3/gradient.cpp
)

target_link_libraries(
  gradient_tests
  specfem::quadrature
  specfem::algorithms
  gtest_main
  Kokkos::kokkos
  specfem_environment
  assembly
  specfem::utilities
  -lpthread -lm
)

# Turn unity build off for gradient_tests
set_target_properties(gradient_tests PROPERTIES UNITY_BUILD OFF)

add_executable(
  transfer_tests
  algorithms/dim2/transfer.cpp
)

target_link_libraries(
  transfer_tests
  specfem::algorithms
  enumerations
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::utilities
  -lpthread -lm
)

add_executable(
  policies_tests
  policies/policies.cpp
)

target_link_libraries(
  policies_tests
  specfem::mesh
  source_class
  specfem::receivers
  specfem_environment
  yaml-cpp
  boost
  -lpthread -lm
)

# add_executable(
#   chunked_edge_tests
#   policies/chunked_edge.cpp
# )

# target_link_libraries(
#   chunked_edge_tests
#   kokkos_environment
#   gtest_main
#   Kokkos::kokkos
#   specfem_environment
#   enumerations
#   ${BOOST_LIBS}
#   -lpthread -lm
# )

add_executable(
  mass_matrix_tests
  medium/mass_matrix/main.cpp
  # 2D mass matrix tests
  medium/mass_matrix/dim2/elastic_isotropic.cpp
  medium/mass_matrix/dim2/elastic_anisotropic.cpp
  medium/mass_matrix/dim2/acoustic.cpp
  medium/mass_matrix/dim2/poroelastic.cpp
  # 3D mass matrix tests
  medium/mass_matrix/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  mass_matrix_tests
  point
  gtest_main
)

add_executable(
  stress_tests
  medium/stress/main.cpp
  # 2D stress tests
  medium/stress/dim2/acoustic.cpp
  medium/stress/dim2/elastic_isotropic.cpp
  medium/stress/dim2/elastic_anisotropic.cpp
  medium/stress/dim2/elastic_isotropic_cosserat.cpp
  medium/stress/dim2/poroelastic_isotropic.cpp
  # 3D stress tests
  medium/stress/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  stress_tests
  point
  gtest_main
)

add_executable(
  compute_coupling_tests
  compute_coupling/acoustic_elastic.cpp
  compute_coupling/elastic_acoustic.cpp
  compute_coupling/nonconforming.cpp
)

target_link_libraries(
  compute_coupling_tests
  point
  gtest
  specfem_environment
  Kokkos::kokkos
)

add_executable(
  source_tests
  medium/source/main.cpp
  # 2D source tests
  medium/source/dim2/acoustic.cpp
  medium/source/dim2/elastic_isotropic.cpp
  medium/source/dim2/elastic_anisotropic.cpp
  medium/source/dim2/elastic_isotropic_cosserat.cpp
  medium/source/dim2/poroelastic.cpp
  # 3D source tests
  medium/source/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  source_tests
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
  source_time_functions
  specfem_environment
  -lpthread -lm
)

add_executable(
  source_time_function_tests
  source_time_function/source_time_function_tests.cpp
)

target_link_libraries(
  source_time_function_tests
  source_time_functions
  Kokkos::kokkos
  yaml-cpp
  gtest_main
)

# # add_executable(
# #   source_location_tests
# #   source/source_location_tests.cpp
# # )

# # target_link_libraries(
# #   source_location_tests
# #   material_class
# #   specfem_environment
# #
# #   yaml-cpp
# #   mesh
# #   quadrature
# #   assembly
# #   source_class
# #   -lpthread -lm
# # )

add_executable(
  displacement_newmark_2d_tests
  displacement_tests/Newmark/dim2/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_2d_tests
  specfem::quadrature
  specfem::mesh
  # material_class
  yaml-cpp
  specfem_environment
  assembly
  specfem::runtime_configuration
  compare_arrays
  timescheme
  point
  specfem::algorithms
  specfem::kokkos_kernels
  specfem::solver
  specfem::periodic_tasks
  boost
  -lpthread -lm
)


add_executable(
  displacement_newmark_3d_tests
  displacement_tests/Newmark/dim3/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_3d_tests
  specfem::quadrature
  specfem::mesh
  # material_class
  yaml-cpp
  specfem_environment
  assembly
  specfem::runtime_configuration
  compare_arrays
  timescheme
  point
  specfem::algorithms
  specfem::kokkos_kernels
  specfem::solver
  specfem::periodic_tasks
  boost
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
#   specfem_environment
#   assembly
#   specfem::runtime_configuration
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
#   specfem_environment
#   assembly
#   specfem::runtime_configuration
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
#   specfem_environment
#   assembly
#   compare_arrays
#   point
#   edge
#   boost
#   -lpthread -lm
# )




# Link to gtest only if MPI is disabled
if(NOT MPI_PARALLEL)
  include(GoogleTest)
  set(TEST_TARGETS
    abort_tests
    assembly_receivers_tests
    assembly_tests
    # compute_acoustic_tests
    # compute_coupled_interfaces_tests
    # compute_elastic_tests
    # compute_jacobian_matrix_tests
    # compute_tests
    displacement_newmark_2d_tests
    displacement_newmark_3d_tests
    fortranio_test
    enumerations_tests
    gll_tests
    interpolate_function
    io_framework_tests
    io_tests
    is_close_tests
    jacobian_tests
    lagrange_tests
    locate_point_fixture_2d
    locate_point_fixture_3d
    gradient_tests
    transfer_tests
    mass_matrix_tests
    compute_coupling_tests
    mesh_dim2_tests
    mesh_dim3_tests
    nonconforming_tests
    point_tests
    policies_tests
    # chunked_edge_tests
    receivers_tests
    simd_tests
    source_class_tests
    source_tests
    source_time_function_tests
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
    specfem::algorithms
    assembly
    assembly_mesh
    data
    displacement_tests
    domain
    fortran_io
    io
    specfem::mesh
    policies
  )

  foreach(dir_name IN LISTS LINK_DIRS)
      file(CREATE_LINK
          ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
          ${CMAKE_CURRENT_BINARY_DIR}/${dir_name}
          SYMBOLIC)
  endforeach()


  # gtest_discover_tests(seismogram_elastic_tests)
  # gtest_discover_tests(seismogram_acoustic_tests)
endif()


add_subdirectory(data/dim2/ncmarked_conforming_grid_curved/provenance)
add_subdirectory(data/dim2/ncmarked_conforming_grid_flat/provenance)
