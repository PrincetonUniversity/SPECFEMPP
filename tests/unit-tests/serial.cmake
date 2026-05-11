# Serial (non-MPI) test definitions and setup
# This file contains all non-MPI test executables and their test discovery registration.

# Test framework setup for serial tests
include(GoogleTest)

# Non-MPI test executables

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
  specfem::io
  ${BOOST_LIBS}
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
  serial_mpi_tests
  mpi/serial_mpi_tests.cpp
)

target_link_libraries(
  serial_mpi_tests
  specfem::program
  specfem::mpi
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
  logspace_tests
  utilities/logspace_tests.cpp
  utilities/logarithmic_center_tests.cpp
  utilities/band_tests.cpp
)

target_link_libraries(
  logspace_tests
  specfem::utilities
  specfem_environment
  gtest_main
  Kokkos::kokkos
)

add_executable(
  units_tests
  units/quantity_tests.cpp
  units/unit_cast_tests.cpp
)

target_link_libraries(
  units_tests
  specfem::utilities
  gtest_main
  Kokkos::kokkos
)

add_executable(
  attenuation_tests
  attenuation/runner.cpp
  attenuation/compute_band_tests.cpp
  attenuation/compute_tau_sigma_tests.cpp
  attenuation/maxwell_tests.cpp
  attenuation/compute_tau_eps_tests.cpp
  attenuation/compute_factors_tests.cpp
  attenuation/compute_integration_factors_tests.cpp
)

target_link_libraries(
  attenuation_tests
  gtest_main
  specfem::attenuation
  Kokkos::kokkos
)

add_executable(
  optimization_tests
  optimization/runner.cpp
  optimization/neldermead_tests.cpp
  optimization/steepestdescent_tests.cpp
)

target_link_libraries(
  optimization_tests
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
  specfem::enums
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
  specfem::io
  -lpthread -lm
)

add_executable(
  point_tests
  point/index_tests.cpp
  point/coordinates_tests.cpp
  point/boundary_tests.cpp
  point/jacobian_matrix_tests.cpp
  point/attenuation_tests.cpp
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
  point/properties/dim2/elastic_isotropic_cosserat.cpp
  point/properties/dim2/electromagnetic_isotropic.cpp
  point/properties/dim2/poroelastic_isotropic.cpp
  # Dim 3
  point/properties/dim3/elastic_isotropic.cpp
  point/properties/dim3/elastic_isotropic_cosserat.cpp
)

target_link_libraries(
  point_tests
  point
  specfem_environment
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
  ${BOOST_LIBS}
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
  specfem::io
  specfem::utilities
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
  specfem::io
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
  specfem::assembly
  specfem::quadrature
  specfem_environment
  yaml-cpp
  specfem::utilities
  ${BOOST_LIBS}
  -lpthread -lm
  gtest_main
)

add_executable(
  nonconforming_assembly_tests
  assembly/runner.cpp
  assembly/nonconforming_interfaces/container_init_test.cpp
)

target_link_libraries(
  nonconforming_assembly_tests
  specfem::io
  specfem::mesh
  specfem::assembly
  specfem::quadrature
  specfem_environment
  yaml-cpp
  specfem::utilities
  ${BOOST_LIBS}
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
  assembly/info/compute_tests.cpp
  assembly/info/scatter_minmax_tests.cpp
  assembly/element_intersections/edge_types_tests.cpp
  assembly/element_intersections/face_types_tests.cpp
  assembly/dim3/mesh/shape_functions.cpp
  assembly/dim3/mesh/points.cpp
  assembly/dim3/mesh/control_nodes.cpp
  assembly/dim3/jacobian_matrix/jacobian_matrix.cpp
  assembly/dim3/properties/properties.cpp
)

target_compile_definitions(assembly_tests PRIVATE TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR})
set_target_properties(assembly_tests PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  assembly_tests
  specfem::mesh
  specfem::assembly
  specfem::quadrature
  specfem_environment
  specfem::io
  yaml-cpp
  specfem::utilities
  specfem::enums
  specfem::element
  shape_functions
  specfem::quadrature
  ${BOOST_LIBS}
  -lpthread -lm
  gtest_main
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
  specfem::assembly
  specfem::quadrature
  specfem_environment
  yaml-cpp
  specfem::io
  ${BOOST_LIBS}
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
  specfem::io
  gtest_main
  yaml-cpp
  specfem::enums
  ${BOOST_LIBS}
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
  specfem::io
  ${BOOST_LIBS}
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
  specfem::assembly
  specfem::utilities
  -lpthread -lm
)

set_target_properties(gradient_tests PROPERTIES UNITY_BUILD OFF)

add_executable(
  transfer_tests
  algorithms/dim2/transfer.cpp
)

target_link_libraries(
  transfer_tests
  specfem::algorithms
  specfem::enums
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::utilities
  -lpthread -lm
)

add_executable(
  coupling_integral_tests
  algorithms/dim2/coupling_integral/dshape.cpp
  algorithms/dim2/coupling_integral/timesshape.cpp
)

target_link_libraries(
  coupling_integral_tests
  specfem::algorithms
  specfem::element_connections
  specfem::mesh
  assembly
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::utilities
  specfem::quadrature
  yaml-cpp
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
  ${BOOST_LIBS}
  -lpthread -lm
)

add_executable(
  chunked_edge_tests
  policies/chunked_edge.cpp
)

target_link_libraries(
  chunked_edge_tests
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::assembly
  specfem::enums
  ${BOOST_LIBS}
  -lpthread -lm
)

add_executable(
  chunked_face_tests
  policies/chunked_face.cpp
)

target_link_libraries(
  chunked_face_tests
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::enums
  point
  -lpthread -lm
)

add_executable(
  chunked_face_intersection_tests
  policies/chunked_face_intersection.cpp
)

target_link_libraries(
  chunked_face_intersection_tests
  gtest_main
  Kokkos::kokkos
  specfem_environment
  specfem::enums
  point
  -lpthread -lm
)

add_executable(
  mass_matrix_tests
  medium/mass_matrix/main.cpp
  medium/mass_matrix/dim2/elastic_isotropic.cpp
  medium/mass_matrix/dim2/elastic_anisotropic.cpp
  medium/mass_matrix/dim2/acoustic.cpp
  medium/mass_matrix/dim2/poroelastic.cpp
  medium/mass_matrix/dim3/elastic_isotropic.cpp
  medium/mass_matrix/dim3/acoustic.cpp
)

target_link_libraries(
  mass_matrix_tests
  point
  gtest_main
)

add_executable(
  stress_tests
  medium/stress/main.cpp
  medium/stress/dim2/acoustic.cpp
  medium/stress/dim2/elastic_isotropic.cpp
  medium/stress/dim2/elastic_anisotropic.cpp
  medium/stress/dim2/elastic_isotropic_cosserat.cpp
  medium/stress/dim2/poroelastic_isotropic.cpp
  medium/stress/dim3/elastic_isotropic.cpp
  medium/stress/dim3/acoustic.cpp
)

target_link_libraries(
  stress_tests
  point
  gtest_main
)

add_executable(
  strain_tests
  medium/strain/main.cpp
  medium/strain/dim2/elastic_isotropic.cpp
  medium/strain/dim3/elastic_isotropic.cpp
)

target_link_libraries(
  strain_tests
  point
  gtest_main
)

add_executable(
  compute_coupling_tests
  compute_coupling/acoustic_elastic.cpp
  compute_coupling/elastic_acoustic.cpp
  compute_coupling/nonconforming/acoustic_elastic.cpp
  compute_coupling/nonconforming/elastic_acoustic.cpp
  compute_coupling/runner.cpp
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
  medium/source/dim2/acoustic.cpp
  medium/source/dim2/elastic_isotropic.cpp
  medium/source/dim2/elastic_anisotropic.cpp
  medium/source/dim2/elastic_isotropic_cosserat.cpp
  medium/source/dim2/poroelastic.cpp
  medium/source/dim3/elastic_isotropic.cpp
  medium/source/dim3/acoustic.cpp
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

add_executable(
  displacement_newmark_2d_tests
  displacement_tests/Newmark/dim2/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_2d_tests
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

add_custom_command(TARGET displacement_newmark_2d_tests POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/dim2/tests.yaml
          ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2/tests.yaml
     COMMENT "Moving displacement_newmark_2d_tests data files to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim2"
)

add_executable(
  displacement_newmark_3d_tests
  displacement_tests/Newmark/dim3/newmark_tests.cpp
)

target_link_libraries(
  displacement_newmark_3d_tests
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

add_custom_command(TARGET displacement_newmark_3d_tests POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/displacement_tests/Newmark/dim3/tests.yaml
          ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3/tests.yaml
     COMMENT "Moving displacement_newmark_3d_tests data files to ${TEST_OUTPUT_DIR}/displacement_tests/Newmark/dim3"
)

# Register serial tests for discovery
set(SERIAL_TEST_TARGETS
  serial_mpi_tests
  assembly_receivers_tests
  assembly_tests
  attenuation_tests
  chunked_edge_tests
  chunked_face_tests
  chunked_face_intersection_tests
  compute_coupling_tests
  displacement_newmark_2d_tests
  displacement_newmark_3d_tests
  fortranio_test
  enumerations_tests
  gll_tests
  interpolate_function
  io_framework_tests
  io_tests
  is_close_tests
  logspace_tests
  jacobian_tests
  lagrange_tests
  locate_point_fixture_2d
  locate_point_fixture_3d
  gradient_tests
  transfer_tests
  mass_matrix_tests
  mesh_dim2_tests
  mesh_dim3_tests
  nonconforming_tests
  point_tests
  policies_tests
  optimization_tests
  receivers_tests
  simd_tests
  source_class_tests
  source_tests
  source_time_function_tests
  strain_tests
  stress_tests
  test_mesh_utilities_mapping_2d
  test_mesh_utilities_mapping_3d
  units_tests
)

if (NOT SPECFEM_ENABLE_MPI)
    list(APPEND SERIAL_TEST_TARGETS abort_tests)
endif()

# Link test data directories for serial tests
set(SERIAL_LINK_DIRS
  algorithms
  assembly
  assembly_mesh
  data
  displacement_tests
  fortran_io
  io
  mesh
  policies
)

# Setup test script writer (called once for all targets)
specfem_write_copy_test_cmake_script()

# Register each test target for discovery with optional path-fix
foreach(test_target IN LISTS SERIAL_TEST_TARGETS)
    specfem_register_test_target(${test_target})
endforeach()
