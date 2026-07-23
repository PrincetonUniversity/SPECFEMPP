# Serial (non-MPI) unit test executables.
#
# Each specfem_add_test() both defines the target and registers it with CTest, so a
# test cannot be built and then silently left out of the suite. See
# tests/test_registration.cmake for the full set of options.

specfem_add_test(test_mesh_utilities_mapping_2d
  SOURCES   mesh_utilities/test_mapping_2d.cpp
  LIBRARIES mesh_utilities_mapping
            specfem::utilities
            specfem_environment
            gtest_main
)

specfem_add_test(test_mesh_utilities_mapping_3d
  SOURCES   mesh_utilities/test_mapping_3d.cpp
  LIBRARIES mesh_utilities_mapping
            specfem::utilities
            specfem_environment
            gtest_main
)

specfem_add_test(io_framework_tests
  SOURCES     io/io_framework_tests.cpp
  LIBRARIES   specfem::io
              ${BOOST_LIBS}
              gtest_main
              Kokkos::kokkos
              specfem_environment
              $<$<BOOL:${SPECFEM_ENABLE_NPZ}>:zlib>
              $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
              $<$<BOOL:${SPECFEM_ENABLE_ADIOS2}>:adios2>
  DEFINITIONS $<$<NOT:$<BOOL:${SPECFEM_ENABLE_NPZ}>>:NO_NPZ>
              $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:NO_HDF5>
              $<$<NOT:$<BOOL:${SPECFEM_ENABLE_ADIOS2}>>:NO_ADIOS2>
)

# Asserts on MPI-less abort behaviour, so it is meaningless in an MPI build.
if(NOT SPECFEM_ENABLE_MPI)
  specfem_add_test(abort_tests
    SOURCES   program/abort_tests.cpp
    LIBRARIES specfem::program
              gtest_main
              Kokkos::kokkos
  )
endif()

specfem_add_test(serial_mpi_tests
  SOURCES   mpi/serial_mpi_tests.cpp
  LIBRARIES specfem::program
            specfem::mpi
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(is_close_tests
  SOURCES   utilities/is_close_tests.cpp
  LIBRARIES specfem::utilities
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(logspace_tests
  SOURCES   utilities/logspace_tests.cpp
            utilities/logarithmic_center_tests.cpp
            utilities/band_tests.cpp
  LIBRARIES specfem::utilities
            specfem_environment
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(units_tests
  SOURCES   units/quantity_tests.cpp
            units/unit_cast_tests.cpp
            units/parse_tests.cpp
  LIBRARIES specfem::utilities
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(attenuation_tests
  SOURCES   attenuation/runner.cpp
            attenuation/compute_band_tests.cpp
            attenuation/compute_tau_sigma_tests.cpp
            attenuation/maxwell_tests.cpp
            attenuation/compute_tau_eps_tests.cpp
            attenuation/compute_factors_tests.cpp
            attenuation/compute_integration_factors_tests.cpp
  LIBRARIES gtest_main
            specfem::attenuation
            Kokkos::kokkos
)

specfem_add_test(optimization_tests
  SOURCES   optimization/runner.cpp
            optimization/neldermead_tests.cpp
            optimization/steepestdescent_tests.cpp
  LIBRARIES gtest_main
            Kokkos::kokkos
)

specfem_add_test(gll_tests
  SOURCES   gll/gll_tests.cpp
  LIBRARIES gtest_main
            specfem::quadrature
            specfem_environment
            point
            -lpthread -lm
)

specfem_add_test(lagrange_tests
  SOURCES   lagrange/Lagrange_tests.cpp
  LIBRARIES gtest_main
            specfem::quadrature
            specfem_environment
            -lpthread -lm
)

specfem_add_test(jacobian_tests
  SOURCES   jacobian/jacobian_tests.cpp
            jacobian/dim2/shape_functions_tests.cpp
            jacobian/dim2/compute_locations_tests.cpp
            jacobian/dim2/compute_jacobian_tests.cpp
            jacobian/dim3/shape_functions_tests.cpp
            jacobian/dim3/compute_locations_tests.cpp
            jacobian/dim3/compute_jacobian_tests.cpp
  LIBRARIES jacobian
            specfem_environment
            gtest_main
            -lpthread -lm
)

specfem_add_test(enumerations_tests
  NO_UNITY
  SOURCES   enumerations/dim2/mesh_entity.cpp
            enumerations/dim2/connections.cpp
            enumerations/dim3/mesh_entity.cpp
            enumerations/dim3/connections.cpp
            enumerations/runner.cpp
  LIBRARIES specfem::enums
            specfem::quadrature
            shape_functions
            gtest_main
            specfem_environment
            specfem::utilities
            -lpthread -lm
)

specfem_add_test(simd_tests
  SOURCES   datatype/simd_tests.cpp
  LIBRARIES gtest_main
            gmock_main
            Kokkos::kokkos
            -lpthread -lm
)

specfem_add_test(datatype_operators_tests
  SOURCES   datatype/tensor_point_view_operators_tests.cpp
  LIBRARIES gtest_main
            gmock_main
            Kokkos::kokkos
            -lpthread -lm
)

specfem_add_test(fortranio_test
  SOURCES   fortran_io/fortranio_tests.cpp
  LIBRARIES gtest_main
            gmock_main
            specfem::io
            -lpthread -lm
)

specfem_add_test(point_tests
  SOURCES   point/index_tests.cpp
            point/coordinates_tests.cpp
            point/boundary_tests.cpp
            point/jacobian_matrix_tests.cpp
            point/attenuation_tests.cpp
            point/field_derivatives_tests.cpp
            point/source_tests.cpp
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
  LIBRARIES point
            specfem_environment
            gtest_main
            gmock_main
)

specfem_add_test(receivers_tests
  SOURCES   receivers/dim2/receiver_tests.cpp
            receivers/dim3/receiver_tests.cpp
  LIBRARIES gtest_main
            specfem::receivers
            specfem_environment
            yaml-cpp
            ${BOOST_LIBS}
            -lpthread -lm
)

specfem_add_test(mesh_dim2_tests
  SOURCES   mesh/dim2/test_fixture/test_fixture.cpp
            mesh/dim2/materials/materials.cpp
            mesh/dim2/materials/properties.cpp
            mesh/dim2/runner.cpp
            mesh/dim2/adjacency_graph/adjacency_graph.cpp
            mesh/dim2/adjacency_graph/adjacency_graph_regular_mesh.cpp
            mesh/dim2/adjacency_graph/adjacency_graph_irregular_mesh.cpp
  LIBRARIES gtest_main
            specfem::mesh
            specfem_environment
            yaml-cpp
            specfem::io
            specfem::utilities
            -lpthread -lm
)

specfem_add_test(mesh_dim3_tests
  SOURCES   mesh/dim3/mesh.cpp
            mesh/dim3/control_nodes.cpp
            mesh/dim3/materials.cpp
            mesh/dim3/boundaries.cpp
            mesh/dim3/adjacency_graph.cpp
            mesh/dim3/tags.cpp
            mesh/dim3/test.cpp
  LIBRARIES gtest_main
            specfem::mesh
            specfem_environment
            specfem::io
            specfem::utilities
            -lpthread -lm
)

specfem_add_test(nonconforming_tests
  SOURCES   nonconforming/reparameterizations/compute_intersection_test.cpp
            nonconforming/reparameterizations/set_transfer_functions_test.cpp
            nonconforming/runner.cpp
  LIBRARIES specfem::mesh
            specfem::assembly
            specfem::quadrature
            specfem_environment
            yaml-cpp
            specfem::utilities
            ${BOOST_LIBS}
            -lpthread -lm
            gtest_main
)

# NO_CTEST: built but not run, which is the state this target has always been in --
# it was missing from the old hand-maintained SERIAL_TEST_TARGETS list. It cannot be
# registered as-is: it links assembly/runner.cpp for main(), which also instantiates
# the Assembly3DTest suite whose TEST_P bodies live in assembly/dim3/ and are linked
# only into assembly_tests. That leaves the suite instantiated with no tests, which
# GoogleTest reports as a failing test. Fixing that needs its own runner -- out of
# scope here; drop NO_CTEST once it has one.
specfem_add_test(nonconforming_assembly_tests
  NO_CTEST
  SOURCES   assembly/runner.cpp
            assembly/nonconforming_interfaces/container_init_test.cpp
  LIBRARIES specfem::io
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

specfem_add_test(assembly_tests
  NO_UNITY
  SOURCES   assembly/test_fixture/test_fixture.cpp
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
  LIBRARIES specfem::mesh
            specfem::assembly
            specfem::quadrature
            specfem_environment
            specfem::io
            yaml-cpp
            specfem::utilities
            specfem::enums
            specfem::element
            shape_functions
            ${BOOST_LIBS}
            -lpthread -lm
            gtest_main
)

specfem_add_test(assembly_receivers_tests
  SOURCES   assembly/receivers/receivers_tests.cpp
            assembly/receivers/dim2/receivers_tests.cpp
            assembly/receivers/impl/receiver_iterator_tests.cpp
            assembly/receivers/impl/dim2/seismogram_iterator_tests.cpp
  LIBRARIES specfem::mesh
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

specfem_add_test(element_types_tests
  SOURCES   assembly/element_types/runner.cpp
            assembly/element_types/dim2/element_types_tests.cpp
            assembly/element_types/dim3/element_types_tests.cpp
  LIBRARIES specfem::assembly
            specfem::element
            specfem_environment
            Kokkos::kokkos
            gtest_main
            ${BOOST_LIBS}
            -lpthread -lm
)

specfem_add_test(io_tests
  SOURCES   io/sources/test_read_sources_file.cpp
            io/sources/test_read_sources_yaml.cpp
            io/sources/test_read_sources_datetime.cpp
            io/sources/test_source_solutions.cpp
            io/receivers/test_receiver_solutions.cpp
            io/receivers/test_read_stations_file.cpp
            io/receivers/test_read_yaml.cpp
  LIBRARIES specfem::io
            gtest_main
            yaml-cpp
            specfem::enums
            ${BOOST_LIBS}
)

specfem_add_test(timing_tests
  SOURCES   io/sources/timing.cpp
  LIBRARIES specfem::io
            specfem_environment
            specfem::enums
            specfem::datetime
)

specfem_add_test(seismogram_writer_tests
  SOURCES   io/seismogram/seismogram_writer_tests.cpp
  LIBRARIES specfem::io
            specfem::enums
            specfem::element
            specfem::program
            gtest_main
)

specfem_add_test(inside_outside_tests
  SOURCES   algorithms/inside_outside_tests.cpp
  LIBRARIES specfem::algorithms
            point
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(interpolate_function
  SOURCES   algorithms/interpolate_function/dim2/interpolate_function.cpp
            algorithms/interpolate_function/dim3/interpolate_function.cpp
            algorithms/interpolate_function/runner.cpp
  LIBRARIES specfem::quadrature
            specfem_environment
            specfem::algorithms
            specfem::io
            ${BOOST_LIBS}
            point
)

specfem_add_test(locate_point_fixture_2d
  SOURCES   algorithms/dim2/locate_point_fixture.cpp
            algorithms/dim2/locate_point_test.cpp
            algorithms/dim2/locate_point_on_edge_test.cpp
  LIBRARIES mesh_utilities_mapping
            specfem::algorithms
            jacobian
            point
            specfem::utilities
            specfem_environment
            gtest_main
)

specfem_add_test(locate_point_fixture_3d
  SOURCES   algorithms/dim3/locate_point_fixture.cpp
            algorithms/dim3/locate_point_on_face_test.cpp
  LIBRARIES mesh_utilities_mapping
            specfem::algorithms
            jacobian
            point
            specfem::utilities
            specfem_environment
            gtest_main
)

specfem_add_test(gradient_tests
  NO_UNITY
  SOURCES   algorithms/dim2/gradient.cpp
            algorithms/dim3/gradient.cpp
  LIBRARIES specfem::quadrature
            specfem::algorithms
            gtest_main
            Kokkos::kokkos
            specfem_environment
            specfem::assembly
            specfem::utilities
            -lpthread -lm
)

specfem_add_test(transfer_tests
  SOURCES   algorithms/dim2/transfer.cpp
            algorithms/dim3/transfer_interpolate.cpp
  LIBRARIES specfem::algorithms
            specfem::enums
            gtest_main
            Kokkos::kokkos
            specfem_environment
            specfem::utilities
            -lpthread -lm
)

specfem_add_test(coupling_integral_tests
  SOURCES   algorithms/dim2/coupling_integral/dshape.cpp
            algorithms/dim2/coupling_integral/timesshape.cpp
  LIBRARIES specfem::algorithms
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

specfem_add_test(policies_tests
  SOURCES   policies/policies.cpp
  LIBRARIES specfem::mesh
            source_class
            specfem::receivers
            specfem_environment
            yaml-cpp
            ${BOOST_LIBS}
            -lpthread -lm
)

specfem_add_test(chunked_edge_tests
  SOURCES   policies/chunked_edge.cpp
  LIBRARIES gtest_main
            Kokkos::kokkos
            specfem_environment
            specfem::assembly
            specfem::enums
            ${BOOST_LIBS}
            -lpthread -lm
)

specfem_add_test(chunked_face_tests
  SOURCES   policies/chunked_face.cpp
  LIBRARIES gtest_main
            Kokkos::kokkos
            specfem_environment
            specfem::enums
            point
            -lpthread -lm
)

specfem_add_test(chunked_face_intersection_tests
  SOURCES   policies/chunked_face_intersection.cpp
  LIBRARIES gtest_main
            Kokkos::kokkos
            specfem_environment
            specfem::enums
            point
            -lpthread -lm
)

specfem_add_test(mass_matrix_tests
  SOURCES   medium/mass_matrix/main.cpp
            medium/mass_matrix/dim2/elastic_isotropic.cpp
            medium/mass_matrix/dim2/elastic_anisotropic.cpp
            medium/mass_matrix/dim2/acoustic.cpp
            medium/mass_matrix/dim2/poroelastic.cpp
            medium/mass_matrix/dim3/elastic_isotropic.cpp
            medium/mass_matrix/dim3/elastic_isotropic_cosserat.cpp
            medium/mass_matrix/dim3/acoustic.cpp
  LIBRARIES point
            gtest_main
)

specfem_add_test(stress_tests
  SOURCES   medium/stress/main.cpp
            medium/stress/dim2/acoustic.cpp
            medium/stress/dim2/elastic_isotropic.cpp
            medium/stress/dim2/elastic_anisotropic.cpp
            medium/stress/dim2/elastic_isotropic_cosserat.cpp
            medium/stress/dim2/poroelastic_isotropic.cpp
            medium/stress/dim3/elastic_isotropic.cpp
            medium/stress/dim3/elastic_isotropic_cosserat.cpp
            medium/stress/dim3/acoustic.cpp
  LIBRARIES point
            gtest_main
)

specfem_add_test(frechet_derivatives_tests
  SOURCES   medium/frechet_derivatives/main.cpp
            medium/frechet_derivatives/dim2/acoustic.cpp
            medium/frechet_derivatives/dim2/elastic_isotropic.cpp
            medium/frechet_derivatives/dim2/elastic_anisotropic.cpp
            medium/frechet_derivatives/dim3/acoustic.cpp
  LIBRARIES point
            gtest_main
)

specfem_add_test(strain_tests
  SOURCES   medium/strain/main.cpp
            medium/strain/dim2/elastic_isotropic.cpp
            medium/strain/dim3/elastic_isotropic.cpp
  LIBRARIES point
            gtest_main
)

specfem_add_test(medium_attenuation_tests
  SOURCES   medium/attenuation/main.cpp
            medium/attenuation/dim2/elastic_isotropic.cpp
            medium/attenuation/dim3/elastic_isotropic.cpp
  LIBRARIES point
            gtest_main
)

# Provides its own main() via runner.cpp, so it links gtest rather than gtest_main.
specfem_add_test(compute_coupling_tests
  SOURCES   compute_coupling/acoustic_elastic.cpp
            compute_coupling/elastic_acoustic.cpp
            compute_coupling/nonconforming/acoustic_elastic.cpp
            compute_coupling/nonconforming/elastic_acoustic.cpp
            compute_coupling/runner.cpp
  LIBRARIES point
            gtest
            specfem_environment
            Kokkos::kokkos
)

specfem_add_test(source_tests
  SOURCES   medium/source/main.cpp
            medium/source/dim2/acoustic.cpp
            medium/source/dim2/elastic_isotropic.cpp
            medium/source/dim2/elastic_anisotropic.cpp
            medium/source/dim2/elastic_isotropic_cosserat.cpp
            medium/source/dim2/poroelastic.cpp
            medium/source/dim3/elastic_isotropic.cpp
            medium/source/dim3/elastic_isotropic_cosserat.cpp
            medium/source/dim3/acoustic.cpp
  LIBRARIES point
            gtest_main
)

specfem_add_test(source_class_tests
  SOURCES   source/source.cpp
            source/base_source_tests.cpp
            source/typed_source_tests.cpp
  LIBRARIES source_class
            source_time_functions
            specfem_environment
            -lpthread -lm
)

specfem_add_test(source_time_function_tests
  SOURCES   source_time_function/source_time_function_tests.cpp
  LIBRARIES source_time_functions
            Kokkos::kokkos
            yaml-cpp
            specfem_environment
            gtest_main
)

specfem_add_test(coordinate_systems_tests
  SOURCES   coordinate_systems/utm_tests.cpp
  LIBRARIES specfem::coordinate_systems
            gtest_main
)

specfem_add_test(resolve_coordinates_tests
  SOURCES   assembly/resolve_coordinates/test_resolve_coordinates.cpp
  LIBRARIES specfem::assembly
            specfem::coordinate_systems
            specfem_environment
)

specfem_add_test(surface_elevation_tests
  SOURCES   algorithms/dim3/surface_elevation_test.cpp
  LIBRARIES specfem::assembly
            specfem::algorithms
            specfem::coordinate_systems
            specfem::mesh
            specfem::mesh_entity
            specfem_environment
            gtest_main
            Kokkos::kokkos
)

specfem_add_test(trilinos_smoke_tests
  SOURCES   linear_system/trilinos_smoke_tests.cpp
  LIBRARIES specfem::linear_system
            specfem_environment
            gtest_main
            Kokkos::kokkos
)
