#pragma once

#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "boundary_conditions/boundary_conditions.tpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "execution/for_each_level.hpp"
#include "medium/compute_cosserat_couple_stress.hpp"
#include "medium/compute_cosserat_stress.hpp"
#include "medium/compute_damping_force.hpp"
#include "medium/compute_stress.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/lagrange_derivative.hpp"
#include "specfem/assembly.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
int specfem::kokkos_kernels::impl::compute_stiffness_interaction(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const int &istep) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension_tag = DimensionTag;
  constexpr int ngll = NGLL;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

  // Get the number of elements that match the specified tags
  const int nelements = elements.extent(0);

  // Get the element grid information (ngll, ngllx, ngllz, order)
  const auto &element_grid = assembly.mesh.element_grid;

  // Return if there are no elements matching the tag combination
  if (nelements == 0)
    return 0;

  // Check if the number of GLL points in the mesh elements matches the template
  if (element_grid != NGLL) {
    throw std::runtime_error(
        "The number of GLL points in the mesh elements must match "
        "the template parameter NGLL.");
  }

  // Alias some assembly members for easier acces
  const auto &mesh = assembly.mesh;
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &properties = assembly.properties;
  const auto &boundaries = assembly.boundaries;

  // Get the simulation field and boundary values
  const auto field = assembly.fields.template get_simulation_field<wavefield>();
  const auto boundary_values =
      assembly.boundary_values.template get_container<boundary_tag>();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  // TODO(Rohit : DIM3_SIMD) Enable simd execution for dim3 solver
  constexpr bool using_simd = (DimensionTag == specfem::dimension::type::dim2) ? true : false;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
                  dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::displacement<
      ParallelConfig::chunk_size, ngll, dimension_tag, medium_tag, using_simd>;
  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      ParallelConfig::chunk_size, ngll, dimension_tag, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      using_simd>;
  using ElementQuadratureType = specfem::quadrature::lagrange_derivative<
      ngll, dimension_tag, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, using_simd>;
  using PointDisplacementType =
      specfem::point::displacement<dimension_tag, medium_tag, using_simd>;
  using PointVelocityType =
      specfem::point::velocity<dimension_tag, medium_tag, using_simd>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension_tag, medium_tag, using_simd>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, using_simd>;
  using PointPropertyType =
      specfem::point::properties<dimension_tag, medium_tag, property_tag,
                                 using_simd>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension_tag, medium_tag, using_simd>;

  using PointWeightsType = specfem::point::weights<dimension_tag>;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size() +
                     ElementQuadratureType::shmem_size();


  specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), elements,
                                                  element_grid);

  Kokkos::Profiling::pushRegion("Compute Stiffness Interaction");

  if constexpr (BoundaryTag == specfem::element::boundary_tag::stacey &&
                WavefieldType ==
                    specfem::wavefield::simulation_field::backward) {

    specfem::execution::for_all(
        "specfem::kokkos_kernels::compute_stiffness_interaction", chunk,
        KOKKOS_LAMBDA(
            const typename decltype(chunk)::base_index_type &iterator_index) {
            const auto index = iterator_index.get_index();
          PointAccelerationType acceleration;
          specfem::assembly::load_on_device(istep, index, boundary_values,
                                            acceleration);

          specfem::assembly::atomic_add_on_device(index, field, acceleration);
        });
  } else {

    specfem::execution::for_each_level(
        "specfem::kokkos_kernels::compute_stiffness_interaction",
        chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(
            const typename decltype(chunk)::index_type &chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();
          ChunkElementFieldType element_field(team.team_scratch(0));
          ElementQuadratureType lagrange_derivative(team);
          ChunkStressIntegrandType stress_integrand(team);
          specfem::assembly::load_on_device(team, mesh, lagrange_derivative);
          specfem::assembly::load_on_device(chunk_index, field, element_field);

          team.team_barrier();

          specfem::algorithms::gradient(
              chunk_index, jacobian_matrix, lagrange_derivative, element_field,
              [&](const auto &iterator_index,
                  const typename PointFieldDerivativesType::value_type &du) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();
                PointJacobianMatrixType point_jacobian_matrix;
                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                PointFieldDerivativesType field_derivatives(du);

                PointDisplacementType point_displacement;
                specfem::assembly::load_on_device(index, field,
                                                  point_displacement);

                auto point_stress = specfem::medium::compute_stress(
                    point_property, field_derivatives);

                specfem::medium::compute_cosserat_stress(
                    point_property, point_displacement, point_stress);

                stress_integrand.F(local_index) =
                    point_stress * point_jacobian_matrix;
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              chunk_index, mesh.weights, lagrange_derivative,
              stress_integrand.F,
              [&](const auto &iterator_index,
                  const typename PointAccelerationType::value_type &result) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();
                PointAccelerationType acceleration(result);

                acceleration *= static_cast<type_real>(-1.0);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                PointVelocityType velocity;
                specfem::assembly::load_on_device(index, field, velocity);

                PointBoundaryType point_boundary;
                specfem::assembly::load_on_device(index, boundaries,
                                                  point_boundary);

                PointWeightsType point_weights;
                specfem::assembly::load_on_device(index, mesh.weights,
                                                  point_weights);

                specfem::point::jacobian_matrix<dimension_tag, true, using_simd>
                    point_jacobian_matrix;

                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                // Computing the integration factor
                const auto factor =
                    point_weights.product() * point_jacobian_matrix.jacobian;

                specfem::medium::compute_damping_force(factor, point_property,
                                                       velocity, acceleration);

                // Compute the couple stress from the stress integrand
                specfem::medium::compute_couple_stress(
                    point_jacobian_matrix, point_property, factor,
                    stress_integrand.F(local_index), acceleration);

                // Apply boundary conditions
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, point_property, velocity, acceleration);

                // Store forward boundary values for reconstruction during
                // adjoint simulations. The function does nothing if the
                // boundary tag is not stacey
                if (wavefield ==
                    specfem::wavefield::simulation_field::forward) {
                  specfem::assembly::store_on_device(istep, index, acceleration,
                                                     boundary_values);
                }

                specfem::assembly::atomic_add_on_device(index, field,
                                                        acceleration);
              });
        });
  }

  Kokkos::Profiling::popRegion();

  return nelements;
}
