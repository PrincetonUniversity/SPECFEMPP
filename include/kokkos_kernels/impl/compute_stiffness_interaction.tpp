#pragma once

#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "boundary_conditions/boundary_conditions.tpp"
#include "chunk_element/field.hpp"
#include "chunk_element/stress_integrand.hpp"
#include "compute/assembly/assembly.hpp"
#include "datatypes/simd.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_damping_force.hpp"
#include "medium/compute_stress.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
int specfem::kokkos_kernels::impl::compute_stiffness_interaction(
    const specfem::compute::assembly &assembly, const int &istep) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension = DimensionTag;
  constexpr int ngll = NGLL;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

  const int nelements = elements.extent(0);

  const int ngllz = assembly.mesh.ngllz;
  const int ngllx = assembly.mesh.ngllx;

  if (nelements == 0)
    return 0;

  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;
  const auto &properties = assembly.properties;
  const auto field = assembly.fields.get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;
  const auto boundary_values =
      assembly.boundary_values.get_container<boundary_tag>();

  if (ngllz != NGLL || ngllx != NGLL) {
    throw std::runtime_error("The number of GLL points in z and x must match "
                             "the template parameter NGLL.");
  }

#ifdef KOKKOS_ENABLE_CUDA
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  constexpr int chunk_size = parallel_config::chunk_size;

  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components;
  constexpr int num_dimensions =
      specfem::element::attributes<dimension, medium_tag>::dimension;

  using ChunkElementFieldType = specfem::chunk_element::field<
      parallel_config::chunk_size, ngll, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;
  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      parallel_config::chunk_size, ngll, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      using_simd>;
  using ElementQuadratureType = specfem::element::quadrature<
      ngll, dimension, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, using_simd>;
  using PointVelocityType =
      specfem::point::field<dimension, medium_tag, false, true, false, false,
                            using_simd>;
  using PointAccelerationType =
      specfem::point::field<dimension, medium_tag, false, false, true, false,
                            using_simd>;
  using PointPartialDerivativesType =
      specfem::point::partial_derivatives<dimension, true, using_simd>;
  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension, medium_tag, using_simd>;

  const auto wgll = assembly.mesh.quadratures.gll.weights;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements,
                                               ngllz, ngllx);

  constexpr int simd_size = simd::size();

  if constexpr (BoundaryTag == specfem::element::boundary_tag::stacey &&
                WavefieldType ==
                    specfem::wavefield::simulation_field::backward) {

    specfem::execution::for_all(
        "specfem::domain::impl::kernels::elements::compute_stiffness_"
        "interaction",
        chunk,
        KOKKOS_LAMBDA(
            const specfem::point::index<dimension, using_simd> &index) {
          PointAccelerationType acceleration;
          specfem::compute::load_on_device(istep, index, boundary_values,
                                           acceleration);

          specfem::compute::atomic_add_on_device(index, acceleration, field);
        });
  } else {

    specfem::execution::for_each_level(
        "specfem::domain::impl::kernels::elements::compute_stiffness_"
        "interaction",
        chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const typename decltype(chunk)::index_type &chunk_index) {
          const auto team = chunk_index.get_policy_index();
          ChunkElementFieldType element_field(team);
          ElementQuadratureType element_quadrature(team);
          ChunkStressIntegrandType stress_integrand(team);
          specfem::compute::load_on_device(team, quadrature,
                                           element_quadrature);
          specfem::compute::load_on_device(chunk_index, field, element_field);

          team.team_barrier();

          specfem::algorithms::gradient(
              chunk_index, partial_derivatives, element_quadrature.hprime_gll,
              element_field.displacement,
              [&](const auto &iterator_index,
                  const typename PointFieldDerivativesType::value_type &du) {
                const auto &index = iterator_index.get_index();
                const int &ielement = iterator_index.get_policy_index();
                PointPartialDerivativesType point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);

                PointPropertyType point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);

                PointFieldDerivativesType field_derivatives(du);

                const auto point_stress = specfem::medium::compute_stress(
                    point_property, field_derivatives);

                const auto F = point_stress * point_partial_derivatives;

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  for (int idim = 0; idim < num_dimensions; ++idim) {
                    stress_integrand.F(ielement, index.iz, index.ix, icomponent,
                                       idim) = F(icomponent, idim);
                  }
                }
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              chunk_index, partial_derivatives, wgll,
              element_quadrature.hprime_wgll, stress_integrand.F,
              [&](const auto &iterator_index,
                  const typename PointAccelerationType::value_type &result) {
                const auto &index = iterator_index.get_index();
                PointAccelerationType acceleration(result);

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  acceleration.acceleration(icomponent) *=
                      static_cast<type_real>(-1.0);
                }

                PointPropertyType point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);

                PointVelocityType velocity;
                specfem::compute::load_on_device(index, field, velocity);

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                const auto jacobian = [&]() {
                  specfem::point::partial_derivatives<dimension, true,
                                                      using_simd>
                      point_partial_derivatives;
                  specfem::compute::load_on_device(index, partial_derivatives,
                                                   point_partial_derivatives);
                  return point_partial_derivatives.jacobian;
                }();

                const auto factor = quadrature.gll.weights(index.iz) *
                                    quadrature.gll.weights(index.ix) * jacobian;

                specfem::medium::compute_damping_force(factor, point_property,
                                                       velocity, acceleration);

                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, point_property, velocity, acceleration);

                // Store forward boundary values for reconstruction during
                // adjoint simulations. The function does nothing if the
                // boundary tag is not stacey
                if (wavefield ==
                    specfem::wavefield::simulation_field::forward) {
                  specfem::compute::store_on_device(istep, index, acceleration,
                                                    boundary_values);
                }

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        });
  }

  return nelements;
}
