#pragma once

#include "compute/assembly/assembly.hpp"
#include "datatypes/simd.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "policies/chunk.hpp"
#include "medium/compute_mass_matrix.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_mass_matrix(
    const type_real &dt, const specfem::compute::assembly &assembly) {

  constexpr auto dimension = DimensionType;
  constexpr auto wavefield = WavefieldType;
  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components();

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

  constexpr bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkPolicyType = specfem::policy::element_chunk<parallel_config>;

  using PointMassType = specfem::point::field<dimension, medium_tag, false,
                                              false, false, true, using_simd>;

  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;

  using PointPartialDerivativesType =
      specfem::point::partial_derivatives<dimension, true, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, using_simd>;

  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;
  const auto &properties = assembly.properties;
    const auto &boundaries = assembly.boundaries;
  const auto field = assembly.fields.get_simulation_field<wavefield>();

  const auto wgll = quadrature.gll.weights;

  constexpr int simd_size = simd::size();

  ChunkPolicyType chunk_policy(elements, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_mass_matrix",
      static_cast<const typename ChunkPolicyType::policy_type &>(chunk_policy),
      KOKKOS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
             tile += ChunkPolicyType::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::tile_size * simd_size +
              tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int i) {
                const auto iterator_index = iterator(i);
                const auto index = iterator_index.index;
                const int ix = iterator_index.index.ix;
                const int iz = iterator_index.index.iz;

                const auto point_property = [&]() -> PointPropertyType {
                  PointPropertyType point_property;

                  specfem::compute::load_on_device(index, properties,
                                                   point_property);
                  return point_property;
                }();

                const auto point_partial_derivatives =
                    [&]() -> PointPartialDerivativesType {
                  PointPartialDerivativesType point_partial_derivatives;
                  specfem::compute::load_on_device(index, partial_derivatives,
                                                   point_partial_derivatives);
                  return point_partial_derivatives;
                }();

                PointMassType mass_matrix =
                    specfem::medium::mass_matrix_component(
                        point_property, point_partial_derivatives);

                for (int icomp = 0; icomp < components; icomp++) {
                  mass_matrix.mass_matrix(icomp) *= wgll(ix) * wgll(iz);
                }

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                specfem::boundary_conditions::
                    compute_mass_matrix_terms(dt, point_boundary,
                                              point_property, mass_matrix);

                specfem::compute::atomic_add_on_device(index, mass_matrix,
                                                       field);
              });
        }
      });

  Kokkos::fence();
}
