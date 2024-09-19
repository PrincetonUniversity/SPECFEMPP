#pragma once

#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "compute/assembly/assembly.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "element.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kernel.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    NGLL>::element_kernel_base(const specfem::compute::assembly &assembly,
                               const specfem::kokkos::HostView1d<int>
                                   h_element_kernel_index_mapping)
    : nelements(h_element_kernel_index_mapping.extent(0)),
      element_kernel_index_mapping("specfem::domain::impl::kernels::element_"
                                   "kernel_base::element_kernel_index_mapping",
                                   nelements),
      h_element_kernel_index_mapping(h_element_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      properties(assembly.properties), boundaries(assembly.boundaries),
      boundary_values(assembly.boundary_values.get_container<BoundaryTag>()) {

  // Check if the elements being allocated to this kernel are of the correct
  // type
  for (int ispec = 0; ispec < nelements; ispec++) {
    const int ielement = h_element_kernel_index_mapping(ispec);
    if ((assembly.properties.h_element_types(ielement) != MediumTag) &&
        (assembly.properties.h_element_property(ielement) != PropertyTag)) {
      throw std::runtime_error("Invalid element detected in kernel");
    }
  }

  // Assert that ispec of the elements is contiguous
  for (int ispec = 0; ispec < nelements; ispec++) {
    if (ispec != 0) {
      if (h_element_kernel_index_mapping(ispec) !=
          h_element_kernel_index_mapping(ispec - 1) + 1) {
        throw std::runtime_error("Element index mapping is not contiguous");
      }
    }
  }

  Kokkos::deep_copy(element_kernel_index_mapping,
                    h_element_kernel_index_mapping);
  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    NGLL>::compute_mass_matrix(const type_real dt,
                               const specfem::compute::simulation_field<
                                   WavefieldType> &field) const {
  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;

  constexpr int simd_size = simd::size();

  ChunkPolicyType chunk_policy(element_kernel_index_mapping, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_mass_matrix",
      static_cast<const typename ChunkPolicyType::policy_type &>(chunk_policy),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
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
                    specfem::domain::impl::elements::mass_matrix_component(
                        point_property, point_partial_derivatives);

                for (int icomp = 0; icomp < components; icomp++) {
                  mass_matrix.mass_matrix(icomp) *= wgll(ix) * wgll(iz);
                }

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                specfem::domain::impl::boundary_conditions::
                    compute_mass_matrix_terms(dt, point_boundary,
                                              point_property, mass_matrix);

                specfem::compute::atomic_add_on_device(index, mass_matrix,
                                                       field);
              });
        }
      });

  Kokkos::fence();

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag, NGLL>::
    compute_stiffness_interaction(
        const int istep,
        const specfem::compute::simulation_field<WavefieldType> &field) const {

  if (nelements == 0)
    return;

  const auto hprime = quadrature.gll.hprime;
  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  ChunkPolicyType chunk_policy(element_kernel_index_mapping, NGLL, NGLL);

  constexpr int simd_size = simd::size();

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_interaction",
      chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        ChunkElementFieldType element_field(team);
        ElementQuadratureType element_quadrature(team);
        ChunkStressIntegrandType stress_integrand(team);

        specfem::compute::load_on_device(team, quadrature, element_quadrature);
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
          specfem::compute::load_on_device(team, iterator, field,
                                           element_field);

          team.team_barrier();

          specfem::algorithms::gradient(
              team, iterator, partial_derivatives,
              element_quadrature.hprime_gll, element_field.displacement,
              // Compute stresses using the gradients
              [&](const typename ChunkPolicyType::iterator_type::index_type
                      &iterator_index,
                  const typename PointFieldDerivativesType::ViewType &du) {
                const auto index = iterator_index.index;

                PointPartialDerivativesType point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);

                PointPropertyType point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);

                PointFieldDerivativesType field_derivatives(du);

                const auto point_stress_integrand =
                    specfem::domain::impl::elements::compute_stress_integrands(
                        point_partial_derivatives, point_property,
                        field_derivatives);

                const int ielement = iterator_index.ielement;

                for (int idim = 0; idim < num_dimensions; ++idim) {
                  for (int icomponent = 0; icomponent < components;
                       ++icomponent) {
                    stress_integrand.F(ielement, index.iz, index.ix, idim,
                                       icomponent) =
                        point_stress_integrand.F(idim, icomponent);
                  }
                }
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              team, iterator, partial_derivatives, wgll,
              element_quadrature.hprime_wgll, stress_integrand.F,
              [&](const typename ChunkPolicyType::iterator_type::index_type
                      &iterator_index,
                  const typename PointAccelerationType::ViewType &result) {
                auto index = iterator_index.index;
                PointAccelerationType acceleration(result);

                for (int icomponent = 0; icomponent < components;
                     icomponent++) {
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

                specfem::domain::impl::boundary_conditions::
                    apply_boundary_conditions(point_boundary, point_property,
                                              velocity, acceleration);

                // Store forward boundary values for reconstruction during
                // adjoint simulations. The function does nothing if the
                // boundary tag is not stacey
                if constexpr (WavefieldType ==
                              specfem::wavefield::type::forward) {
                  specfem::compute::store_on_device(istep, index, acceleration,
                                                    boundary_values);
                }

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        }
      });

  Kokkos::fence();

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType,
          specfem::element::property_tag PropertyTag, int NGLL>
void specfem::domain::impl::kernels::element_kernel<
    specfem::wavefield::type::backward, DimensionType, MediumType, PropertyTag,
    specfem::element::boundary_tag::stacey,
    NGLL>::compute_stiffness_interaction(const int istep) const {

  if (this->nelements == 0)
    return;

  ChunkPolicyType chunk_policy(this->element_kernel_index_mapping, NGLL, NGLL);

  constexpr int simd_size = simd::size();

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_"
      "interaction",
      static_cast<const typename ChunkPolicyType::policy_type &>(chunk_policy),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
             tile += ChunkPolicyType::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::tile_size * simd_size +
              tile;

          if (starting_element_index >= this->nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int i) {
                const auto iterator_index = iterator(i);
                const auto index = iterator_index.index;

                PointAccelerationType acceleration;
                specfem::compute::load_on_device(
                    istep, index, this->boundary_values, acceleration);

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       this->field);
              });
        }
      });
}
