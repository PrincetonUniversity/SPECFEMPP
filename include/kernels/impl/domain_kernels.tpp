#pragma once

#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "chunk_element/field.hpp"
#include "chunk_element/stress_integrand.hpp"
#include "compute/assembly/assembly.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "medium/compute_mass_matrix.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "parallel_configuration/range_config.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "policies/chunk.hpp"
#include "policies/range.hpp"
#include "domain_kernels.hpp"

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kernels::impl::domain_kernels<
    WavefieldType, DimensionType,
    NGLL>::compute_stiffness_interaction(const int istep) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;
  const auto &properties = assembly.properties;
  const auto field = assembly.fields.get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;
  const auto boundary_values =
      assembly.boundary_values.get_container<boundary_tag>();

  constexpr bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  constexpr int chunk_size = parallel_config::chunk_size;

  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components();
  constexpr int num_dimensions =
      specfem::element::attributes<dimension, medium_tag>::dimension();

  using ChunkPolicyType = specfem::policy::element_chunk<parallel_config>;
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

  ChunkPolicyType chunk_policy(elements, ngll, ngll);

  constexpr int simd_size = simd::size();

  Kokkos::parallel_for(
      "specfem::kernels::impl::domain_kernels::compute_stiffness_interaction",
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
                const auto &index = iterator_index.index;

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

                const int &ielement = iterator_index.ielement;

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  for (int idim = 0; idim < num_dimensions; ++idim) {
                    stress_integrand.F(ielement, index.iz, index.ix, idim,
                                       icomponent) = F(idim, icomponent);
                  }
                }
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              team, iterator, partial_derivatives, wgll,
              element_quadrature.hprime_wgll, stress_integrand.F,
              [&, istep = istep](
                  const typename ChunkPolicyType::iterator_type::index_type
                      &iterator_index,
                  const typename PointAccelerationType::ViewType &result) {
                const auto &index = iterator_index.index;
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

                // specfem::domain::impl::boundary_conditions::
                //     apply_boundary_conditions(point_boundary, point_property,
                //                               velocity, acceleration);

                // Store forward boundary values for reconstruction during
                // adjoint simulations. The function does nothing if the
                // boundary tag is not stacey
                if constexpr (wavefield ==
                              specfem::wavefield::simulation_field::forward) {
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

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kernels::impl::domain_kernels<
    WavefieldType, DimensionType,
    NGLL>::compute_source_interaction(const int timestep) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;

  const auto elements = assembly.sources.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag, WavefieldType);

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

  auto &sources = assembly.sources;
  const auto &properties = assembly.properties;
  const auto &boundaries = assembly.boundaries;
  const auto field = assembly.fields.get_simulation_field<wavefield>();

  sources.update_timestep(timestep);

  using PointSourcesType =
      specfem::point::source<dimension, medium_tag, wavefield>;
  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, false>;
  using PointVelocityType = specfem::point::field<dimension, medium_tag, false,
                                                  true, false, false, false>;

  using simd = specfem::datatype::simd<type_real, false>;
  constexpr int simd_size = simd::size();

#ifdef KOKKOS_ENABLE_CUDA
  constexpr int nthreads = 32;
  constexpr int lane_size = 1;
#else
  constexpr int nthreads = 1;
  constexpr int lane_size = 1;
#endif

  using ParallelConfig =
      specfem::parallel_config::chunk_config<DimensionType, 1, 1, nthreads,
                                             lane_size, simd,
                                             Kokkos::DefaultExecutionSpace>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;

  ChunkPolicy chunk_policy(elements, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::kernels::impl::domain_kernels::compute_source_interaction",
      static_cast<const typename ChunkPolicy::policy_type &>(chunk_policy),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicy::member_type &team) {
        for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
             tile += ChunkPolicy::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicy::tile_size * simd_size + tile;

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

                PointSourcesType point_source;
                specfem::compute::load_on_device(index, sources, point_source);

                PointPropertiesType point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);

                const auto acceleration =
                    specfem::medium::compute_source_contribution(
                        point_source, point_property);

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                PointVelocityType velocity;
                specfem::compute::load_on_device(index, field, velocity);

                // specfem::domain::impl::boundary_conditions::
                //     apply_boundary_conditions(point_boundary, point_property,
                //                               velocity, acceleration);

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        }
      });

  Kokkos::fence();
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void specfem::kernels::impl::domain_kernels<
    WavefieldType, DimensionType, NGLL>::compute_seismograms(const int
                                                                 &isig_step) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;

  const auto elements =
      assembly.receivers.get_elements_on_device(MediumTag, PropertyTag);

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

  auto &receivers = assembly.receivers;
  const auto seismogram_types = receivers.get_seismogram_types();

  const int nseismograms = seismogram_types.size();
  const auto field = assembly.fields.get_simulation_field<WavefieldType>();
  const auto &quadrature = assembly.mesh.quadratures;

  constexpr bool using_simd = false;

  using no_simd = specfem::datatype::simd<type_real, using_simd>;

  constexpr int simd_size = no_simd::size();

#ifdef KOKKOS_ENABLE_CUDA
  constexpr int nthreads = 32;
  constexpr int lane_size = 1;
#else
  constexpr int nthreads = 1;
  constexpr int lane_size = 1;
#endif

  using ParallelConfig =
      specfem::parallel_config::chunk_config<DimensionType, 1, 1, nthreads,
                                             lane_size, no_simd,
                                             Kokkos::DefaultExecutionSpace>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;
  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, ngll, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, true, true, false, using_simd>;
  using ElementQuadratureType = specfem::element::quadrature<
      ngll, DimensionType, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;
  using ViewType = specfem::datatype::ScalarChunkViewType<
      type_real, ParallelConfig::chunk_size, ngll, 2,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      false>;
  using ResultsViewType =
      Kokkos::View<type_real[ParallelConfig::chunk_size][2], Kokkos::LayoutLeft,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ElementQuadratureType::shmem_size() +
                     2 * ViewType::shmem_size() + ResultsViewType::shmem_size();

  receivers.set_seismogram_step(isig_step);

  ChunkPolicy policy(elements, ngll, ngll);

  for (int iseis = 0; iseis < nseismograms; ++iseis) {

    receivers.set_seismogram_type(iseis);
    const auto wavefield_component = seismogram_types[iseis];

    Kokkos::parallel_for(
        "specfem::kernels::impl::domain_kernels::compute_seismograms",
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const typename ChunkPolicy::member_type &team_member) {
          // Instantiate shared views
          // ----------------------------------------------------------------
          ChunkElementFieldType element_field(team_member);
          ElementQuadratureType element_quadrature(team_member);
          ViewType wavefield(team_member.team_scratch(0));
          ViewType lagrange_interpolant(team_member.team_scratch(0));
          ResultsViewType seismogram_components(team_member.team_scratch(0));

          specfem::compute::load_on_device(team_member, quadrature,
                                           element_quadrature);

          for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
               tile += ChunkPolicy::chunk_size * simd_size) {
            const int starting_element_index =
                team_member.league_rank() * ChunkPolicy::tile_size * simd_size +
                tile;

            if (starting_element_index >= nelements) {
              break;
            }

            const auto iterator =
                policy.league_iterator(starting_element_index);

            specfem::compute::load_on_device(team_member, iterator, field,
                                             element_field);

            team_member.team_barrier();

            specfem::medium::compute_wavefield<MediumTag, PropertyTag>(
                team_member, iterator, assembly, element_quadrature,
                element_field, wavefield_component, wavefield);

            specfem::compute::load_on_device(team_member, iterator, receivers,
                                             lagrange_interpolant);

            team_member.team_barrier();

            specfem::algorithms::interpolate_function(
                team_member, iterator, lagrange_interpolant, wavefield,
                seismogram_components);

            team_member.team_barrier();

            specfem::compute::store_on_device(team_member, iterator,
                                              seismogram_components, receivers);
          }
        });

    Kokkos::fence();
  }

  return;
}
