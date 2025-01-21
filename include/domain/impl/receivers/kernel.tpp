#ifndef _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP
#define _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP

#include "algorithms/interpolate.hpp"
#include "domain/impl/receivers/acoustic/interface.hpp"
#include "domain/impl/receivers/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "medium/compute_wavefield.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
specfem::domain::impl::kernels::receiver_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::receiver_kernel(const specfem::compute::assembly &assembly)
    : assembly(assembly) {

  this->elements =
      assembly.receivers.get_elements_on_device(MediumTag, PropertyTag);
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
void specfem::domain::impl::kernels::receiver_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::compute_seismograms(const int &isig_step) const {
  const int nelements = elements.extent(0);
  constexpr int ngll = NGLL;

  if (nelements == 0)
    return;

  auto receivers = assembly.receivers;
  const auto seismogram_types = receivers.get_seismogram_types();

  auto field = assembly.fields.get_simulation_field<WavefieldType>();
  auto quadrature = assembly.mesh.quadratures;

  const int nseismograms = seismogram_types.size();

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
        "specfem::domain::impl::receivers::compute_seismograms",
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
#endif /* _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP */
