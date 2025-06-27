#pragma once

#include "algorithms/interpolate.hpp"
#include "chunk_element/field.hpp"
#include "specfem/assembly.hpp"
#include "compute_seismogram.hpp"
#include "datatypes/simd.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_wavefield.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "execution/mapped_chunked_domain_iterator.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void specfem::kokkos_kernels::impl::compute_seismograms(
    specfem::compute::assembly &assembly, const int &isig_step) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto wavefield_type = WavefieldType;
  constexpr int ngll = NGLL;
  constexpr auto dimension = DimensionTag;

  const auto [elements, receiver_indices] =
      assembly.receivers.get_indices_on_device(medium_tag, property_tag);

  const int ngllz = assembly.mesh.ngllz;
  const int ngllx = assembly.mesh.ngllx;

  const int nreceivers = receiver_indices.extent(0);

  if (nreceivers == 0)
    return;

  auto &receivers = assembly.receivers;
  const auto seismogram_types = receivers.get_seismogram_types();

  const int nseismograms = seismogram_types.size();
  const auto field = assembly.fields.get_simulation_field<wavefield_type>();
  const auto &quadrature = assembly.mesh.quadratures;


  if (ngllz != ngll || ngllx != ngll) {
    throw std::runtime_error("The number of GLL points in z and x must match "
                             "the template parameter NGLL.");
  }

  constexpr bool using_simd = false;

  using no_simd = specfem::datatype::simd<type_real, using_simd>;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr int nthreads = 32;
  constexpr int lane_size = 1;
#else
  constexpr int nthreads = 1;
  constexpr int lane_size = 1;
#endif

  using ParallelConfig =
      specfem::parallel_config::chunk_config<dimension, 1, 1, nthreads,
                                             lane_size, no_simd,
                                             Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, ngll, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, true, true, false, using_simd>;
  using ElementQuadratureType = specfem::element::quadrature<
      ngll, dimension, specfem::kokkos::DevScratchSpace,
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

  specfem::execution::MappedChunkedDomainIterator chunk(
      ParallelConfig(), elements, receiver_indices, ngllz, ngllx);

  for (int iseis = 0; iseis < nseismograms; ++iseis) {

    receivers.set_seismogram_type(iseis);
    const auto wavefield_component = seismogram_types[iseis];

    specfem::execution::for_each_level(
        "specfem::kokkos_kernels::compute_seismograms",
        chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const typename decltype(chunk)::index_type &chunk_index) {
          const auto team = chunk_index.get_policy_index();
          ChunkElementFieldType element_field(team);
          ElementQuadratureType element_quadrature(team);
          ViewType wavefield(team.team_scratch(0));
          ViewType lagrange_interpolant(team.team_scratch(0));
          ResultsViewType seismogram_components(team.team_scratch(0));

          specfem::compute::load_on_device(team, quadrature,
                                           element_quadrature);

          specfem::compute::load_on_device(chunk_index, field, element_field);
          team.team_barrier();

          specfem::medium::compute_wavefield<medium_tag, property_tag>(
              chunk_index, assembly, element_quadrature, element_field,
              wavefield_component, wavefield);

          specfem::compute::load_on_device(chunk_index, receivers,
                                           lagrange_interpolant);

          team.team_barrier();

          specfem::algorithms::interpolate_function(
              chunk_index, lagrange_interpolant, wavefield,
              seismogram_components);
          team.team_barrier();
          specfem::compute::store_on_device(chunk_index, seismogram_components,
                                            receivers);
        });
  }

  return;
}
