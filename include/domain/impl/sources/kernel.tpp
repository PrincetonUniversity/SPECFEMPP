#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_TPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "medium/compute_source.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::source_kernel(const specfem::compute::assembly &assembly)
    : sources(assembly.sources), properties(assembly.properties),
      field(assembly.fields.get_simulation_field<WavefieldType>()) {

  this->elements = sources.get_elements_on_device(medium_tag, PropertyTag, specfem::element::boundary_tag::none, wavefield_tag);
  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
void specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::compute_source_interaction(const int timestep) {

  sources.update_timestep(timestep);

  const int nelements = elements.size();

  if (nelements == 0)
    return;

  using PointSourcesType =
      specfem::point::source<dimension, medium_tag, wavefield_tag>;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;

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
      "specfem::domain::impl::kernels::elements::compute_source_interaction",
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

                PointPropertiesType point_properties;
                specfem::compute::load_on_device(index, properties,
                                                 point_properties);

                const auto acceleration =
                    specfem::medium::compute_source_contribution(
                        point_source, point_properties);

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        }
      });

  Kokkos::fence();
}

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_TPP
