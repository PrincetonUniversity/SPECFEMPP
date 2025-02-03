#pragma once

#include "chunk_element/field.hpp"
#include "compute/assembly/assembly.hpp"
#include "datatypes/simd.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/properties.hpp"
#include "point/sources.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_source_interaction(
    specfem::compute::assembly &assembly, const int &timestep) {

constexpr auto medium_tag = MediumTag;
constexpr auto property_tag = PropertyTag;
constexpr auto boundary_tag = BoundaryTag;
constexpr auto dimension = DimensionType;
constexpr int ngll = NGLL;
constexpr auto wavefield = WavefieldType;

const auto [element_indices, source_indices] = assembly.sources.get_sources_on_device(
    MediumTag, PropertyTag, BoundaryTag, WavefieldType);

auto &sources = assembly.sources;

const int nsources = source_indices.extent(0);

if (nsources == 0)
  return;

// Some aliases
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

using ChunkPolicy = specfem::policy::mapped_element_chunk<ParallelConfig>;

ChunkPolicy mapped_policy(element_indices, source_indices, NGLL, NGLL);

Kokkos::parallel_for(
    "specfem::kernels::impl::domain_kernels::compute_source_interaction",
    static_cast<const typename ChunkPolicy::policy_type &>(mapped_policy),
    KOKKOS_LAMBDA(const typename ChunkPolicy::member_type &team) {
      for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
           tile += ChunkPolicy::chunk_size * simd_size) {
        const int starting_element_index =
            team.league_rank() * ChunkPolicy::tile_size * simd_size + tile;

        if (starting_element_index >= nsources) {
          break;
        }

        // This is a mapped_chunk iterator
        const auto mapped_iterator =
            mapped_policy.mapped_league_iterator(starting_element_index);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, mapped_iterator.chunk_size()),
            [&](const int i) {
              // mapped_chunk_index_type
              const auto mapped_iterator_index = mapped_iterator(i);

              // element_index is specfem::point::index
              const auto element_index = mapped_iterator_index.index;

              // need mapped_chunk_index here to get the imap=isource
              PointSourcesType point_source;
              specfem::compute::load_on_device(mapped_iterator_index, sources,
                                               point_source);

              PointPropertiesType point_property;
              specfem::compute::load_on_device(element_index, properties,
                                               point_property);

              auto acceleration =
                  specfem::medium::compute_source_contribution(point_source,
                                                               point_property);

            //   PointBoundaryType point_boundary;
            //   specfem::compute::load_on_device(element_index, boundaries,
            //                                    point_boundary);

            //   PointVelocityType velocity;
            //   specfem::compute::load_on_device(element_index, field, velocity);

            //   specfem::boundary_conditions::
            //       apply_boundary_conditions(point_boundary, point_property,
            //                                 velocity, acceleration);

              specfem::compute::atomic_add_on_device(element_index, acceleration,
                                                     field);
            });
      }
    });

Kokkos::fence();
}
