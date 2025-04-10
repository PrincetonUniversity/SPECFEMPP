#pragma once

#include "chunk_element/field.hpp"
#include "compute/assembly/assembly.hpp"
#include "compute_energy.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_energy.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/field.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
type_real specfem::kokkos_kernels::impl::compute_energy(
    const specfem::compute::assembly &assembly) {

  constexpr auto dimension = DimensionType;
  constexpr auto wavefield = WavefieldType;
  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;

  const auto &properties = assembly.properties;
  const auto &kernels = assembly.kernels;
  const auto &field = assembly.fields.get_simulation_field<wavefield>();
  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;

  auto &weights = quadrature.gll.weights;

  const auto element_index =
      assembly.element_types.get_elements_on_device(medium_tag, property_tag);

  const int nelements = element_index.size();

  if (nelements == 0) {
    return 0.0;
  }

  constexpr static bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, NGLL, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, true, true, false, using_simd>;

  using ViewType = typename ChunkElementFieldType::ViewType;

  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, dimension, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension, medium_tag, using_simd>;

  using PointFieldType = specfem::point::field<dimension, medium_tag, false,
                                               true, false, true, using_simd>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;

  int scratch_size =
      ChunkElementFieldType::shmem_size() + ElementQuadratureType::shmem_size();

  ChunkPolicy chunk_policy(element_index, NGLL, NGLL);

  constexpr int simd_size = simd::size();

  type_real total_energy = 0.0;

  Kokkos::parallel_reduce(
      "specfem::frechet_derivatives::frechet_elements::compute",
      chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const typename ChunkPolicy::member_type &team,
                    type_real &chunk_energy) {
        // Allocate scratch memory
        ChunkElementFieldType element_field(team);
        ElementQuadratureType element_quadrature(team);
        ViewType energy(team.team_scratch(0));

        specfem::compute::load_on_device(team, quadrature, element_quadrature);

        for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
             tile += ChunkPolicy::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicy::tile_size * simd_size + tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          // Populate Scratch Views
          specfem::compute::load_on_device(team, iterator, field,
                                           element_field);

          team.team_barrier();

          specfem::medium::compute_energy<wavefield, dimension, medium_tag,
                                          property_tag>(
              team, iterator, assembly, element_quadrature, element_field,
              [&](const typename ChunkPolicy::iterator_type::index_type
                      &iterator_index,
                  const typename simd::datatype &point_energy) {
                const auto &index = iterator_index.index;
                const int ix = index.ix;
                const int iz = index.iz;

                const auto jacobian = [&]() {
                  specfem::point::partial_derivatives<DimensionType, true,
                                                      using_simd>
                      point_partial_derivatives;
                  specfem::compute::load_on_device(index, partial_derivatives,
                                                   point_partial_derivatives);
                  return point_partial_derivatives.jacobian;
                }();

                energy(iterator_index.ielement, iz, ix, 0) =
                    point_energy * weights(ix) * weights(iz) * jacobian;

                return;
              });

          team.team_barrier();

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int &i, type_real &local_energy) {
                const auto iterator_index = iterator(i);
                const auto &index = iterator_index.index;
                const int ix = index.ix;
                const int iz = index.iz;

                for (int j = 0; j < simd_size; ++j) {
                  local_energy +=
                      energy(iterator_index.ielement, iz, ix, 0)[j];
                }

                // energy(iterator_index.ielement, iz, ix, 0);
              },
              chunk_energy);
        }
      },
      total_energy);

  return total_energy;
}
