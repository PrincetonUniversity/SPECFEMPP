#include "chunk_edge/field.hpp"

template <int CHUNK_SIZE, int NGLL, specfem::dimension::type DimensionType,
          bool USE_SIMD, typename FieldValFunction>
void verify_chunk_edges(std::shared_ptr<specfem::compute::assembly> assembly,
                        FieldValFunction &fieldval) {

  using ChunkEdgeAcoustic = specfem::chunk_edge::field<
      CHUNK_SIZE, NGLL, DimensionType, specfem::element::medium_tag::acoustic,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, true, true, true, USE_SIMD>;
  using ChunkEdgeElastic = specfem::chunk_edge::field<
      CHUNK_SIZE, NGLL, DimensionType, specfem::element::medium_tag::elastic,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, true, true, true, USE_SIMD>;

  // TODO redefine to chunk_edge
  using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

  int scratch_size =
      ChunkEdgeAcoustic::shmem_size() + ChunkEdgeElastic::shmem_size();

  // TODO define ChunkPolicyType
  ChunkPolicyType chunk_policy(element_kernel_index_mapping, NGLL, NGLL);

  Kokkos::parallel_for(
      "test accessor/chunk_edge.hpp",
      static_cast<const typename ChunkPolicyType::policy_type &>(chunk_policy),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        ChunkEdgeAcoustic edge_acoustic(team);
        ChunkEdgeElastic edge_elastic(team);

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

                // const auto point_property = [&]() -> PointPropertyType {
                //   PointPropertyType point_property;

                //   specfem::compute::load_on_device(index, properties,
                //                                    point_property);
                //   return point_property;
                // }();

                // const auto point_partial_derivatives =
                //     [&]() -> PointPartialDerivativesType {
                //   PointPartialDerivativesType point_partial_derivatives;
                //   specfem::compute::load_on_device(index,
                //   partial_derivatives,
                //                                    point_partial_derivatives);
                //   return point_partial_derivatives;
                // }();
              });
        }
      });

  for (int ispec = 0; ispec < nspec; ispec++) {
    switch (element_type(ispec)) {
    case specfem::element::medium_tag::acoustic: {
      constexpr auto medium = specfem::element::medium_tag::acoustic;
      ChunkEdgeAcoustic edgefield;
      break;
    }
    case specfem::element::medium_tag::elastic: {
      constexpr auto medium = specfem::element::medium_tag::elastic;
      ChunkEdgeElastic edgefield;
      break;
    }
    }
  }
}
