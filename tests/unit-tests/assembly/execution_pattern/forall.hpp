#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace execution_pattern {
template <bool using_simd, typename ViewType, typename ClosureType>
void forall(const std::string name, const ViewType &elements, int ngll,
            const ClosureType closure) {

  using ExecutionSpace = typename ViewType::execution_space;

  // Create a chunk policy to iterate over the elements
  constexpr auto dimension = specfem::dimension::type::dim2;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  constexpr int simd_size = simd::size();
  using ParallelConfig =
      specfem::parallel_config::default_chunk_config<dimension, simd,
                                                     ExecutionSpace>;
  using PolicyType = specfem::policy::element_chunk<ParallelConfig>;
  PolicyType policy(elements, ngll, ngll);

  Kokkos::parallel_for(
      name, static_cast<const typename PolicyType::policy_type &>(policy),
      KOKKOS_LAMBDA(const typename PolicyType::member_type &team) {
        for (int tile = 0; tile < PolicyType::tile_size * simd_size;
             tile += PolicyType::chunk_size * simd_size) {
          const auto iterator =
              policy.league_iterator(team.league_rank(), tile);
          if (iterator.is_end()) {
            return; // No elements to process in this tile
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [=](const int i) {
                const auto iterator_index = iterator(i);
                closure(iterator_index);
              });
        }
      });
}
} // namespace execution_pattern
