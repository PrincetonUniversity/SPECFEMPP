#pragma once

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

namespace specfem {
namespace policy {
/**
 * @brief Chunk policy for parallel execution
 *
 * A version of Kokkos::TeamPolicy that is used to chunk a set of elements into
 * chunks of ParallelConfig::ChunkSize. Every physical team gets a consecutive
 * chunk of elements to work on.
 *
 * For best performance, every thread in a team should work on a different
 * elements, and the data should be contiguous in element index.
 *
 * @tparam ParallelConfig Configuration of the parallel execution. Must have
 * ParallelConfig::num_threads and ParallelConfig::vector_lanes as static
 * members.
 * @tparam PolicyTraits Traits for the Kokkos::TeamPolicy
 */
template <typename ParallelConfig, typename... PolicyTraits>
struct element_chunk : public Kokkos::TeamPolicy<PolicyTraits...> {
public:
  constexpr static bool isChunkable = true; ///< Chunkable policy
  constexpr static bool isIterable = false; ///< Not iterable
  constexpr static int ChunkSize = ParallelConfig::chunk_size;   ///< Chunk size
  constexpr static int NumThreads = ParallelConfig::num_threads; ///< Chunk size
  constexpr static int VectorLanes =
      ParallelConfig::vector_lanes;                          ///< Vector lanes
  constexpr static int TileSize = ParallelConfig::tile_size; ///< Tile size

  using PolicyType =
      Kokkos::TeamPolicy<PolicyTraits...>; ///< Kokkos::TeamPolicy type

  using member_type =
      typename PolicyType::member_type; ///< Member type of the policy

  using ViewType = Kokkos::View<
      int *, typename member_type::execution_space::memory_space>; ///< View
                                                                   ///< type
                                                                   ///< used to
                                                                   ///< store
                                                                   ///< elements

  /**
   * @brief Construct a new element chunk policy
   *
   * @param view View of elements to chunk
   */
  element_chunk(const ViewType &view)
      : PolicyType(view.extent(0) / TileSize + (view.extent(0) % TileSize != 0),
                   NumThreads, VectorLanes),
        elements(view) {
    static_assert(ViewType::Rank == 1, "View must be rank 1");
  }

  /**
   * @brief Get the policy object
   *
   * @return PolicyType&  Reference to the policy
   */
  inline PolicyType &get_policy() { return *this; }

  /**
   * @brief Get the chunk of elements for the team
   *
   * @param team_rank Rank of the team, usually referenced by member.team_rank()
   * @return auto  Kokkos::subview of the elements for the team
   */
  KOKKOS_INLINE_FUNCTION
  auto league_chunk(const int start_index) const {
    const int start = start_index;
    const int end = (start + ChunkSize > elements.extent(0))
                        ? elements.extent(0)
                        : start + ChunkSize;
    return Kokkos::subview(elements, Kokkos::make_pair(start, end));
  }

private:
  ViewType elements; ///< View of elements
};
} // namespace policy
} // namespace specfem
