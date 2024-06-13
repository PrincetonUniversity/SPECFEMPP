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
      ParallelConfig::vector_lanes; ///< Vector lanes

  using PolicyType =
      Kokkos::TeamPolicy<PolicyTraits...>; ///< Kokkos::TeamPolicy type

  using member_type =
      typename PolicyType::member_type; ///< Team handle type. See
                                        ///< Kokkos::TeamHandleConcept

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
      : elements(view),
        policy(view.extent(0) / ChunkSize + (view.extent(0) % ChunkSize != 0),
               NumThreads, VectorLanes) {
    static_assert(ViewType::Rank == 1, "View must be rank 1");
  }

  /**
   * @brief Set the Chunk Size for the team
   *
   * Set the chunk size. Each physical team of threads will get assigned chunk
   * consecutive teams. Default is 1.
   * This is different from element_chunk::ChunkSize, which sets how many
   * elements a team will work on.
   * @param chunk_size Chunk size
   * @return PolicyType& Reference to the policy
   */
  inline PolicyType &set_chunk_size(const int chunk_size) {
    return policy.set_chunk_size(chunk_size);
  }

  /**
   * @brief Set the scratch size object
   *
   * Forwards the arguments to the policy's set_scratch_size method. Check
   * Kokkos::TeamPolicy::set_scratch_size for more information.
   *
   * @tparam Args Args for the scratch size
   * @param args Args for the scratch size
   * @return PolicyType&  Reference to the policy
   */
  template <typename... Args>
  inline PolicyType &set_scratch_size(Args... args) {
    return policy.set_scratch_size(args...);
  }

  /**
   * @brief Get the policy object
   *
   * @return PolicyType&  Reference to the policy
   */
  inline PolicyType &get_policy() { return policy; }

  /**
   * @brief Get the chunk of elements for the team
   *
   * @param team_rank Rank of the team, usually referenced by member.team_rank()
   * @return auto  Kokkos::subview of the elements for the team
   */
  KOKKOS_INLINE_FUNCTION
  auto league_chunk(const int team_rank) const {
    const int start = team_rank * ChunkSize;
    const int end = (team_rank + 1) * ChunkSize < elements.extent(0)
                        ? (team_rank + 1) * ChunkSize
                        : elements.extent(0);
    return Kokkos::subview(elements, Kokkos::make_pair(start, end));
  }

private:
  ViewType elements; ///< View of elements
  PolicyType policy; ///< Kokkos::TeamPolicy
};
} // namespace policy
} // namespace specfem
