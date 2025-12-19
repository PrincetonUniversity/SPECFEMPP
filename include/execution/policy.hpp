#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace execution {

enum class PolicyType {
  KokkosPolicy, ///< Indicates a Kokkos policy
  TilePolicy,   ///< Indicates a tile policy
  VoidPolicy    ///< Indicates a void policy
};

/**
 * @brief Base class for all iterators using on @c Kokkos::RangePolicy.
 *
 * @tparam ParallelConfig Configuration for parallel execution. @ref
 * specfem::parallel_configurationuration::range_config
 */
template <typename ParallelConfig>
class RangePolicy
    : public Kokkos::RangePolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type =
      PolicyType::KokkosPolicy; ///< Indicates this is a Kokkos policy
  using base_policy_type = Kokkos::RangePolicy<
      typename ParallelConfig::execution_space>; ///< Base policy type.
                                                 ///< Evaluates to @c
                                                 ///< Kokkos::RangePolicy
  using policy_index_type =
      typename base_policy_type::index_type; ///< Policy index type. Must be
  ///< convertible to integral type.
  using execution_space =
      typename base_policy_type::execution_space; ///< Execution space type.

  using base_policy_type::base_policy_type;
  constexpr static bool is_top_level_policy =
      true; ///< Indicates this is a top-level policy
};

/**
 * @brief Base class for all iterators using on @c Kokkos::TeamPolicy.
 *
 * @tparam ParallelConfig Configuration for parallel execution. @ref
 * specfem::parallel_configurationuration::team_config
 */
template <typename ParallelConfig>
class TeamPolicy
    : public Kokkos::TeamPolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type =
      PolicyType::KokkosPolicy; ///< Indicates this is a Kokkos policy
  using base_policy_type = Kokkos::TeamPolicy<
      typename ParallelConfig::execution_space>; ///< Base policy type.
                                                 ///< Evaluates to @c
                                                 ///< Kokkos::TeamPolicy
  using policy_index_type =
      typename base_policy_type::member_type; ///< Policy index type.
  using execution_space =
      typename base_policy_type::execution_space; ///< Execution space type.

  using base_policy_type::base_policy_type;
  constexpr static bool is_top_level_policy =
      true; ///< Indicates this is a top-level policy
};

/**
 * @brief TeamThreadRangePolicy class is used to iterate over a range of indices
 * within a Kokkos team.
 *
 * This policy is used to iterate over a range of indices in a team parallel
 * execution context.
 *
 * @tparam TeamMemberType The type of the Kokkos team member.
 * @tparam IndexType The type of the index to iterate over.
 */
template <typename TeamMemberType, typename IndexType>
class TeamThreadRangePolicy
    : public decltype(Kokkos::TeamThreadRange(std::declval<TeamMemberType>(),
                                              std::declval<IndexType>())) {
public:
  constexpr static PolicyType policy_type =
      PolicyType::KokkosPolicy; ///< Indicates this is a Kokkos policy
  using base_policy_type = decltype(Kokkos::TeamThreadRange(
      std::declval<TeamMemberType>(), std::declval<IndexType>()));
  using policy_index_type =
      IndexType; ///< Policy index type. Must be convertible to integral type.
  using execution_space =
      typename TeamMemberType::execution_space; ///< Execution space type.

  /**
   * @brief Constructs a TeamThreadRangePolicy for a given team member and
   * range.
   *
   * @param team The Kokkos team member.
   * @param range The range of indices to iterate over.
   */
  KOKKOS_INLINE_FUNCTION
  TeamThreadRangePolicy(const TeamMemberType &team, const IndexType &range)
      : base_policy_type(Kokkos::TeamThreadRange(team, range)) {}

  constexpr static bool is_top_level_policy =
      false; ///< Indicates this is not a top-level policy
};

template <std::size_t TileSize, typename ExecutionSpace> class TeamTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = TeamTilePolicy;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;
  using execution_space = ExecutionSpace;
  constexpr static bool is_top_level_policy =
      false; ///< Indicates this is not a top-level policy
};

template <std::size_t TileSize, typename ExecutionSpace> class RangeTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = RangeTilePolicy;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;

  constexpr static bool is_top_level_policy =
      false; ///< Indicates this is a top-level policy

  using execution_space =
      ExecutionSpace; ///< Execution space type for the tile policy
};

/**
 * @brief VoidPolicy class is used when there is no valid index to iterate over.
 *
 * This policy is used as a placeholder when no iteration is needed.
 */
template <typename ExecutionSpace> class VoidPolicy {
public:
  constexpr static PolicyType policy_type =
      PolicyType::VoidPolicy;          ///< Indicates this is a void policy
  using base_policy_type = VoidPolicy; ///< Base policy type, which is itself
  using policy_index_type = void;      ///< No index type for void policy
  using execution_space =
      ExecutionSpace; ///< Execution space type for the void policy

  constexpr bool static is_top_level_policy =
      false; ///< Indicates this is not a top-level policy
};

} // namespace execution
} // namespace specfem
