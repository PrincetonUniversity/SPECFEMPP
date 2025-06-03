#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace execution {

enum class PolicyType { KokkosPolicy, TilePolicy, VoidPolicy };

template <typename ParallelConfig>
class RangePolicy
    : public Kokkos::RangePolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type =
      Kokkos::RangePolicy<typename ParallelConfig::execution_space>;
  using policy_index_type = typename base_policy_type::index_type;

  using base_policy_type::base_policy_type;
  constexpr static bool is_top_level_policy =
      true; ///< Indicates this is a top-level policy
};

template <typename ParallelConfig>
class TeamPolicy
    : public Kokkos::TeamPolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type =
      Kokkos::TeamPolicy<typename ParallelConfig::execution_space>;
  using policy_index_type = typename base_policy_type::member_type;

  using base_policy_type::base_policy_type;
  constexpr static bool is_top_level_policy =
      true; ///< Indicates this is a top-level policy
};

template <typename TeamMemberType, typename IndexType>
class TeamThreadRangePolicy
    : public decltype(Kokkos::TeamThreadRange(std::declval<TeamMemberType>(),
                                              std::declval<IndexType>())) {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type = decltype(Kokkos::TeamThreadRange(
      std::declval<TeamMemberType>(), std::declval<IndexType>()));
  using policy_index_type = IndexType;

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangePolicy(const TeamMemberType &team, const IndexType &range)
      : base_policy_type(Kokkos::TeamThreadRange(team, range)) {}

  constexpr static bool is_top_level_policy =
      false; ///< Indicates this is not a top-level policy
};

template <std::size_t TileSize> class TeamTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = TeamTilePolicy<TileSize>;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;
};

template <std::size_t TileSize> class RangeTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = RangeTilePolicy<TileSize>;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;

  constexpr static bool is_top_level_policy =
      false; ///< Indicates this is a top-level policy
};

class VoidPolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::VoidPolicy;
  using base_policyType = VoidPolicy;
  using policy_index_type = void; ///< No index type for void policy

  constexpr bool static is_top_level_policy =
      false; ///< Indicates this is not a top-level policy
};

} // namespace execution
} // namespace specfem
