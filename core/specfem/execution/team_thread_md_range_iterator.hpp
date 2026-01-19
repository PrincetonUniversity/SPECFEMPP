#pragma once

#include "policy.hpp"
#include "specfem/datatype/impl/register_array.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace execution {
/**
 * @brief Multi-dimensional range iterator for team-based parallel execution
 *
 * This class provides a way to iterate over multi-dimensional ranges in
 * parallel using Kokkos team-based execution. It supports both 2D and 3D
 * iterations and handles different execution spaces (host and device)
 * appropriately.
 *
 * @tparam TeamMemberType The Kokkos team member type
 * @tparam Rank The dimensionality of the iterator (2 or 3)
 */
template <typename TeamMemberType, int Rank>
class TeamThreadMDRangeIterator
    : public TeamThreadRangePolicy<TeamMemberType, int> {

private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;
  int extents_[Rank]; ///< Extents of each dimension

public:
  constexpr static auto rank = Rank; ///< Rank of the iterator (2 or 3)
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using index_type = specfem::datatype::impl::RegisterArray<
      int, Kokkos::extents<std::size_t, rank>,
      Kokkos::layout_left>; ///< Underlying index type. This
                            ///< index will be passed to the
                            ///< closure when calling @ref
                            ///< kokkos::parallel_for with
                            ///< this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  /**
   * @brief Convert linear index to 2D / 3D coordinates
   *
   * @param i Linear index
   * @return index_type 2D / 3D coordinates
   */
  template <
      typename T = TeamMemberType,
      std::enable_if_t<std::is_same_v<typename T::execution_space::memory_space,
                                      Kokkos::HostSpace>,
                       int> = 0>
  KOKKOS_INLINE_FUNCTION const index_type operator()(const int &i) const {
    index_type result;
    result(0) = i % extents_[0];
    int temp = i / extents_[0];

    for (int r = 1; r < rank - 1; ++r) {
      result(r) = temp % extents_[r];
      temp /= extents_[r];
    }

    if constexpr (rank > 1) {
      result(rank - 1) = temp;
    }
    return result;
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

private:
#ifdef KOKKOS_ENABLE_CUDA
  using device_memory_space = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
  using device_memory_space = Kokkos::HIPSpace;
#endif
public:
  template <
      typename T = TeamMemberType,
      std::enable_if_t<std::is_same_v<typename T::execution_space::memory_space,
                                      device_memory_space>,
                       int> = 0>
  KOKKOS_INLINE_FUNCTION const index_type operator()(const int &i) const {
    index_type result;
    result(rank - 1) = i % extents_[rank - 1];
    int temp = i / extents_[rank - 1];

    for (int r = rank - 2; r > 0; --r) {
      result(r) = temp % extents_[r];
      temp /= extents_[r];
    }

    if constexpr (rank > 1) {
      result(0) = temp;
    }
    return result;
  }
#endif

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION constexpr
  TeamThreadMDRangeIterator(const TeamMemberType &team, Indices... indices)
      : base_type(team,
                  [&]() {
                    size_t product = 1;
                    ((product *= indices), ...);
                    return product;
                  }()),
        extents_{ indices... } {}
};

// User-defined deduction guide
template <typename TeamMemberType, typename... Indices>
TeamThreadMDRangeIterator(const TeamMemberType &, Indices...)
    -> TeamThreadMDRangeIterator<TeamMemberType, sizeof...(Indices)>;

} // namespace execution
} // namespace specfem
