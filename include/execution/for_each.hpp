#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "policy.hpp"

namespace specfem {
namespace execution {

template <typename Iterator, typename ClosureType>
inline void for_each(const std::string &name, const Iterator &iterator,
                     const ClosureType &closure) {

  static_assert(Iterator::policy_type ==
                    specfem::execution::PolicyType::KokkosPolicy,
                "Iterator must be a Kokkos policy type");

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

  Kokkos::parallel_for(
      name, static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    Iterator::policy_type == specfem::execution::PolicyType::KokkosPolicy, void>
for_each(const Iterator &iterator, const ClosureType &closure) {

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

  Kokkos::parallel_for(
      static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    Iterator::policy_type == specfem::execution::PolicyType::TilePolicy, void>
for_each(const Iterator &iterator, const ClosureType &closure) {

  constexpr std::size_t tile_size = Iterator::tile_size;

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the kokkos_index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

#pragma unroll
  for (std::size_t itile = 0; itile < tile_size; ++itile) {
    const typename Iterator::index_type index = iterator(itile);

    // Check if the index is valid
    if (index.is_end()) {
      break; // Skip if the index is at the end
    }

    closure(index);
  }
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    Iterator::policy_type == specfem::execution::PolicyType::VoidPolicy, void>
for_each(const Iterator &iterator, const ClosureType &closure) {
  static_assert(Iterator::policy_type ==
                    specfem::execution::PolicyType::VoidPolicy,
                "Calling for_each on a VoidPolicy is not allowed");
}

} // namespace execution
} // namespace specfem
