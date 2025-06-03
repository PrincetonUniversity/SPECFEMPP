#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "policy.hpp"

namespace specfem {
namespace execution {

namespace impl {

template <typename Iterator, typename ClosureType>
constexpr void check_compatibility() {
  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<
          const typename Iterator::index_type,
          decltype(std::declval<Iterator>()(
              std::declval<typename Iterator::policy_index_type>()))>,
      "Iterator must have a function call operator that returns index_type");
}
} // namespace impl

template <typename Iterator, typename ClosureType>
constexpr inline std::enable_if_t<
    ((Iterator::is_top_level_policy) &&
     (Iterator::policy_type == specfem::execution::PolicyType::KokkosPolicy)),
    void>
for_each_level(const std::string &name, const Iterator &iterator,
               const ClosureType &closure) {

  impl::check_compatibility<Iterator, ClosureType>();

  Kokkos::parallel_for(
      name, static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
constexpr inline std::enable_if_t<
    ((Iterator::is_top_level_policy) &&
     (Iterator::policy_type == specfem::execution::PolicyType::KokkosPolicy)),
    void>
for_each_level(const Iterator &iterator, const ClosureType &closure) {

  impl::check_compatibility<Iterator, ClosureType>();

  Kokkos::parallel_for(
      static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
constexpr KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    ((!Iterator::is_top_level_policy) &&
     (Iterator::policy_type == specfem::execution::PolicyType::KokkosPolicy)),
    void>
for_each_level(const Iterator &iterator, const ClosureType &closure) {

  impl::check_compatibility<Iterator, ClosureType>();

  Kokkos::parallel_for(
      static_cast<const typename Iterator::base_policy_type &>(iterator),
      [&](const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
constexpr KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    Iterator::policy_type == specfem::execution::PolicyType::TilePolicy, void>
for_each_level(const Iterator &iterator, const ClosureType &closure) {

  constexpr std::size_t tile_size = Iterator::tile_size;

  impl::check_compatibility<Iterator, ClosureType>();

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
constexpr KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    Iterator::policy_type == specfem::execution::PolicyType::VoidPolicy, void>
for_each_level(const Iterator &iterator, const ClosureType &closure) {
  static_assert(Iterator::policy_type ==
                    specfem::execution::PolicyType::VoidPolicy,
                "Calling for_each on a VoidPolicy is not allowed");
}

} // namespace execution
} // namespace specfem
