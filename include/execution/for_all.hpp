#pragma once

#include "for_each_level.hpp"

namespace specfem {
namespace execution {

template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<IndexType::iterator_type::policy_type ==
                         specfem::execution::PolicyType::VoidPolicy,
                     void>
    for_all(const IndexType &index, const ClosureType &closure) {
  const auto i = index.get_index();
  closure(i);
}

template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<((!IndexType::iterator_type::is_top_level_policy) &&
                      (IndexType::iterator_type::policy_type !=
                       specfem::execution::PolicyType::VoidPolicy)),
                     void>
    for_all(const IndexType &index, const ClosureType &closure) {

  for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iter_index) {
        for_all(iter_index, closure);
      });
}

template <typename IndexType, typename ClosureType>
inline std::enable_if_t<((IndexType::iterator_type::is_top_level_policy) &&
                         (IndexType::iterator_type::policy_type !=
                          specfem::execution::PolicyType::VoidPolicy)),
                        void>
for_all(const IndexType &index, const ClosureType &closure) {

  for_each_level(
      index.get_iterator(),
      KOKKOS_LAMBDA(
          const typename IndexType::iterator_type::index_type &iter_index) {
        for_all(iter_index, closure);
      });
}

template <typename Iterator, typename ClosureType>
inline std::enable_if_t<((Iterator::is_top_level_policy) &&
                         (Iterator::policy_type ==
                          specfem::execution::PolicyType::KokkosPolicy)),
                        void>
for_all(const std::string &name, const Iterator &iterator,
        const ClosureType &closure) {

  for_each_level(
      name, iterator,
      KOKKOS_LAMBDA(const typename Iterator::index_type &iter_index) {
        for_all(iter_index, closure);
      });
}

} // namespace execution
} // namespace specfem
