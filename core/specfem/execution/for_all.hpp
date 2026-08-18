#pragma once

#include "for_each_level.hpp"
#include <type_traits>

namespace specfem {
namespace execution {

namespace impl {

/// Base level call when running on Host execution space
template <typename IndexType, typename ClosureType>
std::enable_if_t<
    ((IndexType::iterator_type::policy_type ==
      specfem::execution::PolicyType::VoidPolicy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultHostExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {
  closure(index);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

/// Base level call when running on Device execution space
template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    ((IndexType::iterator_type::policy_type ==
      specfem::execution::PolicyType::VoidPolicy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {
  closure(index);
}

#endif

/// Recursive call for iterators that are not top-level policies
/// This could be Team ThreadRangePolicy or a Tile Policy
/// Call runs on Host execution space
template <typename IndexType, typename ClosureType>
inline std::enable_if_t<
    ((!IndexType::iterator_type::is_top_level_policy) &&
     (IndexType::iterator_type::policy_type !=
      specfem::execution::PolicyType::VoidPolicy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultHostExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {
  for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iter_index) {
        impl_for_all(iter_index, closure);
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
/// Recursive call for iterators that are not top-level policies
/// This could be Team ThreadRangePolicy or a Tile Policy
/// Call runs on a Device execution space
template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
    ((!IndexType::iterator_type::is_top_level_policy) &&
     (IndexType::iterator_type::policy_type !=
      specfem::execution::PolicyType::VoidPolicy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {

  for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iter_index) {
        impl_for_all(iter_index, closure);
      });
}

#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

/// Recursive call for top-level policies that are based on Kokkos policies
/// This could be a Team policy, Range policy, MDRange policy, etc.
/// Call runs on a Device execution space
template <typename IndexType, typename ClosureType>
inline std::enable_if_t<
    ((IndexType::iterator_type::is_top_level_policy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {

  for_each_level(
      index.get_iterator(),
      KOKKOS_LAMBDA(
          const typename IndexType::iterator_type::index_type &iter_index) {
        impl_for_all(iter_index, closure);
      });
}

#endif

/// Recursive call for top-level policies that are based on Kokkos policies
/// This could be a Team policy, Range policy, MDRange policy, etc.
/// Call runs on Host execution space
template <typename IndexType, typename ClosureType>
inline std::enable_if_t<
    ((IndexType::iterator_type::is_top_level_policy) &&
     (IndexType::iterator_type::policy_type !=
      specfem::execution::PolicyType::VoidPolicy) &&
     (std::is_same_v<typename IndexType::iterator_type::execution_space,
                     Kokkos::DefaultHostExecutionSpace>)),
    void>
impl_for_all(const IndexType &index, const ClosureType &closure) {

  for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iter_index) {
        impl_for_all(iter_index, closure);
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

/// Recursive call for top-level policies that are based on Kokkos policies
/// This could be a Team policy, Range policy, MDRange policy, etc.
/// Call runs on a Device execution space
template <typename Iterator, typename ClosureType>
inline std::enable_if_t<((Iterator::is_top_level_policy) &&
                         (std::is_same_v<typename Iterator::execution_space,
                                         Kokkos::DefaultExecutionSpace>)),
                        void>
impl_for_all(const std::string &name, const Iterator &iterator,
             const ClosureType &closure) {

  for_each_level(
      name, iterator,
      KOKKOS_LAMBDA(const typename Iterator::index_type &iter_index) {
        impl_for_all(iter_index, closure);
      });
}

#endif

/// Recursive call for top-level policies that are based on Kokkos policies
/// This could be a Team policy, Range policy, MDRange policy, etc.
/// Call runs on Host execution space
template <typename Iterator, typename ClosureType>
inline std::enable_if_t<((Iterator::is_top_level_policy) &&
                         (std::is_same_v<typename Iterator::execution_space,
                                         Kokkos::DefaultHostExecutionSpace>)),
                        void>
impl_for_all(const std::string &name, const Iterator &iterator,
             const ClosureType &closure) {

  for_each_level(name, iterator,
                 [&](const typename Iterator::index_type &iter_index) {
                   impl_for_all(iter_index, closure);
                 });
}

} // namespace impl

/**
 * @brief Visit every GLL point within a given index range.
 *
 * This function applies a closure to each GLL point in the specified index
 * range.
 * The closure is passed onto the underlying Kokkos policy for the iterator.
 *
 * @param iterator iterator type that defines the range of indices to iterate
 * over. The iterator must be a top-level policy based on a Kokkos policy.
 * @param closure a callable object that will be invoked for each index. The
 * closure must be callable with a single argument, which is the index of the
 * GLL point.
 */
template <
    typename IteratorType, typename ClosureType,
    typename std::enable_if<IteratorType::is_top_level_policy, int>::type = 0>
inline void for_all(const IteratorType &iterator, const ClosureType &closure) {
  impl::impl_for_all(index, closure);
}

/**
 * @brief Visit every GLL point within a given index range.
 *
 * This function applies a closure to each GLL point in the specified index
 * range.
 *
 * The closure is passed onto the underlying Kokkos policy for the iterator.
 *
 * @param name a string identifier for the operation.
 * @param iterator iterator type that defines the range of indices to iterate
 * over. The iterator must be a top-level policy based on a Kokkos policy.
 * @param closure a callable object that will be invoked for each index. The
 * closure must be callable with a single argument, which is the index of the
 * GLL point.
 */
template <
    typename IteratorType, typename ClosureType,
    typename std::enable_if<IteratorType::is_top_level_policy, int>::type = 0>
inline void for_all(const std::string &name, const IteratorType &iterator,
                    const ClosureType &closure) {
  impl::impl_for_all(name, iterator, closure);
}

} // namespace execution
} // namespace specfem
