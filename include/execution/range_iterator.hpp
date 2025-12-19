#pragma once

#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <type_traits>

namespace specfem {
namespace execution {

/**
 * @brief RangeIndex is used to represent an index for GLL points within
 * a FE assembly.
 *
 * @tparam PolicyIndexType Type of the policy index, must be convertible to an
 * integral type.
 * @tparam UseSIMD Indicates whether SIMD is used for the index.
 */
template <typename PolicyIndexType, bool UseSIMD, typename ExecutionSpace>
class RangeIndex {
public:
  using iterator_type =
      VoidIterator<ExecutionSpace>; ///< Iterator used to iterate over GLL
                                    ///< points within this index. @c
                                    ///< VoidIterator is used when the index
                                    ///< refers to a single GLL point.

  /**
   * @brief Get the policy index that defined this range index. See @ref
   * specfem::execution::RangeIterator::operator()
   *
   * @return cosnt PolicyIndexType The policy index that defined this range
   * index.
   */
  KOKKOS_FORCEINLINE_FUNCTION
  const PolicyIndexType get_policy_index() const {
    return this->index.policy_index;
  }

  /**
   * @brief Get the underlying index used to define the GLL point.
   *
   * @return const specfem::point::assembly_index<UseSIMD> The assembly index
   * that defines the GLL point.
   */
  KOKKOS_FORCEINLINE_FUNCTION
  const specfem::point::assembly_index<UseSIMD> get_index() const {
    return this->index;
  }

  /**
   * @brief Get the iterator for this index.
   *
   * @return const iterator_type The iterator for this index.
   */
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return iterator_type{}; }

  /**
   * @brief Constructor for RangeIndex when SIMD is not used.
   *
   * @param i The policy index which created this range index.
   * @param index The index of the GLL point within FE assembly.
   */
  template <bool U = UseSIMD, typename std::enable_if<!U, int>::type = 0>
  KOKKOS_FORCEINLINE_FUNCTION RangeIndex(const PolicyIndexType i,
                                         const int &index)
      : policy_index(i), index(index) {}

  /**
   * @brief Constructor for RangeIndex when SIMD is used.
   *
   * @param i The policy index which created this range index.
   * @param starting_index The starting index of the GLL point within this SIMD
   * vector.
   * @param number_points The number of GLL points in this SIMD vector.
   * @return const RangeIndex The constructed RangeIndex object.
   */
  template <bool U = UseSIMD, typename std::enable_if<U, int>::type = 0>
  KOKKOS_FORCEINLINE_FUNCTION RangeIndex(const PolicyIndexType i,
                                         const int &starting_index,
                                         const int &number_points)
      : policy_index(i), index(starting_index, number_points) {}

  template <bool U = UseSIMD, typename std::enable_if<U, int>::type = 0>
  KOKKOS_FORCEINLINE_FUNCTION RangeIndex(const bool end_iterator)
      : end_iterator(end_iterator) {}

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr bool is_end() const {
    return end_iterator; ///< Returns true if the iterator is at the end
  }

private:
  specfem::point::assembly_index<UseSIMD> index; ///< Assembly index
  PolicyIndexType policy_index;
  bool end_iterator = false; ///< Indicates if the iterator is at the end
};

/**
 * @brief RangeIterator class is used to iterate over a range of assembled
 * quadrature points within an FE assembly.
 *
 * This iterator iterates over a range of GLL points between [0, RangeSize).
 *
 * @tparam ParallelConfig Configuration for parallel execution. @ref
 * specfem::parallel_configurationuration::range_config
 *
 */
template <typename ParallelConfig>
class RangeIterator : public RangePolicy<ParallelConfig> {
private:
  using base_type = RangePolicy<ParallelConfig>; ///< Base policy type
  constexpr static auto simd_size = ParallelConfig::simd::size(); ///< SIMD size
  constexpr static auto tile_size = ParallelConfig::tile_size;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::RangePolicy`
  using policy_index_type = typename base_type::
      policy_index_type; ///< Policy index type. Must be
                         ///< convertible to intergral type.
                         ///< Evaluates to @c Kokkos::RangePolicy::index_type
  using index_type =
      RangeIndex<policy_index_type, ParallelConfig::simd::using_simd,
                 typename base_type::
                     execution_space>; ///< Underlying index type. This index
                                       ///< will be passed to the closure when
                                       ///< calling @ref
                                       ///< specfem::execution::for_each_level

  using base_index_type = RangeIndex<
      typename base_type::index_type, ParallelConfig::simd::using_simd,
      typename base_type::execution_space>; ///< Index type to be
                                            ///< used when calling
                                            ///< @ref
                                            ///< specfem::execution::for_all
                                            ///< with this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  RangeIterator() = default;

  /**
   * @brief Constructs a RangeIterator for a given range size.
   *
   * @param range_size The size of the range to iterate over.
   */
  RangeIterator(const policy_index_type range_size)
      : simd_range_size(range_size / simd_size +
                        ((range_size % simd_size) != 0)),
        base_type(0, range_size / simd_size + ((range_size % simd_size) != 0)),
        range_size(range_size) {}

  /**
   * @brief Construct a new Range Iterator object for a given parallel
   * configuration
   *
   * @param range_size The size of the range to iterate over.
   */
  RangeIterator(const ParallelConfig, const policy_index_type range_size)
      : RangeIterator(range_size) {}

  /**
   * @brief Compute the index for a given policy index.
   *
   * @param i The policy index for which to compute the range index.
   * @return const index_type The computed index for the given policy index.
   */
  template <bool UseSIMD = ParallelConfig::simd::using_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<UseSIMD, const index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index = i * simd_size;
    const int number_points = (starting_index + simd_size < range_size)
                                  ? simd_size
                                  : range_size - starting_index;
    return index_type(i, starting_index, number_points);
  }

  template <bool UseSIMD = ParallelConfig::simd::using_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!UseSIMD, const index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index = i;
    return index_type(i, starting_index);
  }

private:
  policy_index_type simd_range_size; ///< Range size adjusted for SIMD length
  policy_index_type range_size;      ///< Original range size
};

} // namespace execution
} // namespace specfem
