#pragma once

#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <type_traits>

namespace specfem {
namespace execution {

template <typename PolicyIndexType, bool UseSIMD> class RangeIndex {
public:
  using iterator_type = VoidIterator;

  KOKKOS_FORCEINLINE_FUNCTION
  const PolicyIndexType get_policy_index() const {
    return this->index.policy_index;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const specfem::point::assembly_index<UseSIMD> get_index() const {
    return this->index;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return VoidIterator{}; }

  template <bool U = UseSIMD, typename std::enable_if<!U, int>::type = 0>
  KOKKOS_FORCEINLINE_FUNCTION RangeIndex(const PolicyIndexType i,
                                         const int &starting_index)
      : policy_index(i), index(starting_index) {}

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

template <typename ParallelConfig>
class RangeIterator : public RangePolicy<ParallelConfig> {
private:
  using base_type = RangePolicy<ParallelConfig>;
  constexpr static auto simd_size = ParallelConfig::simd::size();
  constexpr static auto tile_size = ParallelConfig::tile_size;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type
  using policy_index_type =
      typename base_type::policy_index_type; ///< Policy index type
  using index_type =
      RangeIndex<policy_index_type, ParallelConfig::simd::using_simd>;

  RangeIterator() = default;

  RangeIterator(const policy_index_type range_size)
      : simd_range_size(range_size / simd_size +
                        ((range_size % simd_size) != 0)),
        base_type(0, range_size / simd_size + ((range_size % simd_size) != 0)),
        range_size(range_size) {}

  RangeIterator(const ParallelConfig, const policy_index_type range_size)
      : RangeIterator(range_size) {}

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
  policy_index_type simd_range_size;
  policy_index_type range_size;
};

} // namespace execution
} // namespace specfem
