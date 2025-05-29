#pragma once

#include "policy.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace policy {
namespace impl {

// forward declaration
template <std::size_t TileSize, typename SIMD> class RangeTileIterator;

template <typename PolicyIndexType, bool UseSIMD> class RangeIndex {
public:
  using iterator_type = VoidIterator;

  KOKKOS_FORCEINLINE_FUNCTION
  PolicyIndexType get_policy_index() const { return this->index.policy_index; }

  KOKKOS_FORCEINLINE_FUNCTION
  specfem::point::assembly_index<UseSIMD> get_index() const {
    return this->index;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr iterator_type get_iterator() const { return VoidIterator(); };

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

template <std::size_t TileSize, typename SIMD>
class RangeTileIterator : public RangeTilePolicy<TileSize> {
private:
  using base_type = RangeTilePolicy<TileSize>;
  constexpr static bool use_simd = SIMD::using_simd;
  constexpr static std::size_t simd_size = SIMD::size();

  int tile_starting_index;
  int starting_index;
  int range_size; ///< Size of the range for this tile

public:
  using index_type =
      RangeIndex<int, use_simd>; ///< Index type for this iterator
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;

  template <bool U = use_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<U, index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index = tile_starting_index + i * simd_size;

    if (starting_index > range_size) {
      return index_type(true); // Return end iterator if starting index is out
                               // of range
    }

    const int number_elements = (starting_index + simd_size < range_size)
                                    ? simd_size
                                    : range_size - starting_index;

    return index_type(i, starting_index, number_elements);
  }

  template <bool U = use_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!U, index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index =
        tile_starting_index + i * RangeTilePolicy<TileSize>::tile_size;
    return index_type(i, starting_index);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  RangeTileIterator() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  RangeTileIterator(const int &tile_starting_index,
                    const policy_index_type &starting_index,
                    const int &range_size)
      : starting_index(starting_index),
        tile_starting_index(tile_starting_index), range_size(range_size) {}
};

template <std::size_t TileSize, typename SIMD> class RangeTileIndex {
public:
  using iterator_type = RangeTileIterator<TileSize, SIMD>;

  KOKKOS_FORCEINLINE_FUNCTION
  int get_policy_index() const { return this->policy_index; }

  KOKKOS_FORCEINLINE_FUNCTION
  const RangeTileIndex &get_index() const {
    return *this; ///< Returns itself as the index
  }

  KOKKOS_FORCEINLINE_FUNCTION
  iterator_type get_iterator() const {
    return iterator_type(this->tile_starting_index, this->policy_index,
                         this->range_size);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  RangeTileIndex(const int &tile_starting_index, const int &policy_index,
                 const int &range_size)
      : tile_starting_index(tile_starting_index), policy_index(policy_index),
        range_size(range_size) {}

private:
  int tile_starting_index;
  int policy_index;
  int range_size;
};

} // namespace impl

template <typename ParallelConfig>
class RangeIterator : public impl::RangePolicy<ParallelConfig> {
private:
  using base_type = impl::RangePolicy<ParallelConfig>;
  constexpr static auto simd_size = ParallelConfig::simd::size();
  constexpr static auto tile_size = ParallelConfig::tile_size;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type
  using policy_index_type =
      typename base_type::policy_index_type; ///< Policy index type
  using index_type =
      impl::RangeIndex<policy_index_type, ParallelConfig::simd::using_simd>;

  KOKKOS_FORCEINLINE_FUNCTION
  RangeIterator() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  RangeIterator(const policy_index_type range_size)
      : simd_range_size(range_size / simd_size +
                        ((range_size % simd_size) != 0)),
        base_type(0, range_size / simd_size + ((range_size % simd_size) != 0)),
        range_size(range_size) {}

  template <bool UseSIMD = ParallelConfig::simd::using_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<UseSIMD, index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index = i * simd_size;
    const int number_elements = (starting_index + simd_size < range_size)
                                    ? simd_size
                                    : range_size - starting_index;
    return index_type(i, starting_index, number_elements);
  }

  template <bool UseSIMD = ParallelConfig::simd::using_simd>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!UseSIMD, index_type>
  operator()(const policy_index_type &i) const {
    const int starting_index = i;
    return index_type(i, starting_index);
  }

private:
  policy_index_type simd_range_size;
  policy_index_type range_size;
};

} // namespace policy
} // namespace specfem
