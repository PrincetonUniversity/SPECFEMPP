#pragma once

#include <Kokkos_Core.hpp>

namespace specfem::datatype {

/**
 * @brief Fixed-size LRU (Least Recently Used) cache for use in Kokkos kernels
 *
 * Stores up to CacheSize entries keyed by Index. On a cache miss during put(),
 * the least recently used entry is evicted. All operations are O(CacheSize).
 *
 * @tparam CacheSize Maximum number of entries in the cache
 * @tparam Index Key type (must support operator==)
 * @tparam T Value type
 */
template <int CacheSize, typename Index, typename T> class LRU_cache;

/**
 * @brief Specialization of LRU_cache for a single entry
 *
 * Uses a linearized int tag for O(1) comparison instead of comparing the full
 * Index struct. Forces inlining of all methods to eliminate function call
 * overhead in hot kernel loops.
 *
 * @tparam Index Key type (must have ispec, iz, ix int members)
 * @tparam T Value type
 */
template <typename Index, typename T> class LRU_cache<1, Index, T> {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  LRU_cache() : tag_(empty_tag) {}

  KOKKOS_FORCEINLINE_FUNCTION
  LRU_cache(const LRU_cache &other) : tag_(other.tag_), value_(other.value_) {}

  KOKKOS_FORCEINLINE_FUNCTION
  bool get(const Index &index, T &value) const {
    if (tag_ == index.linearize()) {
      value = value_;
      return true;
    }
    return false;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void put(const Index &index, const T &value) const {
    tag_ = index.linearize();
    value_ = value;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool contains(const Index &index) const { return tag_ == index.linearize(); }

  KOKKOS_FORCEINLINE_FUNCTION
  void clear() { tag_ = empty_tag; }

  KOKKOS_FORCEINLINE_FUNCTION
  int size() const { return tag_ != empty_tag ? 1 : 0; }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr int capacity() const { return 1; }

private:
  static constexpr unsigned int empty_tag = 0xFFFFFFFFu; ///< Sentinel value
  mutable unsigned int tag_; ///< Linearized index tag (empty_tag = empty)
  mutable T value_;          ///< Cached value
};

} // namespace specfem::datatype
