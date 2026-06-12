#pragma once
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Lightweight contiguous range of element indices [start, end).
 *
 * Zero-allocation container for element indices guaranteed to be contiguous.
 * Provides a Kokkos::View-compatible interface (operator(), extent()) for
 * transparent drop-in replacement at call sites. Device-safe.
 *
 * @code
 * ElementIndexRange range(0, 100);
 * for (int i = 0; i < range.extent(0); ++i) {
 *   int elem_id = range(i);  // elem_id = 0, 1, ..., 99
 * }
 * @endcode
 *
 * @ingroup DataTypes
 */
class ElementIndexRange {
public:
  using value_type = int;

  /**
   * @brief Default constructor: empty range [0, 0).
   */
  KOKKOS_DEFAULTED_FUNCTION ElementIndexRange() = default;

  /**
   * @brief Create range [start, end).
   *
   * @param start First element index (inclusive)
   * @param end One past last element index (exclusive)
   */
  KOKKOS_INLINE_FUNCTION ElementIndexRange(int start, int end)
      : start_(start), end_(end) {}

  // ── View-compatible interface ──────────────────────────────────────────────

  /**
   * @brief Get element index at offset i (i.e., start + i).
   *
   * Provides Kokkos::View-compatible access for transparent drop-in
   * replacement.
   *
   * @param i Offset within range [0, size())
   * @return Element index at position i
   */
  KOKKOS_INLINE_FUNCTION int operator()(int i) const { return start_ + i; }

  /**
   * @brief Get range extent (number of elements).
   *
   * Returns the same value regardless of rank parameter for View compatibility.
   *
   * @param Rank parameter (unused)
   * @return Number of elements in range
   */
  KOKKOS_INLINE_FUNCTION int extent(int) const { return end_ - start_; }

  // ── Size / emptiness ───────────────────────────────────────────────────────

  /**
   * @brief Number of elements in range.
   */
  KOKKOS_INLINE_FUNCTION int size() const { return end_ - start_; }

  /**
   * @brief Check if range is empty (start == end).
   */
  KOKKOS_INLINE_FUNCTION bool empty() const { return end_ == start_; }

  // ── Range-specific access ─────────────────────────────────────────────────

  /**
   * @brief First element index (inclusive).
   */
  KOKKOS_INLINE_FUNCTION int begin_index() const { return start_; }

  /**
   * @brief One past the last element index (exclusive).
   */
  KOKKOS_INLINE_FUNCTION int end_index() const { return end_; }

private:
  int start_{ 0 };
  int end_{ 0 };
};

/**
 * @brief Slice a range using Kokkos::subview semantics.
 *
 * Returns the sub-range [range.begin_index() + slice.first,
 * range.begin_index() + slice.second). Found by ADL, enabling transparent
 * use in generic code that calls subview() without namespace prefix.
 *
 * @param range Source range
 * @param slice Pair of offsets (first, second)
 * @return New range covering slice within original
 *
 * @code
 * ElementIndexRange orig(100, 200);
 * auto sub = subview(orig, Kokkos::pair<int,int>(10, 20));
 * // sub.begin_index() == 110, sub.end_index() == 120
 * @endcode
 *
 * @ingroup DataTypes
 */
KOKKOS_INLINE_FUNCTION ElementIndexRange subview(const ElementIndexRange &range,
                                                 Kokkos::pair<int, int> slice) {
  return { range.begin_index() + slice.first,
           range.begin_index() + slice.second };
}

} // namespace datatype
} // namespace specfem
