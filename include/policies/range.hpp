#pragma once

#include "point/assembly_index.hpp"
#include <type_traits>

namespace specfem {

namespace iterator {
namespace impl {
/**
 * @brief Index type for the range iterator.
 *
 * @tparam UseSIMD Indicates whether SIMD is used or not.
 */
template <bool UseSIMD> struct range_index_type;

/**
 * @brief Template specialization when not using SIMD.
 *
 */
template <> struct range_index_type<false> {
  specfem::point::assembly_index<false> index; ///< Assembly index

  KOKKOS_INLINE_FUNCTION
  range_index_type(const specfem::point::assembly_index<false> index)
      : index(index) {}
};

/**
 * @brief Template specialization when using SIMD.
 *
 */
template <> struct range_index_type<true> {
  specfem::point::simd_assembly_index index; ///< SIMD assembly index

  KOKKOS_INLINE_FUNCTION
  range_index_type(const specfem::point::simd_assembly_index index)
      : index(index) {}
};

} // namespace impl

/**
 * @brief Iterator to generate indices for quadrature points defined within this
 * iterator.
 *
 * @note Current implementation supports only one index within this iterator
 * when not using SIMD, and support n indices where n <=
 * native_simd<type_real>::size() when using SIMD.
 *
 * @note Range iterator is accessed through the @ref specfem::policy::range
 *
 * @tparam SIMD type to generate a SIMD index.
 */
template <typename SIMD> struct range {

private:
  int starting_index; ///< Starting index for the iterator range.
  int number_points;  ///< Number of points in the iterator range. Equal to or
                      ///< less than SIMD size when using SIMD.

  constexpr static bool using_simd = SIMD::using_simd;
  constexpr static int simd_size = SIMD::size();

  // --- SIMD
  // Range constructor for simd execution
  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points, std::true_type)
      : starting_index(starting_index),
        number_points((number_points < simd_size) ? number_points : simd_size) {
  }

  // range_index_type operator for simd execution
  KOKKOS_INLINE_FUNCTION
  impl::range_index_type<true> operator()(const int i, std::true_type) const {
    return impl::range_index_type<true>(
        specfem::point::simd_assembly_index{ starting_index, number_points });
  }

  // --- NON-SIMD
  // Range constructor for non-simd execution
  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points, std::false_type)
      : starting_index(starting_index), number_points(number_points) {}

  KOKKOS_INLINE_FUNCTION
  impl::range_index_type<false> operator()(const int i, std::false_type) const {
    return impl::range_index_type<false>(
        specfem::point::assembly_index<false>{ starting_index });
  }

public:
  /**
   * @name Type definitions
   *
   */
  ///@{
  using index_type = impl::range_index_type<using_simd>; ///< Index type
  using simd = SIMD;                                     ///< SIMD type
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor.
   *
   */
  KOKKOS_INLINE_FUNCTION
  range() = default;

  /**
   * @brief Construct a range iterator with a given starting index and number of
   * points.
   *
   * @param starting_index Starting index for the iterator range.
   * @param number_points Number of points in the iterator range.
   */
  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points)
      : range(starting_index, number_points,
              std::integral_constant<bool, using_simd>()) {}

  /**
   * @brief Returns the index within this iterator at the i-th location.
   *
   * @param i Location within the iterator. Ignored in the current
   * implementation since iterator only holds a single index when not using SIMD
   * or <= SIMD size number of indices when using SIMD.
   * @return index_type Index for the given iterator at the i-th location.
   */
  KOKKOS_INLINE_FUNCTION
  index_type operator()(const int i) const {
    return operator()(i, std::integral_constant<bool, using_simd>());
  }
};
} // namespace iterator

namespace policy {

/**
 * @brief Range policy to iterate over a range of quadrature points.
 *
 * @tparam ParallelConfig Parallel configuration for range policy. @ref
 * specfem::parallel_config::range_config
 */
template <typename ParallelConfig>
struct range : Kokkos::RangePolicy<typename ParallelConfig::execution_space> {
public:
  static_assert(ParallelConfig::is_point_parallel_config,
                "Wrong parallel config type");

  /**
   * @name Type definitions
   *
   */
  ///@{
  using execution_space =
      typename ParallelConfig::execution_space; ///< Execution space
  using simd = typename ParallelConfig::simd;   ///< SIMD type
  using policy_type =
      Kokkos::RangePolicy<execution_space>; ///< Underlying Kokkos range policy
                                            ///< type

  using iterator_type = specfem::iterator::range<simd>; ///< Iterator type
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static bool isKokkosRangePolicy =
      true; ///< Indicates that this is a Kokkos range policy
  constexpr static bool isKokkosTeamPolicy =
      false; ///< Indicates that this is not a Kokkos team policy
  constexpr static bool isPointPolicy =
      true; ///< Indicates that this is a quadrature point policy
  constexpr static bool isEdgePolicy =
      false; ///< Indicates that this is not an element edge policy
  constexpr static bool isFacePolicy =
      false; ///< Indicates that this is not a element face policy
  constexpr static bool isElementPolicy =
      false; ///< Indicates that this is not an element policy
  constexpr static int chunk_size =
      1; ///< Number of chunks a range can be divided into. Indices within a
         ///< chunk are processed serially in a single thread.
  ///@}

private:
  constexpr static int simd_size = simd::size();
  constexpr static bool using_simd = simd::using_simd;

public:
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor.
   *
   */
  KOKKOS_FUNCTION
  range() = default;

  /**
   * @brief Construct a range policy with a given range size.
   *
   * Initializes the index_view with the range indices.
   *
   * @param range_size Size of the range.
   */
  KOKKOS_FUNCTION range(const int range_size)
      : policy_type(0, range_size / simd_size + (range_size % simd_size != 0)),
        range_size(range_size) {}
  ///@}

  /**
   * @brief Implicit conversion to the underlying Kokkos range policy type.
   *
   * @return const PolicyType & Underlying Kokkos range policy type.
   */
  operator const policy_type &() const { return *this; }

  /**
   * @brief Get the range iterator for a given range index.
   *
   * @param range_index starting index within the range.
   * @return iterator_type Range iterator.
   */
  KOKKOS_FUNCTION iterator_type range_iterator(const int range_index) const {
    const int starting_index = range_index * simd_size;
    const int number_elements = (starting_index + simd_size < range_size)
                                    ? simd_size
                                    : range_size - starting_index;
    return iterator_type(starting_index, number_elements);
  }

private:
  int range_size; ///< Size of the range.
};
} // namespace policy
} // namespace specfem
