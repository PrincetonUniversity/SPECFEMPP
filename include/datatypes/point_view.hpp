#pragma once

#include "impl/array1d.hpp"
#include "simd.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values at a GLL point. If N is small,
 * generates a datatype within a register.
 *
 * @tparam T Data type of the scalar values
 * @tparam N Number of scalar values (components) at the GLL point
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int N, bool UseSIMD>
struct ScalarPointViewType : public impl::array1d<T, N, UseSIMD> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using type = impl::array1d<T, N, UseSIMD>; ///< Underlying data type used to
                                             ///< store values
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int components = N; ///< Number of scalar values at the GLL
                                       ///< point
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;
  ///@}

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType() = default;

  /**
   * @brief Construct a new ScalarPointViewType object with all components set
   * to the same value
   *
   * @param value Value to set all components to
   */
  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const value_type value)
      : impl::array1d<T, N, UseSIMD>(value) {}

  /**
   * @brief Construct a new ScalarPointViewType object from an array of values
   *
   * @param values Array of values
   */
  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const value_type *values)
      : impl::array1d<T, N, UseSIMD>(values) {}

  /**
   * @brief Construct a new ScalarPointViewType object from a 1-D Kokkos view
   *
   * @tparam MemorySpace Memory space of the view (deduced)
   * @tparam Layout Layout of the view (deduced)
   * @tparam MemoryTraits Memory traits of the view (deduced)
   * @param view 1-D Kokkos view
   */
  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(
      const Kokkos::View<value_type *, Layout, MemorySpace, MemoryTraits> view)
      : impl::array1d<T, N, UseSIMD>(view) {}

  /**
   * @brief Construct a new ScalarPointViewType object from a N argument list
   *
   * @tparam Args N arguments of type T
   * @param args N-elements of the array
   * @return KOKKOS_INLINE_FUNCTION
   */
  template <typename... Args, typename Enable = typename std::enable_if<
                                  sizeof...(Args) == N>::type>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(const Args &...args)
      : impl::array1d<T, N, UseSIMD>(args...) {}

  ///@}

  /**
   * @name Accessors
   *
   */
  ///@{
  /**
   * @brief Access the i-th component of the array
   *
   * @param i Index of the component
   * @return T& Reference to the i-th component
   */
  KOKKOS_INLINE_FUNCTION
  value_type &operator()(const int i) { return this->data[i]; }

  /**
   * @brief Access the i-th component of the array
   *
   * @param i Index of the component
   * @return T& const Reference to the i-th component
   */
  KOKKOS_INLINE_FUNCTION
  const value_type &operator()(const int i) const { return this->data[i]; }

  /**
   * @brief Access the i-th component of the array
   *
   * Use the operator() method to access the i-th component of the array
   *
   * @param i Index of the component
   * @return T& Reference to the i-th component
   */
  KOKKOS_INLINE_FUNCTION
  value_type &operator[](const int i) = delete;
  ///@}
};

/**
 * @brief Datatype used to store vector values at a GLL point. If
 * NumberOfDimensions && Components is small, generates a datatype within a
 * register.
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfDimensions Number of dimensions of the vector
 * @tparam Components Number of components of the vector
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int NumberOfDimensions, int Components, bool UseSIMD>
struct VectorPointViewType
    : public impl::array1d<T, NumberOfDimensions * Components, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = T; ///< Base type of the array
  using type = impl::array1d<T, NumberOfDimensions * Components,
                             UseSIMD>; ///< Underlying data type used to store
                                       ///< values
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  constexpr static bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int components = Components; ///< Number of components of the
                                                ///< vector
  constexpr static int dimensions =
      NumberOfDimensions; ///< Number of dimensions
                          ///< of the vector
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;
  ///@}

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  VectorPointViewType() = default;

  /**
   * @brief Construct a new VectorPointViewType object with all components set
   * to the same value
   *
   * @param value Value to set all components to
   */
  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const value_type value)
      : impl::array1d<T, NumberOfDimensions * Components, UseSIMD>(value) {}

  /**
   * @brief Construct a new VectorPointViewType object from an array of values
   *
   * @param values Array of values
   */
  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const value_type *values)
      : impl::array1d<T, NumberOfDimensions * Components, UseSIMD>(values) {}

  /**
   * @brief Construct a new VectorPointViewType object from a 2-D Kokkos view
   *
   * @tparam MemorySpace Memory space of the view (deduced)
   * @tparam Layout Layout of the view (deduced)
   * @tparam MemoryTraits Memory traits of the view (deduced)
   * @param view 2-D Kokkos view
   */
  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION VectorPointViewType(
      const Kokkos::View<value_type **, Layout, MemorySpace, MemoryTraits>
          view) {
#ifndef NDEBUG
    assert(view.extent(0) == NumberOfDimensions);
    assert(view.extent(1) == Components);
#endif

    for (int i = 0; i < NumberOfDimensions; ++i) {
      for (int j = 0; j < Components; ++j) {
        this->data[i * Components + j] = view(i, j);
      }
    }

    return;
  }

  /**
   * @name Data accessors
   *
   */
  ///@{

  /**
   * @brief Access the data array at the i-th dimension and j-th component
   *
   * @param i Index of the dimension
   * @param j Index of the component
   * @return T& Reference to the i-th dimension and j-th component
   */
  KOKKOS_INLINE_FUNCTION
  value_type &operator()(const int i, const int j) {
#ifdef Kokkos_DEBUG_ENABLE_BOUNDS_CHECK
    assert(i < NumberOfDimensions);
    assert(j < Components);
#endif
    return this->data[i * Components + j];
  }

  /**
   * @brief Access the data array at the i-th dimension and j-th component
   *
   * @param i Index of the dimension
   * @param j Index of the component
   * @return T& const Reference to the i-th dimension and j-th component
   */
  KOKKOS_INLINE_FUNCTION
  const value_type &operator()(const int i, const int j) const {
#ifdef Kokkos_DEBUG_ENABLE_BOUNDS_CHECK
    assert(i < NumberOfDimensions);
    assert(j < Components);
#endif
    return this->data[i * Components + j];
  }

  /**
   * @brief Access the data array at the i-th dimension and j-th component
   *
   * Use the operator() method to access the i-th dimension and j-th component
   * of the array
   */
  KOKKOS_INLINE_FUNCTION
  value_type &operator[](const int i) = delete;

  ///@}

  /**
   * @name Member functions
   *
   */
  ///@{
  /**
   * @brief Compute the L2 norm of the vector
   *
   * Illdefined to compute the L2 norm of a vector type.
   *
   * @return T L2 norm of the vector
   */
  KOKKOS_INLINE_FUNCTION
  value_type l2_norm() const = delete;
};

} // namespace datatype
} // namespace specfem
