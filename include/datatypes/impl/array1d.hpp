#pragma once

#include "../simd.hpp"

namespace specfem {
namespace datatype {
namespace impl {

/**
 * @brief 1-Dimensional array used to store data within registers
 *
 * array1d is not intended to be used directly. It is used as a register
 * datatype by ScalarPointViewType and VectorPointViewType.
 *
 * @tparam T Data type
 * @tparam N Number of elements in the array
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int N, bool UseSIMD = false> struct array1d {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using value_type =
      typename specfem::datatype::simd<T, UseSIMD>::datatype; ///< Value type
                                                              ///< used to store
                                                              ///< the elements
                                                              ///< of the array
  using base_type = T; ///< Base type of the array
  bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  value_type data[N]; ///< Data array

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{
  /**
   * @brief Construct a new array1d object
   *
   * Constructor with N arguments of type T
   *
   * @tparam Args N arguments of type T
   * @param args N-elements of the array
   */
  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == N, bool>::type = true>
  KOKKOS_INLINE_FUNCTION array1d(const Args &...args) : data{ args... } {}

  /**
   * @brief Construct a new array1d from a 1-D Kokkos view. Copies the data from
   * the view to the array
   *
   * @tparam MemorySpace Memory space of the view (deduced)
   * @tparam Layout Layout of the view (deduced)
   * @tparam MemoryTraits  Memory traits of the view (deduced)
   * @param view 1-D Kokkos view
   */
  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION
  array1d(const Kokkos::View<value_type *, Layout, MemorySpace, MemoryTraits>
              view) {
#ifndef NDEBUG
    assert(view.extent(0) == N);
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = view(i);
    }
  }

  /**
   * @brief Copy assignment operator from a 1-D Kokkos view. Copies the data
   * from the view to the array
   *
   * @tparam MemorySpace Memory space of the view (deduced)
   * @tparam Layout Layout of the view (deduced)
   * @tparam MemoryTraits Memory traits of the view (deduced)
   * @param view 1-D Kokkos view
   */
  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION void
  operator=(const Kokkos::View<value_type *, Layout, MemorySpace, MemoryTraits>
                view) {
#ifndef NDEBUG
    assert(view.extent(0) == N);
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = view(i);
    }
  }

  /**
   * @brief Construct a new array1d object where all elements are initialized to
   * a value
   *
   * @param value Value to initialize the array
   */
  KOKKOS_INLINE_FUNCTION array1d(const value_type value) {
    for (int i = 0; i < N; ++i) {
      data[i] = value;
    }
  }

  /**
   * @brief Construct a new array1d object from an array of values
   *
   * @param values Array of values
   */
  KOKKOS_INLINE_FUNCTION array1d(const value_type *values) {
    for (int i = 0; i < N; ++i) {
      data[i] = values[i];
    }
  }

  /**
   * @brief Default constructor
   *
   * Initializes the array with zeros
   *
   */
  KOKKOS_INLINE_FUNCTION array1d() { init(); }

  /**
   * @brief Copy constructor
   *
   * @param other other array
   */
  KOKKOS_INLINE_FUNCTION array1d(const array1d<value_type, N> &other) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = other[i];
    }
  }
  ///@}

  /**
   * @name Data accessors
   *
   */
  ///@{
  /**
   * @brief operator [] to access the data array
   *
   * @param i index
   * @return T& reference to the data array
   */
  KOKKOS_INLINE_FUNCTION value_type &operator[](const int &i) {
    return data[i];
  }

  /**
   * @brief operator [] to access the data array
   *
   * @param i index
   * @return const T& reference to the data array
   */
  KOKKOS_INLINE_FUNCTION const value_type &operator[](const int &i) const {
    return data[i];
  }
  ///@}

  /**
   * @name Member functions
   *
   */
  ///@{
  /**
   * @brief operator += to add two arrays
   *
   * @param rhs right hand side array
   * @return array1d<T>& reference to the array
   */
  KOKKOS_INLINE_FUNCTION array1d<value_type, N> &
  operator+=(const array1d<value_type, N> &rhs) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] += rhs[i];
    }
    return *this;
  }

  /**
   * @brief Compute the l2 norm of the array
   *
   * @return value_type l2 norm of the array
   */
  KOKKOS_INLINE_FUNCTION value_type l2_norm() const {
    value_type norm = 0.0;
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      norm += data[i] * data[i];
    }
    return Kokkos::sqrt(norm);
  }
  ///@}

  KOKKOS_INLINE_FUNCTION void init() {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = 0.0;
    }
  }
};

} // namespace impl
} // namespace datatype
} // namespace specfem
