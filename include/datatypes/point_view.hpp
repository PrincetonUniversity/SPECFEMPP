#pragma once

#include "simd.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

namespace impl {

/**
 * @brief Array to store temporary values when doing Kokkos reductions
 *
 * @tparam T array type
 * @tparam N size of array
 */
template <typename T, int N, bool UseSIMD = false> struct array1d {
  using value_type = typename specfem::datatype::simd<T, UseSIMD>::datatype;
  value_type data[N]; ///< Data array

  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == N, bool>::type = true>
  KOKKOS_INLINE_FUNCTION array1d(const Args &...args) : data{ args... } {}

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

  KOKKOS_INLINE_FUNCTION array1d(const value_type value) {
    for (int i = 0; i < N; ++i) {
      data[i] = value;
    }
  }

  KOKKOS_INLINE_FUNCTION array1d(const value_type *values) {
    for (int i = 0; i < N; ++i) {
      data[i] = values[i];
    }
  }

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
   * @brief Initialize the array for sum reductions
   *
   */
  KOKKOS_INLINE_FUNCTION void init() {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = 0.0;
    }
  }

  // Default constructor
  /**
   * @brief Construct a new array type object
   *
   */
  KOKKOS_INLINE_FUNCTION array1d() { init(); }

  // Copy constructor
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
};
} // namespace impl

template <typename T, int N, bool UseSIMD>
struct ScalarPointViewType : public impl::array1d<T, N, UseSIMD> {
  constexpr static int components = N;
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;
  using type = impl::array1d<T, N, UseSIMD>;
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using value_type = typename type::value_type;

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType() = default;

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const value_type value)
      : impl::array1d<T, N, UseSIMD>(value) {}

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const value_type *values)
      : impl::array1d<T, N, UseSIMD>(values) {}

  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(
      const Kokkos::View<value_type *, Layout, MemorySpace, MemoryTraits> view)
      : impl::array1d<T, N, UseSIMD>(view) {}

  template <typename... Args, typename Enable = typename std::enable_if<
                                  sizeof...(Args) == N>::type>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(const Args &...args)
      : impl::array1d<T, N, UseSIMD>(args...) {}

  KOKKOS_INLINE_FUNCTION
  value_type &operator()(const int i) { return this->data[i]; }

  KOKKOS_INLINE_FUNCTION
  const value_type &operator()(const int i) const { return this->data[i]; }

  KOKKOS_INLINE_FUNCTION
  value_type &operator[](const int i) = delete;
};

template <typename T, int NumberOfDimensions, int Components, bool UseSIMD>
struct VectorPointViewType
    : public impl::array1d<T, NumberOfDimensions * Components, UseSIMD> {
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;

  using type = impl::array1d<T, NumberOfDimensions * Components, UseSIMD>;
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using value_type = typename type::value_type;

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType() = default;

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const value_type value)
      : impl::array1d<T, NumberOfDimensions * Components, UseSIMD>(value) {}

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const value_type *values)
      : impl::array1d<T, NumberOfDimensions * Components, UseSIMD>(values) {}

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
   * @brief  Access the data array
   *
   * @param i
   * @param j
   * @return T&
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
   * @brief  Access the data array
   *
   * @param i
   * @param j
   * @return T&
   */
  KOKKOS_INLINE_FUNCTION
  const value_type &operator()(const int i, const int j) const {
#ifdef Kokkos_DEBUG_ENABLE_BOUNDS_CHECK
    assert(i < NumberOfDimensions);
    assert(j < Components);
#endif
    return this->data[i * Components + j];
  }

  KOKKOS_INLINE_FUNCTION
  value_type &operator[](const int i) = delete;

  KOKKOS_INLINE_FUNCTION
  value_type l2_norm() const = delete;
};

} // namespace datatype
} // namespace specfem
