#pragma once

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
template <typename T, int N> struct array1d {
  using value_type = T;
  T data[N]; ///< Data array

  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == N, bool>::type = true>
  KOKKOS_INLINE_FUNCTION array1d(const Args &...args) : data{ args... } {}

  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION
  array1d(const Kokkos::View<T *, Layout, MemorySpace, MemoryTraits> view) {
#ifndef NDEBUG
    assert(view.extent(0) == N);
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = view(i);
    }
  }

  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION void
  operator=(const Kokkos::View<T *, Layout, MemorySpace, MemoryTraits> view) {
#ifndef NDEBUG
    assert(view.extent(0) == N);
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = view(i);
    }
  }

  KOKKOS_INLINE_FUNCTION array1d(const T value) {
    for (int i = 0; i < N; ++i) {
      data[i] = value;
    }
  }

  KOKKOS_INLINE_FUNCTION array1d(const T *values) {
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
  KOKKOS_INLINE_FUNCTION T &operator[](const int &i) { return data[i]; }

  /**
   * @brief operator [] to access the data array
   *
   * @param i index
   * @return const T& reference to the data array
   */
  KOKKOS_INLINE_FUNCTION const T &operator[](const int &i) const {
    return data[i];
  }

  /**
   * @brief operator += to add two arrays
   *
   * @param rhs right hand side array
   * @return array1d<T>& reference to the array
   */
  KOKKOS_INLINE_FUNCTION array1d<T, N> &operator+=(const array1d<T, N> &rhs) {
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
  KOKKOS_INLINE_FUNCTION array1d(const array1d<T, N> &other) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = other[i];
    }
  }

  KOKKOS_INLINE_FUNCTION type_real l2_norm() const {
    type_real norm = 0.0;
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      norm += data[i] * data[i];
    }
    return sqrt(norm);
  }
};
} // namespace impl

template <typename T, int N>
struct ScalarPointViewType : public impl::array1d<T, N> {
  constexpr static int components = N;
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;
  using type = impl::array1d<T, N>;

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType() = default;

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const T value) : impl::array1d<T, N>(value) {}

  KOKKOS_INLINE_FUNCTION
  ScalarPointViewType(const T *values) : impl::array1d<T, N>(values) {}

  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(
      const Kokkos::View<T *, Layout, MemorySpace, MemoryTraits> view)
      : impl::array1d<T, N>(view) {}

  template <typename... Args, typename Enable = typename std::enable_if<
                                  sizeof...(Args) == N>::type>
  KOKKOS_INLINE_FUNCTION ScalarPointViewType(const Args &...args)
      : impl::array1d<T, N>(args...) {}

  KOKKOS_INLINE_FUNCTION
  T &operator()(const int i) { return this->data[i]; }

  KOKKOS_INLINE_FUNCTION
  const T &operator()(const int i) const { return this->data[i]; }

  KOKKOS_INLINE_FUNCTION
  T &operator[](const int i) = delete;
};

template <typename T, int NumberOfDimensions, int Components>
struct VectorPointViewType
    : public impl::array1d<T, NumberOfDimensions * Components> {
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
  constexpr static bool isPointViewType = true;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;

  using type = impl::array1d<T, NumberOfDimensions * Components>;

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType() = default;

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const T value)
      : impl::array1d<T, NumberOfDimensions * Components>(value) {}

  KOKKOS_INLINE_FUNCTION
  VectorPointViewType(const T *values)
      : impl::array1d<T, NumberOfDimensions * Components>(values) {}

  template <typename MemorySpace, typename Layout, typename MemoryTraits>
  KOKKOS_INLINE_FUNCTION VectorPointViewType(
      const Kokkos::View<T **, Layout, MemorySpace, MemoryTraits> view) {
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
  T &operator()(const int i, const int j) {
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
  const T &operator()(const int i, const int j) const {
#ifdef Kokkos_DEBUG_ENABLE_BOUNDS_CHECK
    assert(i < NumberOfDimensions);
    assert(j < Components);
#endif
    return this->data[i * Components + j];
  }

  KOKKOS_INLINE_FUNCTION
  T &operator[](const int i) = delete;

  KOKKOS_INLINE_FUNCTION
  type_real l2_norm() const = delete;
};

} // namespace datatype
} // namespace specfem
