#pragma once

#include "bounds.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem::assembly::info::impl {

/**
 * @brief Lightweight struct for element-local min/max tracking inside lambdas
 */
template <typename T = type_real>
struct LocalMinMax {
  T min_val;
  T max_val;

  KOKKOS_INLINE_FUNCTION
  LocalMinMax()
      : min_val(Kokkos::Experimental::finite_max_v<T>),
        max_val(Kokkos::Experimental::finite_min_v<T>) {}

  KOKKOS_INLINE_FUNCTION
  void update(T value) {
    min_val = Kokkos::fmin(min_val, value);
    max_val = Kokkos::fmax(max_val, value);
  }

  KOKKOS_INLINE_FUNCTION
  void update_min(T value) {
    min_val = Kokkos::fmin(min_val, value);
  }

  KOKKOS_INLINE_FUNCTION
  void update_max(T value) {
    max_val = Kokkos::fmax(max_val, value);
  }
};

/**
 * @brief Scatter-based min/max reducer for parallel reductions
 *
 * Encapsulates Kokkos views, scatter views, initialization, and finalization
 * for computing global min/max values across parallel iterations.
 *
 * @tparam T Value type for min/max tracking
 * @tparam Extent Compile-time extent (0 = dynamic, use runtime size)
 */
template <typename T = type_real, size_t Extent = 0>
struct ScatterMinMax {
  using view_type = Kokkos::View<T*>;
  using scatter_min_type = Kokkos::Experimental::ScatterView<
      T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
      Kokkos::Experimental::ScatterMin>;
  using scatter_max_type = Kokkos::Experimental::ScatterView<
      T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
      Kokkos::Experimental::ScatterMax>;

  view_type min_view;
  view_type max_view;
  scatter_min_type scatter_min;
  scatter_max_type scatter_max;
  size_t size_;

  // Deduce access types from the scatter views
  using scatter_min_access_type =
      decltype(std::declval<scatter_min_type>().access());
  using scatter_max_access_type =
      decltype(std::declval<scatter_max_type>().access());

  /**
   * @brief Accessor for use inside Kokkos lambdas
   */
  struct Accessor {
    scatter_min_access_type min_access;
    scatter_max_access_type max_access;

    // Single-value interface (backwards compatible, uses index 0)
    KOKKOS_INLINE_FUNCTION
    void update(T value) const {
      min_access(0).update(value);
      max_access(0).update(value);
    }

    KOKKOS_INLINE_FUNCTION
    void update_min(T value) const {
      min_access(0).update(value);
    }

    KOKKOS_INLINE_FUNCTION
    void update_max(T value) const {
      max_access(0).update(value);
    }

    // Indexed interface for multi-value tracking
    KOKKOS_INLINE_FUNCTION
    void update(size_t i, T value) const {
      min_access(i).update(value);
      max_access(i).update(value);
    }

    KOKKOS_INLINE_FUNCTION
    void update_min(size_t i, T value) const {
      min_access(i).update(value);
    }

    KOKKOS_INLINE_FUNCTION
    void update_max(size_t i, T value) const {
      max_access(i).update(value);
    }

    // Array interface - updates each element against corresponding index
    // Requires compile-time Extent > 0 and ArrayType::size() == Extent
    template <typename ArrayType,
              size_t E = Extent,
              std::enable_if_t<(E > 0), int> = 0>
    KOKKOS_INLINE_FUNCTION
    void update(const ArrayType& values) const {
      static_assert(ArrayType::size() == Extent,
                    "ArrayType extent must match ScatterMinMax Extent");
      for (size_t i = 0; i < Extent; ++i) {
        min_access(i).update(values[i]);
        max_access(i).update(values[i]);
      }
    }

    template <typename ArrayType,
              size_t E = Extent,
              std::enable_if_t<(E > 0), int> = 0>
    KOKKOS_INLINE_FUNCTION
    void update_min(const ArrayType& values) const {
      static_assert(ArrayType::size() == Extent,
                    "ArrayType extent must match ScatterMinMax Extent");
      for (size_t i = 0; i < Extent; ++i) {
        min_access(i).update(values[i]);
      }
    }

    template <typename ArrayType,
              size_t E = Extent,
              std::enable_if_t<(E > 0), int> = 0>
    KOKKOS_INLINE_FUNCTION
    void update_max(const ArrayType& values) const {
      static_assert(ArrayType::size() == Extent,
                    "ArrayType extent must match ScatterMinMax Extent");
      for (size_t i = 0; i < Extent; ++i) {
        max_access(i).update(values[i]);
      }
    }
  };

  ScatterMinMax(const std::string &name, size_t size = 1)
      : min_view("min_" + name, size), max_view("max_" + name, size),
        scatter_min(min_view), scatter_max(max_view), size_(size) {
    Kokkos::deep_copy(min_view, std::numeric_limits<T>::max());
    Kokkos::deep_copy(max_view, std::numeric_limits<T>::lowest());
  }

  Accessor access() const {
    return Accessor{ scatter_min.access(), scatter_max.access() };
  }

  void contribute() {
    Kokkos::Experimental::contribute(min_view, scatter_min);
    Kokkos::Experimental::contribute(max_view, scatter_max);
  }

  size_t size() const { return size_; }

  // Single-value interface (backwards compatible, uses index 0)
  Bounds get_bounds() const {
    auto min_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_view);
    auto max_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_view);
    return Bounds(min_h(0), max_h(0));
  }

  // Indexed interface for multi-value tracking
  Bounds get_bounds(size_t i) const {
    auto min_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_view);
    auto max_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_view);
    return Bounds(min_h(i), max_h(i));
  }

  // Get all bounds as a vector
  std::vector<Bounds> get_all_bounds() const {
    auto min_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_view);
    auto max_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_view);
    std::vector<Bounds> bounds;
    bounds.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
      bounds.emplace_back(min_h(i), max_h(i));
    }
    return bounds;
  }
};

} // namespace specfem::assembly::info::impl
