#pragma once

#include "bounds.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <limits>
#include <string>
#include <utility>

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
 */
template <typename T = type_real>
struct ScatterMinMax {
  using view_type = Kokkos::View<T[1]>;
  using scatter_min_type = Kokkos::Experimental::ScatterView<
      T[1], Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
      Kokkos::Experimental::ScatterMin>;
  using scatter_max_type = Kokkos::Experimental::ScatterView<
      T[1], Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
      Kokkos::Experimental::ScatterMax>;

  view_type min_view;
  view_type max_view;
  scatter_min_type scatter_min;
  scatter_max_type scatter_max;

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
  };

  ScatterMinMax(const std::string &name)
      : min_view("min_" + name), max_view("max_" + name), scatter_min(min_view),
        scatter_max(max_view) {
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

  Bounds get_bounds() const {
    auto min_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_view);
    auto max_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_view);
    return Bounds(min_h(0), max_h(0));
  }
};

} // namespace specfem::assembly::info::impl
