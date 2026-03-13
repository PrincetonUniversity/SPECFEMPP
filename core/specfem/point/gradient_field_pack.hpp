#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem::point {

/// @brief Primary template — undefined; use 1-, 2-, or 3-type specializations.
template <typename... Ts> struct GradientPack;

// ---------------------------------------------------------------------------
// 1-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Single gradient tensor pack (accessed as .df).
 *
 * @tparam T Point-level gradient tensor type
 *           (e.g., specfem::datatype::TensorPointViewType<...>)
 */
template <typename T> struct GradientPack<T> {
  static constexpr std::size_t size = 1;
  T df;

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION explicit GradientPack(const T &t) : df(t) {}

  KOKKOS_FUNCTION const T &get_df() const { return df; }
  KOKKOS_FUNCTION void get_dg() const {}
};

// ---------------------------------------------------------------------------
// 2-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Two gradient tensor pack (accessed as .df and .dg).
 *
 * @tparam T1 First point-level gradient tensor type
 * @tparam T2 Second point-level gradient tensor type
 */
template <typename T1, typename T2> struct GradientPack<T1, T2> {
  static constexpr std::size_t size = 2;
  T1 df;
  T2 dg;

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack(const T1 &t1, const T2 &t2) : df(t1), dg(t2) {}

  KOKKOS_FUNCTION const T1 &get_df() const { return df; }
  KOKKOS_FUNCTION const T2 &get_dg() const { return dg; }
};

// ---------------------------------------------------------------------------
// 3-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Three gradient tensor pack (accessed as .df, .dg, and .dh).
 *
 * @tparam T1 First point-level gradient tensor type
 * @tparam T2 Second point-level gradient tensor type
 * @tparam T3 Third point-level gradient tensor type
 */
template <typename T1, typename T2, typename T3>
struct GradientPack<T1, T2, T3> {
  static constexpr std::size_t size = 3;
  T1 df;
  T2 dg;
  T3 dh;

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack(const T1 &t1, const T2 &t2, const T3 &t3)
      : df(t1), dg(t2), dh(t3) {}

  KOKKOS_FUNCTION const T1 &get_df() const { return df; }
  KOKKOS_FUNCTION const T2 &get_dg() const { return dg; }
};

} // namespace specfem::point
