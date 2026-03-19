#pragma once

#include "specfem/element/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::point {

/// @brief Primary template — undefined; use 1-, 2-, or 3-type specializations.
template <typename... Ts> struct GradientPack;

// ---------------------------------------------------------------------------
// 1-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Single gradient tensor pack.
 *
 * Raw output of the gradient algorithm for one vector field.
 *
 * @tparam T TensorPointViewType (gradient of displacement field)
 */
template <typename T> struct GradientPack<T> {
  static constexpr std::size_t size = 1;
  T du; ///< Gradient tensor of the first (displacement) field

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION explicit GradientPack(const T &u) : du(u) {}
};

// ---------------------------------------------------------------------------
// 2-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Two gradient tensor pack.
 *
 * Raw output of the gradient algorithm for two vector fields (e.g.
 * displacement + velocity for SLS attenuation).
 *
 * @tparam T1 TensorPointViewType (gradient of displacement field)
 * @tparam T2 TensorPointViewType (gradient of velocity field)
 */
template <typename T1, typename T2> struct GradientPack<T1, T2> {
  static constexpr std::size_t size = 2;
  T1 du; ///< Gradient tensor of the first (displacement) field
  T2 dv; ///< Gradient tensor of the second (velocity) field

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack(const T1 &u, const T2 &v) : du(u), dv(v) {}
};

// ---------------------------------------------------------------------------
// 3-type specialization
// ---------------------------------------------------------------------------

/**
 * @brief Three gradient tensor pack.
 *
 * @tparam T1 TensorPointViewType (gradient of first field)
 * @tparam T2 TensorPointViewType (gradient of second field)
 * @tparam T3 TensorPointViewType (gradient of third field)
 */
template <typename T1, typename T2, typename T3>
struct GradientPack<T1, T2, T3> {
  static constexpr std::size_t size = 3;
  T1 du; ///< Gradient tensor of the first field
  T2 dv; ///< Gradient tensor of the second field
  T3 dw; ///< Gradient tensor of the third field

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack(const T1 &u, const T2 &v, const T3 &w)
      : du(u), dv(v), dw(w) {}
};

// ---------------------------------------------------------------------------
// Convenience alias — mirrors stiffness_field_pack pattern
// ---------------------------------------------------------------------------

/**
 * @brief Selects GradientPack<T> for attenuation::none,
 *        GradientPack<T, T> otherwise.
 *
 * @tparam AttenuationTag Element attenuation tag
 * @tparam T              Raw TensorPointViewType
 */
template <specfem::element::attenuation_tag AttenuationTag, typename T>
using stiffness_gradient_pack =
    std::conditional_t<AttenuationTag ==
                           specfem::element::attenuation_tag::none,
                       GradientPack<T>, GradientPack<T, T> >;

} // namespace specfem::point
