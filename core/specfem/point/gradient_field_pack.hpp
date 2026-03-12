#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem::point {

/**
 * @brief Named gradient holder for displacement gradient (accessed as .du)
 *
 * @tparam V Point-level gradient tensor type
 *           (e.g., specfem::datatype::TensorPointViewType<...>)
 */
template <typename V> struct holds_du {
  V du;

  KOKKOS_FUNCTION holds_du() = default;
  KOKKOS_FUNCTION holds_du(const holds_du &) = default;
  KOKKOS_FUNCTION holds_du &operator=(const holds_du &) = default;
  KOKKOS_FUNCTION explicit holds_du(const V &v) : du(v) {}
};

/**
 * @brief Named gradient holder for velocity gradient (accessed as .dv)
 *
 * @tparam V Point-level gradient tensor type
 */
template <typename V> struct holds_dv {
  V dv;

  KOKKOS_FUNCTION holds_dv() = default;
  KOKKOS_FUNCTION holds_dv(const holds_dv &) = default;
  KOKKOS_FUNCTION holds_dv &operator=(const holds_dv &) = default;
  KOKKOS_FUNCTION explicit holds_dv(const V &v) : dv(v) {}
};

/**
 * @brief Named gradient holder for acceleration gradient (accessed as .da)
 *
 * @tparam V Point-level gradient tensor type
 */
template <typename V> struct holds_da {
  V da;

  KOKKOS_FUNCTION holds_da() = default;
  KOKKOS_FUNCTION holds_da(const holds_da &) = default;
  KOKKOS_FUNCTION holds_da &operator=(const holds_da &) = default;
  KOKKOS_FUNCTION explicit holds_da(const V &v) : da(v) {}
};

/**
 * @brief Variadic named-holder pack for point-level gradient tensors.
 *
 * Bundles multiple named gradient holders (holds_du, holds_dv, holds_da) via
 * multiple inheritance. Plain value types — no scratch memory involved.
 *
 * Example:
 * @code
 * // Displacement + velocity gradient pack
 * using GradPack = GradientPack<holds_du<TensorType>, holds_dv<TensorType>>;
 * GradPack gp(holds_du<TensorType>(g_u), holds_dv<TensorType>(g_v));
 * gp.du  // ∂u/∂x
 * gp.dv  // ∂v/∂x
 * @endcode
 *
 * @tparam Holders Variadic list of named gradient holder types
 */
template <typename... Holders> struct GradientPack : Holders... {
  static constexpr std::size_t size = sizeof...(Holders);

  KOKKOS_FUNCTION GradientPack() = default;
  KOKKOS_FUNCTION GradientPack(const GradientPack &) = default;
  KOKKOS_FUNCTION GradientPack &operator=(const GradientPack &) = default;

  KOKKOS_FUNCTION GradientPack(const Holders &...holders)
      : Holders(holders)... {}
};

} // namespace specfem::point
