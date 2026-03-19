#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/point/gradient_field_pack.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::point {

/**
 * @brief Physics wrapper around a GradientPack.
 *
 * Interprets the raw gradient tensors produced by the gradient algorithm as
 * physically meaningful displacement/velocity derivatives, and wraps them in
 * typed field_derivatives objects.
 *
 * - get_du<Tags>() always returns field_derivatives<Tags> wrapping du.
 * - get_dv<Tags>() returns field_derivatives<Tags> wrapping dv when the
 *   underlying GradientPack has size >= 2, or null_field_derivatives otherwise.
 *   Callers can branch with:
 *     if constexpr (!is_same_v<decltype(fd.get_dv<Tags>()),
 * null_field_derivatives>)
 *
 * @tparam GradPackType A GradientPack<...> specialization
 */
template <typename GradPackType> struct FieldDerivativesPack {
  GradPackType grad; ///< Underlying raw gradient pack

  /// True when a velocity gradient is available (attenuation active).
  static constexpr bool has_dv = (GradPackType::size >= 2);

  KOKKOS_FUNCTION FieldDerivativesPack() = default;
  KOKKOS_FUNCTION FieldDerivativesPack(const FieldDerivativesPack &) = default;
  KOKKOS_FUNCTION FieldDerivativesPack &
  operator=(const FieldDerivativesPack &) = default;

  /// Construct from a GradientPack produced by the gradient algorithm.
  KOKKOS_FUNCTION explicit FieldDerivativesPack(const GradPackType &g)
      : grad(g) {}

  /**
   * @brief Return the displacement field derivative du = ∂u/∂x.
   *
   * @tparam Tags Point tags determining the field_derivatives type
   */
  template <typename Tags>
  KOKKOS_FUNCTION specfem::point::field_derivatives<Tags> get_du() const {
    return specfem::point::field_derivatives<Tags>(grad.du);
  }

  /**
   * @brief Return the velocity field derivative dv = ∂v/∂x.
   *
   * Return type depends on whether a velocity gradient is stored:
   *   - has_dv == true  → field_derivatives<Tags>
   *   - has_dv == false → null_field_derivatives  (sentinel, zero size)
   *
   * Always compiles. Branch on the result with:
   *   if constexpr (FieldDerivativesPackType::has_dv) { ... }
   *
   * @tparam Tags Point tags determining the field_derivatives type
   */
  template <typename Tags> KOKKOS_FUNCTION auto get_dv() const {
    if constexpr (has_dv) {
      return specfem::point::field_derivatives<Tags>(grad.dv);
    } else {
      return specfem::point::null_field_derivatives{};
    }
  }
};

// ---------------------------------------------------------------------------
// Convenience alias
// ---------------------------------------------------------------------------

/**
 * @brief Selects FieldDerivativesPack<GradientPack<T>> for attenuation::none,
 *        FieldDerivativesPack<GradientPack<T,T>> otherwise.
 *
 * @tparam AttenuationTag Element attenuation tag
 * @tparam T              Raw TensorPointViewType
 */
template <specfem::element::attenuation_tag AttenuationTag, typename T>
using stiffness_field_derivatives_pack =
    FieldDerivativesPack<stiffness_gradient_pack<AttenuationTag, T> >;

} // namespace specfem::point
