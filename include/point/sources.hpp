#pragma once

#include "enumerations/interface.hpp"
#include "point/field.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store source information at a given quadrature point
 *
 * @tparam DimensionType Dimension of the spectral element
 * @tparam MediumTag Medium tag of the spectral element
 * @tparam WavefieldType Wavefield type on which the source is applied
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
struct source {

  constexpr static auto dimension = DimensionType; ///< Dimension of the
                                                   ///< spectral element
  constexpr static auto medium_tag = MediumTag; ///< Medium tag of the spectral
                                                ///< element
  constexpr static auto wavefield_tag = WavefieldType; ///< Wavefield type on
                                                       ///< which the source is
                                                       ///< applied
  constexpr static bool is_point_source = true; ///< Boolean indicating whether
                                                ///< the point type is a source

  constexpr static int components =
      specfem::element::attributes<DimensionType,
                                   MediumTag>::components(); ///< Number
                                                             ///< of
                                                             ///< components
                                                             ///< in
                                                             ///< the
                                                             ///< medium

  using value_type =
      specfem::datatype::ScalarPointViewType<type_real, components,
                                             false>; ///<
                                                     ///< Value
                                                     ///< type
                                                     ///< to
                                                     ///< store
                                                     ///< source
                                                     ///< information

  using AccelerationType =
      specfem::point::field<dimension, medium_tag, false, false, true, false,
                            false>; ///< Acceleration return type

  value_type stf;                  ///< Source time function
  value_type lagrange_interpolant; ///< Lagrange interpolant

  KOKKOS_INLINE_FUNCTION source() = default;

  /**
   * @brief Constructor
   *
   * @param stf Source time function
   * @param lagrange_interpolant Lagrange interpolant
   *
   */
  KOKKOS_INLINE_FUNCTION source(const value_type &stf,
                                const value_type &lagrange_interpolant)
      : stf(stf), lagrange_interpolant(lagrange_interpolant) {}

  /**
   * @brief Compute acceleration when source is applied to the wavefield
   *
   * @return value_type Acceleration at the quadrature point
   */
  KOKKOS_INLINE_FUNCTION AccelerationType compute_acceleration() const {
    AccelerationType acceleration;
    for (int i = 0; i < components; i++) {
      acceleration.acceleration(i) = stf(i) * lagrange_interpolant(i);
    }
    return acceleration;
  }
};

} // namespace point
} // namespace specfem
