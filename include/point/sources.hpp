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
};

} // namespace point
} // namespace specfem
