#pragma once

#include "specfem/point/local_coordinates.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

/**
 * @brief Whether the point lies inside the reference element.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param coordinates Local coordinates of the located point.
 * @return true if the element index is valid and every local coordinate
 * magnitude is within the reference element \f$ [-1, 1] \f$.
 */
template <specfem::element::dimension_tag DimensionTag>
KOKKOS_INLINE_FUNCTION bool
inside(const specfem::point::local_coordinates<DimensionTag> &coordinates) {
  bool result = coordinates.ispec >= 0 &&
                Kokkos::abs(coordinates.xi) <= type_real(1) &&
                Kokkos::abs(coordinates.gamma) <= type_real(1);
  if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
    result = result && Kokkos::abs(coordinates.eta) <= type_real(1);
  }
  return result;
}

/**
 * @brief Whether the point lies outside the reference element beyond @p
 * tolerance.
 *
 * Deliberately not the negation of @ref inside(): a located point in the
 * tolerance band (1 < |coord| <= tolerance) is neither inside nor outside.
 * An unlocated point (ispec < 0) is not in any element and therefore counts
 * as outside.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param coordinates Local coordinates of the located point.
 * @param tolerance Tolerance on the reference-element coordinate magnitude.
 * @return true if the element index is invalid (ispec < 0) or any local
 * coordinate magnitude exceeds @p tolerance.
 */
template <specfem::element::dimension_tag DimensionTag>
KOKKOS_INLINE_FUNCTION bool
outside(const specfem::point::local_coordinates<DimensionTag> &coordinates,
        const type_real tolerance) {
  bool result = coordinates.ispec < 0 ||
                Kokkos::abs(coordinates.xi) > tolerance ||
                Kokkos::abs(coordinates.gamma) > tolerance;
  if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
    result = result || Kokkos::abs(coordinates.eta) > tolerance;
  }
  return result;
}

} // namespace algorithms
} // namespace specfem
