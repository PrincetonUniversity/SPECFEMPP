#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename T, typename PointPartialDerivativesType,
          typename PointStressIntegrandViewType, typename PointPropertiesType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPartialDerivativesType &point_partial_derivatives,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  const auto &xix = point_partial_derivatives.xix;
  const auto &xiz = point_partial_derivatives.xiz;
  const auto &gammax = point_partial_derivatives.gammax;
  const auto &gammaz = point_partial_derivatives.gammaz;

  /*
   * From:
   *
   * \partial w_{i}    \partial  w_{i}   \partial xi_{k}
   * -------------- =  ---------------   ---------------
   * \partial x_{j}    \partial xi_{k}   \partial  x_{j}
   *
   * Hence, we invert
   *
   *   | xix     xiz    |
   *   | gammax  gammaz |
   *
   * Compute the determinant
   *
   *   det J = xix * gammaz - xiz * gammax
   *
   * Inverting the matrix
   *
   * | xxi     xgamma | =    1/   |   gammaz   - xiz |
   * | zxi     zgamma |    det J  | - gammax     xix |
   *
   */

  // Compute inv of the determinant
  const auto invD = static_cast<T>(1.0) / (xix * gammaz - xiz * gammax);

  // Invert matrix (double check!)
  const auto xxi = invD * gammaz;
  const auto zxi = -invD * gammax;
  const auto xgamma = -invD * xiz;
  const auto zgamma = invD * xix;

  /* The final contribution for the Forces comes from Levi-Civita symbol
   * dotted with stress tensor. For PSV-T system only the rotation around
   * y-axis is non-zero meaning only t_{zx} and t_{xz} are relevant.
   *
   * Compute the components of the stress tensor from the integrand
   */
  // const auto t_xx = F(0, 0) * xxi + F(1, 0) * xgamma;
  const auto t_xz = F(0, 0) * zxi + F(0, 1) * zgamma;
  const auto t_zx = F(1, 0) * xxi + F(1, 1) * xgamma;
  // const auto t_zz = F(1, 0) * zxi + F(1, 1) * zgamma;

  // Add to acceleration t_{zx} - t_{xz}
  // Note that when nu is 0, this is equivalent to
  //   t_{zx} - t_{xz} = 0
  // Meaning the stress tensor is symmetric and the couple stress
  //   contribution is zero.
  acceleration.acceleration(2) += factor * (t_zx - t_xz);
};

} // namespace medium
} // namespace specfem
