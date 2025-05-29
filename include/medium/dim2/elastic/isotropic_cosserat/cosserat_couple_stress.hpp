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

  // Compute inverse Jacobian elements
  const auto invD = static_cast<T>(1.0) / (xix * gammaz - xiz * gammax);
  const auto xxi = gammaz * invD;
  const auto xgamma = -gammax * invD;
  const auto zxi = -xiz * invD;
  const auto zgamma = xix * invD;

  // Compute transformed stresses
  // F(i, k) = F_{x_i, \xi_k} and x_i = [x,z], \xi_k = [\xi, \gamma]
  // t_{ij} = F_{i,k} * \partial x_j / \partial \xi_k
  const auto t_xx = F(0, 0) * xxi + F(0, 1) * xgamma;
  const auto t_zx = F(1, 0) * xxi + F(1, 1) * xgamma;
  const auto t_xz = F(0, 0) * zxi + F(0, 1) * zgamma;
  const auto t_zz = F(1, 0) * zxi + F(1, 1) * zgamma;

  // Add to acceleration
  acceleration.acceleration(2) -= factor * (t_zx - t_xz);
};

} // namespace medium
} // namespace specfem
